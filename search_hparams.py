# search_hparams.py -- hyperparameter search + plots (follows the paper, section 5.2)
#   stage 1: learning rate alpha   (tau=-1, k=4, lambda=1e-6 fixed)  -> loss curves
#   stage 2: grid over (tau, k) pairs x lambda, using the best alpha  -> dev APTK / MSER
# usage:  python3 search_hparams.py          (full search, ~15-25 min)
#         python3 search_hparams.py --fast   (tiny test run, ~1 min)
# output: data/best_params.json, data/search_results.csv, figs/*.png
import sys, json, csv, os, time
import numpy as np, scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cf_model import cosine_weights, CFModel, evaluate

FAST = "--fast" in sys.argv
D = "data"
os.makedirs("figs", exist_ok=True)

R = sp.load_npz(f"{D}/R_train.npz").tocsc()
X_dev = sp.load_npz(f"{D}/X_dev.npz").tocsc()

# the search runs on a random subset of the training trials to keep it quick
n_search = 3000 if FAST else 10000
idx = np.sort(np.random.default_rng(0).choice(R.shape[1], n_search, replace=False))
Rs = R[:, idx]
Xd = X_dev[:, :1000] if FAST else X_dev

ALPHAS = [0.001, 0.01, 0.1, 1.0]
E1 = 3 if FAST else 10
PAIRS = [(-10, 4), (-1, 4), (-1, 1), (-1, 16), (-0.001, 1000)]
LAMS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
if FAST:
    PAIRS, LAMS = PAIRS[:2], LAMS[:2]
E2 = 2 if FAST else 8

def run(tau, k, lam, alpha, epochs, W_cache={}):
    key = (tau, k)
    if key not in W_cache:
        W_cache[key] = cosine_weights(Rs, tau=tau, k=k)
    m = CFModel(W_cache[key], lam=lam, alpha=alpha)
    hist = m.fit(Rs, epochs=epochs, batch=256, log=lambda r: None)
    losses = [h["loss"] for h in hist]
    if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(m.Theta)):
        return losses, {"MSE": np.nan, "MSER": np.nan, "APTK": -1.0, "APRK": np.nan}
    return losses, evaluate(m, Xd)

def area(tau, k):                       # area under exp(tau (1-x)^k) on [0,1], as in the paper
    x = np.linspace(0, 1, 1001)
    y = np.exp(tau * (1 - x) ** k)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))

# ---------------- stage 1: learning rate
print("== stage 1: learning rate ==")
curves, a_res = {}, []
for a in ALPHAS:
    t0 = time.time()
    losses, met = run(-1, 4, 1e-6, a, E1)
    curves[a] = losses
    a_res.append((a, met))
    print(f"alpha={a:<6} final loss={losses[-1]:.4f}  APTK={met['APTK']:.4f}  MSER={met['MSER']:.5f}  ({time.time()-t0:.0f}s)")
best_alpha = max(a_res, key=lambda r: r[1]["APTK"])[0]
print("best alpha:", best_alpha)

plt.figure(figsize=(6, 4))
for a, l in curves.items():
    plt.plot(range(1, len(l) + 1), l, marker="o", label=f"alpha={a}")
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("Searching the learning rate")
plt.legend(); plt.tight_layout(); plt.savefig("figs/fig1_learning_rate.png", dpi=150); plt.close()

# ---------------- stage 2: tau, k, lambda
print("== stage 2: tau, k, lambda ==")
rows = []
for (tau, k) in PAIRS:
    for lam in LAMS:
        t0 = time.time()
        _, met = run(tau, k, lam, best_alpha, E2)
        rows.append({"tau": tau, "k": k, "lambda": lam, "area": round(area(tau, k), 4), **{m: float(v) for m, v in met.items()}})
        print(f"tau={tau:<7} k={k:<5} lam={lam:<7} APTK={met['APTK']:.4f} MSER={met['MSER']:.5f} ({time.time()-t0:.0f}s)")
with open(f"{D}/search_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

best = max(rows, key=lambda r: r["APTK"])
params = {"tau": best["tau"], "k": best["k"], "lambda": best["lambda"], "alpha": best_alpha}
json.dump(params, open(f"{D}/best_params.json", "w"))
print("BEST:", params, " dev APTK =", round(best["APTK"], 4), " MSER =", round(best["MSER"], 5))

# plots: APTK vs lambda per (tau,k), and the W kernel curves
plt.figure(figsize=(6, 4))
for (tau, k) in PAIRS:
    r = [x for x in rows if x["tau"] == tau and x["k"] == k]
    plt.plot([x["lambda"] for x in r], [x["APTK"] for x in r], marker="o", label=f"tau={tau}, k={k}")
plt.xscale("log"); plt.xlabel("lambda"); plt.ylabel("dev APTK (K=3)"); plt.title("Searching tau, k and lambda")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("figs/fig2_tau_k_lambda.png", dpi=150); plt.close()

x = np.linspace(0, 1, 200)
plt.figure(figsize=(6, 4))
for (tau, k) in PAIRS:
    plt.plot(x, np.exp(tau * (1 - x) ** k), label=f"tau={tau}, k={k}")
plt.xlabel("cosine similarity"); plt.ylabel("neighbour weight W"); plt.title("Shape of the neighbour weight")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("figs/fig3_weight_kernel.png", dpi=150); plt.close()
print("saved figs/fig1_learning_rate.png, fig2_tau_k_lambda.png, fig3_weight_kernel.png")