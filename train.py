# train.py -- train the CF model on the real data, report dev/test results, save the model.
# usage: python3 train.py [epochs]       (default 30)
# Uses data/best_params.json (from search_hparams.py) if it exists, otherwise the paper's values.
import sys, json, os, time
import numpy as np, scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cf_model import cosine_weights, CFModel, evaluate

D = "data"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
P = {"tau": -0.001, "k": 1000, "lambda": 0.01, "alpha": 0.1}          # paper's final values
if os.path.exists(f"{D}/best_params.json"):
    P = json.load(open(f"{D}/best_params.json"))
print("hyperparameters:", P)

R = sp.load_npz(f"{D}/R_train.npz").tocsc()
X_dev = sp.load_npz(f"{D}/X_dev.npz").tocsc()
X_test = sp.load_npz(f"{D}/X_test.npz").tocsc()
print("train", R.shape, "dev", X_dev.shape, "test", X_test.shape)

W = cosine_weights(R, tau=P["tau"], k=P["k"])
model = CFModel(W, lam=P["lambda"], alpha=P["alpha"])
t0 = time.time()
hist = model.fit(R, epochs=EPOCHS, batch=256, X_dev=X_dev,
                 log=lambda r: print({k: round(float(v), 5) for k, v in r.items()}, f"{time.time()-t0:.0f}s"))

dev, test = evaluate(model, X_dev), evaluate(model, X_test)
print("DEV ", dev)
print("TEST", test)
np.savez(f"{D}/model.npz", W=model.W, Theta=model.Theta, b=model.b)
json.dump({"params": P, "epochs": EPOCHS, "dev": dev, "test": test, "history": hist},
          open(f"{D}/final_results.json", "w"), indent=1)

os.makedirs("figs", exist_ok=True)
ep = [h["epoch"] for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, [h["loss"] for h in hist]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("Loss")
ax[1].plot(ep, [h["APTK"] for h in hist], label="model (dev APTK)")
ax[1].axhline(dev["APRK"], color="gray", ls="--", label="random (APRK)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("precision@3"); ax[1].set_title("Dev APTK"); ax[1].legend()
plt.tight_layout(); plt.savefig("figs/fig4_final_training.png", dpi=150)
print("saved data/model.npz, data/final_results.json, figs/fig4_final_training.png")