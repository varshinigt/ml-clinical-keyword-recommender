# cf_model.py
# Collaborative-filtering keyword recommender (Zhou, CS229 2020), ReLU model + L2 reg.
#
# Convention: matrices are (terms m) x (documents n), like the paper.
#   R_hat = ReLU( (W * Theta) @ R + b )
#   W_if  = exp( tau * (1 - cos(R_i, R_f))^k ),  W_ii = 0
#   loss  = 1/(2n) * sum (ReLU(Z) - R)^2  + (lam/2) * sum Theta^2

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------- weights W
def cosine_weights(R, tau=-0.001, k=1000):
    """Neighbour weight matrix W (m x m) from cosine similarity of term rows."""
    R = sp.csr_matrix(R, dtype=np.float32)
    norms = np.sqrt(R.multiply(R).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    Rn = sp.diags(1.0 / norms) @ R
    cos = (Rn @ Rn.T).toarray()                    # in [0,1] because R >= 0
    cos = np.clip(cos, 0.0, 1.0)
    W = np.exp(tau * (1.0 - cos) ** k).astype(np.float32)
    np.fill_diagonal(W, 0.0)                       # remove self interaction
    return W


# ---------------------------------------------------------------- model
class CFModel:
    def __init__(self, W, lam=0.01, alpha=0.1):
        self.W = W.astype(np.float32)
        m = W.shape[0]
        self.Theta = np.zeros((m, m), dtype=np.float32)
        self.b = np.zeros((m, 1), dtype=np.float32)
        self.lam, self.alpha = lam, alpha

    def _Z(self, B):
        return (self.W * self.Theta) @ B + self.b   # pre-activation

    def predict(self, X):
        """X: dense (m x n). Returns ReLU((W*Theta) X + b)."""
        return np.maximum(self._Z(X), 0.0)

    def grads(self, B):
        """Gradients on a dense mini-batch B (m x |B|), averaged over |B|."""
        Z = self._Z(B)
        relu_d = (Z >= 0).astype(np.float32)        # ReLU'(0) = 1 (as in the paper)
        E = (Z - B) * relu_d                        # (Z - B) * ReLU'(Z)
        nb = B.shape[1]
        gTheta = self.W * (E @ B.T) / nb + self.lam * self.Theta
        gb = E.sum(axis=1, keepdims=True) / nb
        return gTheta, gb

    def loss(self, R, batch=2000):
        """J on a sparse/dense matrix R (m x n), computed in column chunks."""
        total, n = 0.0, R.shape[1]
        for s in range(0, n, batch):
            B = _dense(R, s, min(s + batch, n))
            D = self.predict(B) - B
            total += 0.5 * float((D ** 2).sum())
        return total / n + 0.5 * self.lam * float((self.Theta ** 2).sum())

    def fit(self, R, epochs=30, batch=256, seed=0, X_dev=None, log=print):
        rng = np.random.default_rng(seed)
        n = R.shape[1]
        history = []
        for ep in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = np.sort(order[s:s + batch])
                B = _dense_cols(R, idx)
                gT, gb = self.grads(B)
                self.Theta -= self.alpha * gT
                self.b -= self.alpha * gb
            row = {"epoch": ep + 1, "loss": self.loss(R)}
            if X_dev is not None:
                row.update(evaluate(self, X_dev))
            history.append(row)
            log(row)
        return history


def _dense(R, a, b):
    M = R[:, a:b]
    return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32)

def _dense_cols(R, idx):
    M = R[:, idx]
    return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32)


# ---------------------------------------------------------------- evaluation
def mask_matrix(X, frac=0.5, seed=0):
    """Randomly set `frac` of each column's positive entries to 0."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    for j in range(X.shape[1]):
        pos = np.flatnonzero(X[:, j] > 0)
        if len(pos) == 0:
            continue
        k = int(round(frac * len(pos)))
        Xm[rng.choice(pos, size=k, replace=False), j] = 0.0
    return Xm


def evaluate(model, X, K=3, frac=0.5, seed=0, chunk=2000):
    """MSE, MSER, APTK and APRK on X (sparse or dense, m x n)."""
    rng = np.random.default_rng(seed + 1)
    n = X.shape[1]
    sse = 0.0
    n_relevant = 0
    prec_model, prec_rand, n_docs = 0.0, 0.0, 0
    for s in range(0, n, chunk):
        Xd = _dense(X, s, min(s + chunk, n))
        Xm = mask_matrix(Xd, frac, seed=seed + s)
        Xh = model.predict(Xm)
        sse += float(((Xh - Xd) ** 2).sum())
        n_relevant += int(((Xd > 0) | (Xh > 0)).sum())
        # candidates = zero in masked input but positive in prediction
        cand = (Xm == 0) & (Xh > 0)
        score = np.where(cand, Xh, -np.inf)
        top = np.argsort(-score, axis=0)[:K]                 # K x docs
        for j in range(Xd.shape[1]):
            if not (Xd[:, j] > 0).any():
                continue
            n_docs += 1
            valid = [t for t in top[:, j] if np.isfinite(score[t, j])]
            prec_model += sum(Xd[t, j] > 0 for t in valid) / K
            zeros = np.flatnonzero(Xm[:, j] == 0)
            if len(zeros) >= K:
                pick = rng.choice(zeros, size=K, replace=False)
                prec_rand += (Xd[pick, j] > 0).sum() / K
    total = X.shape[0] * n
    return {"MSE": float(sse / total), "MSER": float(sse / max(n_relevant, 1)),
            "APTK": float(prec_model / max(n_docs, 1)), "APRK": float(prec_rand / max(n_docs, 1))}