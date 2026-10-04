# baseline_cooccurrence.py -- co-occurrence baseline for the keyword recommender.
# For a (masked) trial, every candidate concept i is scored with
#     score(i) = sum over the concepts j present in the trial of  P(i in trial | j in trial),
# where P is estimated from the training trials (share of trials containing j that also contain i).
# Evaluated with exactly the same masking and metrics as the model (cf_model.evaluate).
# usage: python3 baseline_cooccurrence.py
import numpy as np, scipy.sparse as sp
from cf_model import evaluate

D = "data"
R = sp.load_npz(f"{D}/R_train.npz").tocsr()
B = (R > 0).astype(np.float32)                    # which concept occurs in which training trial
C = (B @ B.T).toarray()                           # C[i, j] = number of trials containing both i and j
np.fill_diagonal(C, 0)
n_j = np.asarray(B.sum(axis=1)).ravel()           # number of trials containing concept j
P = C / np.maximum(n_j[None, :], 1)               # P[i, j] = P(i | j)

class CoOccurrence:
    def predict(self, X):                         # X: concepts x trials (masked input)
        return P @ (X > 0).astype(np.float32)

for name in ["dev", "test"]:
    X = sp.load_npz(f"{D}/X_{name}.npz").tocsc()
    r = evaluate(CoOccurrence(), X)
    print(f"{name:5s} co-occurrence baseline: APTK = {r['APTK']:.4f}   (random APRK = {r['APRK']:.4f})")
