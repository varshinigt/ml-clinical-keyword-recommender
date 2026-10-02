# baseline_popularity.py -- sanity baseline: recommend the globally most frequent concepts.
# For every dev/test trial, the same 50% masking as in cf_model.evaluate is used; the "model"
# always scores concept i with the fraction of training trials that contain it.
# usage: python3 baseline_popularity.py
import numpy as np, scipy.sparse as sp
from cf_model import evaluate

D = "data"
R = sp.load_npz(f"{D}/R_train.npz").tocsc()
pop = np.asarray((R > 0).mean(axis=1)).ravel().astype(np.float32)[:, None]

class Popularity:
    def predict(self, X):
        return np.repeat(pop, X.shape[1], axis=1)

for name in ["dev", "test"]:
    X = sp.load_npz(f"{D}/X_{name}.npz").tocsc()
    r = evaluate(Popularity(), X)
    print(f"{name:5s} popularity baseline: APTK = {r['APTK']:.4f}   (random APRK = {r['APRK']:.4f})")