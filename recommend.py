# recommend.py  -- model side of the demo (used by 499's demo_query.py)
#
#   from recommend import recommend
#   recommend({"D001943": 2, "D003920": 1}, top_k=5)  ->  [("D009369", 0.83), ...]
#
# Input : dict  concept_id -> count in the query (from 499's query_to_counts)
# Output: list of (concept_id, score), best first. Concepts already in the query are not returned.
# Needs (data/): model.npz (from train.py), vocab_filtered.json, idf.npy

import json
import numpy as np

_D = "data"
_m = np.load(f"{_D}/model.npz")
_WT = (_m["W"] * _m["Theta"]).astype(np.float32)
_b = _m["b"].astype(np.float32).ravel()
_vocab = json.load(open(f"{_D}/vocab_filtered.json"))
_idf = np.load(f"{_D}/idf.npy").astype(np.float32)
_index = {c: i for i, c in enumerate(_vocab)}


def recommend(counts, top_k=5):
    x = np.zeros(len(_vocab), dtype=np.float32)
    for cid, c in counts.items():
        i = _index.get(cid)
        if i is not None and c > 0:                    # concepts outside the kept vocabulary are ignored
            x[i] = np.log10(1.0 + c) * _idf[i]         # same TF-IDF as build_tfidf.py
    if not x.any():
        return []
    scores = np.maximum(_WT @ x + _b, 0.0)             # R_hat = ReLU((W * Theta) x + b)
    scores[x > 0] = -np.inf                            # recommend NEW keywords only
    top = np.argsort(-scores)[:top_k]
    return [(_vocab[i], float(scores[i])) for i in top if scores[i] > 0]