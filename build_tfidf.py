# build_tfidf.py
# counts.npz (concepts x trials, raw counts) -> TF-IDF matrix -> 8:1:1 column split.
#
# Input  (data/): counts.npz, vocab.json, doc_ids.json     (from 499's build_counts.py)
# Output (data/): R_train.npz, X_dev.npz, X_test.npz   TF-IDF, scipy sparse CSC, shape (m, n_split)
#                 vocab_filtered.json   kept concept IDs, in row order
#                 idf.npy               IDF of the kept concepts, aligned with vocab_filtered.json
#                 split_doc_ids.json    {"train": [...], "dev": [...], "test": [...]} nct_ids in column order
#
# U[i,j] = log10(1 + count_ij) * log10(N / df_i)      (Zhou 2020, section 4.1)

import json, os
import numpy as np
import scipy.sparse as sp

MIN_DOC_FRAC = 0.005     # keep concepts present in >= 0.5% of all trials
MIN_TERMS_PER_DOC = 2    # a doc needs >= 2 kept concepts, otherwise 50% masking leaves nothing to test
SEED = 42
D = "data"

counts = sp.load_npz(f"{D}/counts.npz").tocsr()
vocab = json.load(open(f"{D}/vocab.json"))
doc_ids = np.array(json.load(open(f"{D}/doc_ids.json")))
assert counts.shape == (len(vocab), len(doc_ids)), "counts.npz shape must be (concepts, trials)"
N = counts.shape[1]

# document frequency over ALL trials, then keep the frequent concepts
df = np.diff(counts.indptr)
keep = np.flatnonzero(df >= MIN_DOC_FRAC * N)
counts = counts[keep]
vocab_f = [vocab[i] for i in keep]
idf = np.log10(N / df[keep]).astype(np.float32)

# TF-IDF
U = counts.astype(np.float32)
U.data = np.log10(1.0 + U.data)
U = sp.diags(idf) @ U
U = sp.csc_matrix(U)

# drop trials with too few kept concepts
nnz_per_doc = np.diff(U.indptr)
ok = np.flatnonzero(nnz_per_doc >= MIN_TERMS_PER_DOC)
U = U[:, ok]
doc_ids = doc_ids[ok]
n = U.shape[1]

# random 8:1:1 split of the columns
perm = np.random.default_rng(SEED).permutation(n)
a, b = int(0.8 * n), int(0.9 * n)
parts = {"train": np.sort(perm[:a]), "dev": np.sort(perm[a:b]), "test": np.sort(perm[b:])}

os.makedirs(D, exist_ok=True)
for name, fname in [("train", "R_train"), ("dev", "X_dev"), ("test", "X_test")]:
    sp.save_npz(f"{D}/{fname}.npz", U[:, parts[name]])
json.dump(vocab_f, open(f"{D}/vocab_filtered.json", "w"))
np.save(f"{D}/idf.npy", idf)
json.dump({k: doc_ids[v].tolist() for k, v in parts.items()}, open(f"{D}/split_doc_ids.json", "w"))

print(f"kept concepts m = {len(vocab_f)}")
print(f"trials used     = {n} of {N} (dropped {N - n} with < {MIN_TERMS_PER_DOC} kept concepts)")
print("split sizes     =", {k: len(v) for k, v in parts.items()})
print(f"non-zero TF-IDF = {U.nnz}  (avg {U.nnz / n:.1f} per trial)")