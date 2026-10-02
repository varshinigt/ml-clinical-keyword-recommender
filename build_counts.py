"""
build_counts.py - scan every trial for MeSH concepts and save the count matrix.

Run from the repo root (the folder that contains data/).

Outputs (data/ for a full run, data/sample/ for a --n sample run):
    counts.npz    scipy sparse matrix, shape (concepts, trials), raw counts
    vocab.json    list of concept IDs, same order as the ROWS of counts.npz
    doc_ids.json  list of nct_ids,     same order as the COLUMNS of counts.npz

Usage:
    python build_counts.py --n 2000           # sample run first
    python build_counts.py                    # full run (all trials)
    python build_counts.py --workers 1        # no multiprocessing (debug)
    python build_counts.py --report_only      # re-print the report from saved files
    python build_counts.py --n 2000 --report_only
"""
import os
import json
import time
import random
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.sparse as sp

from trie_scanner import DATA_DIR, load_trie, trial_text

MESH_PATH = f"{DATA_DIR}/mesh_terms.json"
NAMES_PATH = f"{DATA_DIR}/mesh_names.json"
MIN_DOC_FRACTION = 0.005  # teammate keeps terms in >= 0.5% of trials

_trie = None  # one trie per worker process


def _init_worker(mesh_path):
    global _trie
    _trie = load_trie(mesh_path)


def _scan_one(text):
    return dict(_trie.scan_counts(text))


def load_trials_df():
    try:
        from load_data import load_trials  # teammate's loader
    except ImportError:
        import pandas as pd

        def load_trials():
            return pd.read_parquet(f"{DATA_DIR}/trials.parquet")
    return load_trials()


def scan_all(texts, workers):
    """Scan every text. Returns a list of {concept_id: count}, SAME ORDER as texts."""
    results = []
    total = len(texts)
    t0 = time.time()

    def progress(i):
        if i % 2000 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  scanned {i}/{total} trials  ({elapsed:.0f}s)", flush=True)

    if workers <= 1:
        _init_worker(MESH_PATH)
        for i, text in enumerate(texts, 1):
            results.append(_scan_one(text))
            progress(i)
    else:
        with Pool(workers, initializer=_init_worker, initargs=(MESH_PATH,)) as pool:
            # imap (not imap_unordered) keeps the order = column order
            for i, res in enumerate(pool.imap(_scan_one, texts, chunksize=100), 1):
                results.append(res)
                progress(i)
    return results


def build_matrix(results):
    """rows = concepts that appear at least once, columns = trials."""
    vocab = sorted({cid for r in results for cid in r})
    row_of = {cid: i for i, cid in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for j, r in enumerate(results):
        for cid, c in r.items():
            rows.append(row_of[cid])
            cols.append(j)
            vals.append(c)
    counts = sp.csr_matrix((vals, (rows, cols)),
                           shape=(len(vocab), len(results)), dtype=np.int32)
    return counts, vocab


def save_outputs(counts, vocab, doc_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sp.save_npz(f"{out_dir}/counts.npz", counts)
    with open(f"{out_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(f"{out_dir}/doc_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    print(f"Saved counts.npz, vocab.json, doc_ids.json to {out_dir}/")


def load_outputs(out_dir):
    counts = sp.load_npz(f"{out_dir}/counts.npz").tocsr()
    with open(f"{out_dir}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(f"{out_dir}/doc_ids.json", encoding="utf-8") as f:
        doc_ids = json.load(f)
    return counts, vocab, doc_ids


def report(counts, vocab, doc_ids, top=100):
    """Sanity checks + the top concepts by number of trials (to spot junk)."""
    m, n = counts.shape
    assert m == len(vocab), "rows != len(vocab)"
    assert n == len(doc_ids), "columns != len(doc_ids)"
    assert len(set(doc_ids)) == n, "duplicate nct_ids in doc_ids"

    with open(NAMES_PATH, encoding="utf-8") as f:
        names = json.load(f)

    doc_freq = counts.getnnz(axis=1)           # trials containing each concept
    per_trial = counts.getnnz(axis=0)          # concepts found in each trial
    keep = int((doc_freq >= MIN_DOC_FRACTION * n).sum())

    print("\n=== REPORT ===")
    print(f"shape (concepts x trials): {m} x {n}")
    print(f"non-zero cells: {counts.nnz}  (sparsity {100 * (1 - counts.nnz / (m * n)):.2f}%)")
    print(f"avg concepts per trial: {per_trial.mean():.1f}  (median {np.median(per_trial):.0f})")
    print(f"trials with ZERO concepts: {int((per_trial == 0).sum())}")
    print(f"concepts in >= {MIN_DOC_FRACTION:.1%} of trials "
          f"(what the model will keep): {keep}")

    print(f"\nTop {top} concepts by number of trials containing them:")
    order = np.argsort(-doc_freq)[:top]
    for rank, i in enumerate(order, 1):
        print(f"{rank:3d}. {doc_freq[i]:6d} trials ({doc_freq[i] / n:6.1%}) | "
              f"{vocab[i]} {names.get(vocab[i], '?')}")
    print("\nAny generic or junk concept here -> add to BLOCK_CONCEPTS in "
          "trie_scanner.py and rescan.")


def spot_check(counts, vocab, texts, k=3):
    """Re-scan k random trials and compare with the saved matrix columns."""
    trie = load_trie(MESH_PATH)
    for j in random.sample(range(len(texts)), min(k, len(texts))):
        expected = dict(trie.scan_counts(texts[j]))
        col = counts[:, j].tocoo()
        saved = {vocab[i]: int(v) for i, v in zip(col.row, col.data)}
        assert expected == saved, f"MISMATCH in column {j}"
    print(f"Spot check passed on {min(k, len(texts))} random trials.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="only scan the first n trials (sample run)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    out_dir = DATA_DIR if args.n is None else f"{DATA_DIR}/sample"

    if args.report_only:
        counts, vocab, doc_ids = load_outputs(out_dir)
        report(counts, vocab, doc_ids, args.top)
        return

    df = load_trials_df()
    if args.n is not None:
        df = df.head(args.n)
    df = df.reset_index(drop=True)

    doc_ids = df["nct_id"].tolist()
    texts = trial_text(df).tolist()
    print(f"Scanning {len(texts)} trials with {args.workers} worker(s)...")

    results = scan_all(texts, args.workers)
    counts, vocab = build_matrix(results)
    save_outputs(counts, vocab, doc_ids, out_dir)

    counts2, vocab2, doc_ids2 = load_outputs(out_dir)   # reload from disk
    assert (counts != counts2).nnz == 0 and vocab == vocab2 and doc_ids == doc_ids2
    spot_check(counts2, vocab2, texts)
    report(counts2, vocab2, doc_ids2, args.top)


if __name__ == "__main__":
    main()
