# ml-clinical-keyword-recommender

## Keyword extraction, count matrix and demo

These parts turn raw clinical-trial text into a MeSH concept x trial matrix and
provide the query demo.

### Files
| File | Purpose |
|---|---|
| `trie_scanner.py` | Loads MeSH terms into a word-level Trie and finds MeSH concepts in text (longest match). Contains the blocklist of generic or junk concepts. |
| `build_counts.py` | Scans every trial and saves the sparse count matrix. |
| `demo_query.py` | Typed query -> MeSH concepts -> model -> suggested keywords. |

### Data layout (`data/`)
| File | Content |
|---|---|
| `trials.parquet` | 50,000 trials (loaded through `load_data.load_trials`) |
| `mesh_terms.json` | `{"lowercased term or synonym": "concept ID"}` (from `parse_mesh.py`, MeSH 2026) |
| `mesh_names.json` | `{"concept ID": "main heading"}` |
| `counts.npz` | scipy sparse matrix, shape (concepts, trials), raw counts |
| `vocab.json` | concept IDs in the row order of `counts.npz` |
| `doc_ids.json` | nct_ids in the column order of `counts.npz` |

### Setup
    pip install -r requirements.txt

### Run (from the repository root)
    python trie_scanner.py                 # self-tests, then scans 100 trials and prints the top 50 concepts
    python build_counts.py --n 2000        # sample run, writes to data/sample/
    python build_counts.py                 # full run, writes counts.npz, vocab.json, doc_ids.json to data/

Expected report after the full run: shape 17337 x 50000, about 99.8% sparse,
about 31 concepts per trial, 1190 concepts in at least 0.5% of trials.
Use `--workers 1` if multiprocessing causes problems (the scan still takes under a minute).

### Demo
After the model has been trained (`data/model.npz`, `data/vocab_filtered.json`, `data/idf.npy`):

    python demo_query.py "eczema in children, topical treatment"
    python demo_query.py "type 2 diabetes insulin" --top_k 10
    python demo_query.py                   # interactive mode, empty line quits
    python demo_query.py --examples        # four ready-made example queries
    python demo_query.py "psoriasis" --raw # raw model output (no display filter)

Generic concepts (Adult, Aged, Child, Women, Neoplasms, ...) are hidden from the
displayed suggestions by default because they co-occur with almost every trial.
They are never hidden from the detected keywords. `--raw` shows everything.

### Notes and limitations
- Matching is exact against MeSH names and synonyms: no stemming, no spelling
  correction. "type 2 diabetes" and "heart attack" are recognised; "diabetes" alone
  and "alzheimers" are not (use "diabetes mellitus", "alzheimer's disease").
- The blocklist in `trie_scanner.py` (`BLOCK_CONCEPTS`, `BLOCK_TERMS`) was built by
  inspecting scan results. After editing it, rerun `build_counts.py`.
- Negation and context are not handled ("no history of stroke" still counts Stroke).
```

---