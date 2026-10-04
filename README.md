# Collaborative Filtering for Keyword Recommendation on Clinical Trial Records

**UE24CS352A – Machine Learning, Mini-Project (Problem Statement No. 14)**

| Team member | SRN |
|---|---|
| Varshini A | PES1UG24CS517 |
| Tejasvi K S | PES1UG24CS499 |

We reproduce and run the keyword-recommending model from Xiao Zhou's CS229 project
*"Collaborative Filtering on Keywords Recommendation for Clinical Trial Records"*
([report](https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf),
[poster](https://cs229.stanford.edu/proj2020spr/poster/Zhou.pdf)).

## What the project does

A researcher searching for similar clinical trials often forgets relevant keywords. Given the
medical keywords found in a query, our model **recommends further keywords** that usually appear
together with them in clinical trial records.

1. **Data**: 50,000 trial records from ClinicalTrials.gov (API v2): titles, summary, detailed description, conditions, eligibility.
2. **Keywords**: medical terms are found with the NLM MeSH 2026 vocabulary (31,110 headings, 267,007 terms and synonyms) using a word-level Trie. All synonyms map to one concept ID.
3. **Utility matrix** `U` (concepts x trials) with TF-IDF: `U_ij = log10(1 + count_ij) * log10(N / df_i)`. Concepts present in at least 0.5% of trials are kept (1,190); the columns are split randomly 8:1:1 into train / dev / test.
4. **Model** (collaborative filtering): `R_hat = ReLU((W ⊙ Θ) R + b)`, where `W_if = exp(τ (1 - cos(R_i, R_f))^k)` is a fixed neighbour weight (`W_ii = 0`) and `Θ`, `b` are learned by mini-batch gradient descent with an L2 penalty `λ`.
5. **Evaluation**: 50% of the non-zero entries of each dev/test trial are masked and the model predicts them back. We report MSE, MSER, and APTK (precision of the top K=3 recommended new keywords) against a random baseline APRK.
6. **Demo**: a typed query is scanned for MeSH concepts, and the model suggests further keywords.

## Results (test set, 4,974 trials)

Selected hyperparameters (by dev APTK): `tau = -0.001, k = 1000, lambda = 0.01, alpha = 0.1`, 30 epochs.

| | MSE | MSER | APTK (K=3) |
|---|---|---|---|
| Random picks (test) | | | 0.0088 |
| Most popular concepts (test, `baseline_popularity.py`) | | | 0.1529 |
| Co-occurrence baseline (test, `baseline_cooccurrence.py`) | | | 0.2249 |
| Our model, dev (4,973 trials) | 0.00794 | 0.02186 | 0.3335 |
| **Our model, test (4,974 trials)** | **0.00814** | **0.02236** | **0.3390** |

The model's top-3 precision is more than 30 times that of random picks, more than twice that of always recommending the most popular concepts, and about 1.5 times that of a co-occurrence baseline (it scores each concept by summing, over the query concepts, the share of training trials with that concept that also contain the candidate). In our search the top four settings differ by less than 0.003 in dev APTK (0.3278 to 0.3307), so we do not claim one of them is clearly best. The full search table is in `results/search_results.csv`, plots are in `results/figs/`. These numbers are for the raw model output; the display filter used in the demo (see below) is not applied during evaluation.

For reference, on its test set the paper reports an MSE of 0.0081 and an APTK of 0.479 with 947 terms, and an MSE of 0.0010 and an APTK of 0.503 with 15,071 terms. Our MSE is similar to the 947-term run (MSE depends on the number of terms, so the runs are not directly comparable), our APTK is lower. We used a 50,000-trial sample, 1,190 terms, and our own extraction with a blocklist of generic concepts. We did not run experiments to find out which of these causes the difference.

## Repository layout

```
download_trials.py    download the 50,000 trial records from the ClinicalTrials.gov API
split_data.py         split the dataset into 4 parts (GitHub file-size limit) -> data_parts/
load_data.py          load_trials(): merges data_parts/ back into one DataFrame
parse_mesh.py         MeSH descriptor XML -> data/mesh_terms.json, data/mesh_names.json
trie_scanner.py       word-level Trie + text scanner, with the blocklist of generic terms/concepts
build_counts.py       scan all trials -> data/counts.npz (concepts x trials), vocab.json, doc_ids.json
build_tfidf.py        counts -> TF-IDF, keep terms in >= 0.5% of trials, 8:1:1 split
cf_model.py           the model (W, forward, gradients), masking and metrics (MSE, MSER, APTK, APRK)
test_cf.py            gradient check + training test on synthetic data
search_hparams.py     learning-rate search and (tau, k, lambda) grid search, plots
train.py              final training with the best hyperparameters, saves data/model.npz
baseline_popularity.py  "always recommend the most popular concepts" baseline
baseline_cooccurrence.py  co-occurrence baseline (sum of P(candidate | query concept))
recommend.py          recommend(counts_dict, top_k): the model side of the demo
demo_query.py         typed query -> MeSH concepts -> model -> suggested keywords
data_parts/           the trial dataset in 4 parquet parts
data/                 derived files needed to train and run the demo (see "Data format")
results/              search_results.csv, best_params.json, final_results.json, figs/
```

## Setup

Tested with Python 3.13 on macOS.

```bash
git clone https://github.com/varshinigt/ml-clinical-keyword-recommender
cd ml-clinical-keyword-recommender
pip3 install -r requirements.txt
```

The raw MeSH file (`desc2026.xml`, 298 MB) is not in the repository. The derived files needed to train the model and run the demo are in `data/`. The trials are loaded from `data_parts/` through `load_data.load_trials()`.

## How to run

All commands are run from the repository root.

### Quick start (uses the files already in `data/`)

```bash
python3 trie_scanner.py --dummy_only   # self-tests of the keyword scanner
python3 test_cf.py                     # gradient check + synthetic training test (about 1 min)
python3 build_tfidf.py                 # TF-IDF + 8:1:1 split (m = 1190, 49,732 trials)
python3 search_hparams.py --fast       # tiny test of the search (about 1 min)
python3 train.py 30                    # final training and evaluation (about 3 min)
python3 baseline_popularity.py         # popularity baseline
python3 baseline_cooccurrence.py       # co-occurrence baseline
```

`train.py` prints the dev and test metrics and writes `data/model.npz`, `data/final_results.json` and `figs/fig4_final_training.png`.

### Demo

Needs `data/model.npz`, `data/vocab_filtered.json` and `data/idf.npy` (all in the repository).

```bash
python3 demo_query.py "eczema in children, topical treatment"
python3 demo_query.py "type 2 diabetes insulin" --top_k 10
python3 demo_query.py                   # interactive mode, an empty line quits
python3 demo_query.py --examples        # four ready-made example queries
python3 demo_query.py "psoriasis" --raw # raw model output (no display filter)
```

The query goes through the same scanner as the training data, so it gets the same cleaning and blocklist. The detected concepts are passed to `recommend()`, which converts them to TF-IDF with the training IDF, runs the model, removes the concepts already in the query and returns the highest-scoring ones. Generic concepts (Adult, Aged, Child, Women, Neoplasms, ...) are hidden from the displayed suggestions by default because they co-occur with almost every trial; they are never hidden from the detected keywords, and `--raw` shows everything.

### Rebuild everything from scratch

```bash
# 1. dataset (50,000 trials). Already provided in data_parts/, so this step is optional.
python3 download_trials.py            # writes trials.parquet
python3 load_data.py                  # prints (50000, 7), merges data_parts/

# 2. MeSH vocabulary
cd data && curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml && cd ..
python3 parse_mesh.py                 # -> data/mesh_terms.json, data/mesh_names.json

# 3. keyword extraction -> utility matrix of raw counts
python3 trie_scanner.py               # self-tests, then scans 100 trials and prints the top 50 concepts
python3 build_counts.py --n 2000      # sample run, writes to data/sample/
python3 build_counts.py               # full run -> data/counts.npz, vocab.json, doc_ids.json
                                      # (use --workers 1 if multiprocessing causes problems)

# 4. TF-IDF, split, hyperparameter search, final model
python3 build_tfidf.py
python3 search_hparams.py             # about 10 min -> data/best_params.json, figs/
python3 train.py 30
```

Expected report after the full count run: shape 17337 x 50000, about 99.8% sparse, about 31 concepts per trial, 1190 concepts in at least 0.5% of trials.

## Data format (shared between scripts)

All matrices are **terms x documents** (rows = MeSH concepts, columns = trials). A concept ID is the MeSH descriptor UI string, for example `D001943`.

| File | Content |
|---|---|
| `data_parts/` | the 50,000 trials, loaded with `load_data.load_trials()` (columns `nct_id`, `brief_title`, `official_title`, `brief_summary`, `detailed_desc`, `conditions`, `eligibility`) |
| `data/mesh_terms.json` | `{"lowercased term or synonym": "concept ID"}` |
| `data/mesh_names.json` | `{"concept ID": "main heading"}` |
| `data/counts.npz` | scipy sparse (17,337 x 50,000), raw int32 counts |
| `data/vocab.json`, `data/doc_ids.json` | concept IDs in row order, nct_ids in column order of `counts.npz` |
| `data/R_train.npz`, `X_dev.npz`, `X_test.npz` | TF-IDF, scipy sparse CSC (1,190 x 39,785 / 4,973 / 4,974) |
| `data/vocab_filtered.json`, `data/idf.npy` | kept concept IDs and their IDF values, same order |
| `data/model.npz` | learned `W`, `Theta`, `b` |

## Design decisions and limitations

- **Keyword matching is exact.** Text and MeSH terms are lowercased, punctuation becomes spaces, and the longest matching term wins. There is no stemming or spelling correction: "type 2 diabetes" and "heart attack" are recognised, "diabetes" alone and "alzheimers" are not (use "diabetes mellitus" and "alzheimer's disease"). Negation and context are ignored ("no history of stroke" still counts Stroke).
- **Generic concepts are blocked.** Concepts such as *Humans*, *Patients*, *Informed Consent* or reproductive-eligibility boilerplate (`BLOCK_CONCEPTS`, 97 concepts) and ordinary English words that are MeSH synonyms (`BLOCK_TERMS`, 5 terms) are removed so they are not counted as keywords. The lists were built by inspecting scan results; after editing them, rerun `build_counts.py` and everything after it. This makes the task harder than predicting very frequent terms.
- **Short terms (under 3 characters) are dropped**, because strings like "A" cause false matches.
- **Trials with fewer than 2 kept concepts are removed** (268 of 50,000), because 50% masking needs at least one remaining concept to predict from.
- **Hyperparameter search runs on a 10,000-trial subset** of the training set, with 10 epochs for the learning-rate search and 8 epochs per setting in the (tau, k, lambda) grid, for speed. The final model is trained on all training trials.
- **With the selected `tau = -0.001, k = 1000`, W is about 1 for almost every pair of concepts**, so the neighbour weighting has little effect, which is also what the paper found (its final values are the same `tau` and `k`).
- **MSER caveat.** MSER divides by the number of cells that are non-zero in the truth or the prediction, so it depends on how sparse the predictions are. We compare models mainly with MSE and APTK.
- **Demo suggestions are usually sensible near the top and sometimes noisy further down.** Only the 1,190 frequent concepts can be recommended.
- The dataset is the first 50,000 records returned by the ClinicalTrials.gov API on 2 Oct 2026 (not a random sample of the registry).


