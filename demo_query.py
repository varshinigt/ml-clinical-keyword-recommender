"""
demo_query.py - type a query, get recommended MeSH keywords.

HOW TO RUN (from the repo root, with the data/ files in place):

    python demo_query.py "eczema in children, topical treatment"
    python demo_query.py "type 2 diabetes insulin" --top_k 10
    python demo_query.py                  # interactive: type queries, empty line quits
    python demo_query.py --examples       # run a few ready-made example queries
    python demo_query.py "psoriasis" --raw   # raw model output, nothing hidden

Needs in data/: mesh_terms.json, mesh_names.json, vocab_filtered.json,
idf.npy, model.npz. And recommend.py in the repo root.

Pipeline:
    query text -> trie_scanner (MeSH concepts + counts, same blocklists as the
    training data) -> recommend() from the model -> concept names printed.

Display filter: very generic concepts (Adult, Aged, Child, Women, Neoplasms ...)
co-occur with almost everything, so they are hidden from the SUGGESTIONS
(never from the detected keywords). Use --raw to see the unfiltered model output.
"""
import sys
import json
import argparse

from trie_scanner import DATA_DIR, load_trie

NAMES_PATH = f"{DATA_DIR}/mesh_names.json"
VOCAB_PATH = f"{DATA_DIR}/vocab_filtered.json"

# Generic concepts hidden from the displayed suggestions (see docstring).
DEMO_HIDE = {
    "D000328",  # Adult
    "D000368",  # Aged
    "D002648",  # Child
    "D000293",  # Adolescent
    "D014930",  # Women
    "D008571",  # Men
    "D011247",  # Pregnancy
    "D009369",  # Neoplasms
    "D006967",  # Hypersensitivity
}

EXAMPLE_QUERIES = [
    "eczema in children, topical treatment",
    "type 2 diabetes insulin",
    "breast cancer chemotherapy",
    "asthma in adults, corticosteroids",
]

_trie = None
_names = None


def _get_trie():
    global _trie
    if _trie is None:
        _trie = load_trie()
    return _trie


def _get_names():
    global _names
    if _names is None:
        with open(NAMES_PATH, encoding="utf-8") as f:
            _names = json.load(f)
    return _names


def _model_vocab():
    """Concept IDs the model knows (vocab_filtered.json), or None if missing."""
    try:
        with open(VOCAB_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return None


def query_to_counts(text):
    """Query text -> {concept_id: count}, e.g. {"D001943": 2, "D003920": 1}."""
    return dict(_get_trie().scan_counts(text))


def run_query(text, top_k=5, raw=False):
    """Print detected keywords and suggestions. Returns the list of
    (concept_id, score) from recommend()."""
    try:
        from recommend import recommend  # teammate's model code
    except ImportError:
        sys.exit("Could not import recommend.py. Put it in the repo root "
                 "(next to demo_query.py) and run from there.")

    names = _get_names()
    name = lambda cid: names.get(cid, cid)

    counts = query_to_counts(text)
    print(f'\nQuery: "{text}"')

    if not counts:
        print("\nNo MeSH keywords were found in this query. "
              "Try more specific medical terms (e.g. a disease or drug name).")
        return []

    known = _model_vocab()
    used = {c: n for c, n in counts.items() if known is None or c in known}
    ignored = [c for c in counts if c not in used]

    print("\nDetected keywords:")
    for cid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        tag = "" if cid in used else "  (too rare for the model, ignored)"
        print(f"  - {name(cid)} ({cid}) x{n}{tag}")

    if not used:
        print("\nNone of the detected keywords are in the model's vocabulary, "
              "so there are no suggestions for this query.")
        return []

    if raw:
        results = recommend(counts, top_k=top_k)
    else:  # ask for extra, drop generic concepts, keep the first top_k
        results = [(c, sc) for c, sc in recommend(counts, top_k=top_k + len(DEMO_HIDE))
                   if c not in DEMO_HIDE][:top_k]
    if not results:
        print("\nThe model had no suggestions for this query.")
        return []

    print("\nSuggested additional keywords:")
    for rank, (cid, score) in enumerate(results, 1):
        print(f"  {rank}. {name(cid)} ({cid})  score {score:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Recommend MeSH keywords for a query.")
    ap.add_argument("query", nargs="?", help="query text (omit for interactive mode)")
    ap.add_argument("--top_k", type=int, default=5, help="number of suggestions")
    ap.add_argument("--raw", action="store_true",
                    help="show the raw model output (do not hide generic concepts)")
    ap.add_argument("--examples", action="store_true",
                    help="run the built-in example queries")
    args = ap.parse_args()

    if args.examples:
        for q in EXAMPLE_QUERIES:
            run_query(q, args.top_k, args.raw)
            print("\n" + "-" * 60)
    elif args.query:
        run_query(args.query, args.top_k, args.raw)
    else:
        print("Type a query and press Enter (empty line to quit).")
        while True:
            try:
                text = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            run_query(text, args.top_k, args.raw)


if __name__ == "__main__":
    main()
