"""
trie_scanner.py - find MeSH terms in text using a Trie.

Data (not pushed to github, lives in data/):
    data/mesh_terms.json  {"lowercased term": "concept ID"}
    data/mesh_names.json  {"concept ID": "main heading"}
    data/trials.parquet   (loaded through load_data.load_trials)

Usage:
    python trie_scanner.py                 # dummy tests + 100-trial check
    python trie_scanner.py --n 500         # check on more trials
    python trie_scanner.py --dummy_only    # only the built-in dummy tests
"""
import re
import json
import argparse
from collections import Counter

DATA_DIR = "data"
MIN_TERM_CHARS = 3  # ignore very short terms
TEXT_COLS = ["brief_title", "official_title", "brief_summary",
             "detailed_desc", "conditions", "eligibility"]

# Terms (after tokenizing) that are ordinary English words but also MeSH
# synonyms, e.g. "who" -> World Health Organization, "will" -> Volition.
# These are never inserted into the trie.
BLOCK_TERMS = {"who", "will", "back", "name", "fire"}

# Concepts so generic they carry no information (they appear in nearly
# every trial). They are dropped from the scanner output.
# Extend this after reading the top-50 list from the real trials.
BLOCK_CONCEPTS = {
    "D006801",  # Humans
    "D010361",  # Patients
    "D008297",  # Male
    "D005260",  # Female
    # eligibility boilerplate
    "D012108",  # Research Personnel
    "D007258",  # Informed Consent
    "D006664",  # History
    # everyday words
    "D008722",  # Methods
    "D012106",  # Research
    "D019542",  # Program
    "D020478",  # Form
    "D006040",  # Goals
    "D016424",  # Overall
    "D013995",  # Time
    "D012306",  # Risk
    "D000339",  # Affect
    "D001076",  # Aptitude
    "D011153",  # Population
    "D006262",  # Health
    "D001132",  # Arm
    "D014894",  # Weights and Measures
    # too vague to be useful suggestions
    "D013812",  # Therapeutics
    "D004364",  # Pharmaceutical Preparations
    "D004194",  # Disease
    "D003933",  # Diagnosis
    # study-design words
    "D012449",  # Safety
    "D016430",  # Clinical Trial
    "D011897",  # Random Allocation
    "D011795",  # Surveys and Questionnaires
    "D008403",  # Mass Screening
    # round 2: ordinary words that are MeSH synonyms
    "D013502",  # General Surgery   ("surgery")
    "D004532",  # Ego               ("self")
    "D009934",  # Organization and Administration ("administration")
    "D020471",  # Collection
    "D018401",  # Sample Size
    "D004864",  # Equipment and Supplies ("device", "equipment")
    "D000068397",  # Clinical Study
    "D001290",  # Attitude          ("opinion")
    "D001244",  # Association
    "D016431",  # Guideline
    "D007753",  # Laboratories
    "D003187",  # Compliance
    "D032962",  # Consent Forms
    "D004630",  # Emergencies
    "D005190",  # Family            ("relatives")
    # round 3
    "D004467",  # Economics
    "D009272",  # Persons
    "D019359",  # Knowledge
    "D032882",  # Comprehension
    "D012926",  # Social Control, Formal
    "D016449",  # Randomized Controlled Trial (study design)
    # optional (judgment call): outcome / sample-type words that show up
    # in every disease area. Uncomment to block them too.
    # "D010336",  # Pathology
    # "D002477",  # Cells
    # "D013534",  # Survival
    # "D011788",  # Quality of Life
    # "D009026",  # Mortality
    # "D014556",  # Urine
    # "D044967",  # Serum
    # "D001769",  # Blood
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def tokenize(text):
    """Lowercase, replace punctuation with spaces, split into words.
    Used for BOTH mesh terms and trial text so they match consistently."""
    return _NON_ALNUM.sub(" ", text.lower()).split()


class Trie:
    """Trie whose steps are WORDS (not letters), so multi-word terms work."""

    END = "__id__"

    def __init__(self, block_terms=BLOCK_TERMS, block_concepts=BLOCK_CONCEPTS):
        self.root = {}
        self.block_terms = block_terms
        self.block_concepts = block_concepts

    def insert(self, term, concept_id):
        tokens = tokenize(term)
        if not tokens or len("".join(tokens)) < MIN_TERM_CHARS:
            return False
        if " ".join(tokens) in self.block_terms:
            return False
        node = self.root
        for tok in tokens:
            node = node.setdefault(tok, {})
        node[self.END] = concept_id
        return True

    def scan(self, text):
        """Concept IDs found in text: longest match, non-overlapping,
        left to right. Repeats are kept so they can be counted."""
        tokens = tokenize(text)
        found = []
        i, n = 0, len(tokens)
        while i < n:
            node = self.root
            best_id, best_end = None, i
            j = i
            while j < n and tokens[j] in node:
                node = node[tokens[j]]
                j += 1
                if self.END in node:
                    best_id, best_end = node[self.END], j
            if best_id is not None:
                if best_id not in self.block_concepts:
                    found.append(best_id)
                i = best_end  # jump past the matched phrase
            else:
                i += 1
        return found

    def scan_counts(self, text):
        """{concept_id: number of times it appears in text}"""
        return Counter(self.scan(text))


def build_trie(term_to_id, **kwargs):
    trie = Trie(**kwargs)
    used = sum(trie.insert(term, cid) for term, cid in term_to_id.items())
    print(f"Trie built: {used} of {len(term_to_id)} terms inserted")
    return trie


def load_trie(mesh_path=f"{DATA_DIR}/mesh_terms.json"):
    with open(mesh_path, encoding="utf-8") as f:
        return build_trie(json.load(f))


def trial_text(df):
    """One text string per trial (all text columns joined)."""
    return df[TEXT_COLS].fillna("").agg(" ".join, axis=1)


def run_dummy_tests():
    terms = {
        "diabetes": "C1",
        "diabetes mellitus, type 2": "C2",
        "type 2 diabetes": "C2",          # synonym -> same ID
        "heart attack": "C3",
        "myocardial infarction": "C3",    # synonym -> same ID
        "insulin": "C4",
        "a": "C5",                         # too short, ignored
        "who": "C6",                       # blocked term, ignored
        "humans": "C7",                    # blocked concept below
    }
    trie = build_trie(terms, block_terms={"who"}, block_concepts={"C7"})

    assert trie.scan("Patients with Diabetes Mellitus, Type 2.") == ["C2"]
    assert trie.scan("diabetes and insulin") == ["C1", "C4"]
    assert trie.scan("Heart-Attack; MYOCARDIAL INFARCTION!") == ["C3", "C3"]
    assert trie.scan("heart failure") == []
    assert trie.scan("a a a") == []
    assert trie.scan("who is here") == []
    assert trie.scan("humans take insulin") == ["C4"]
    assert trie.scan_counts("insulin, insulin and diabetes")["C4"] == 2
    print("All dummy tests passed.")


def check_on_trials(n):
    """Scan the first n real trials and print the 50 most common concepts."""
    try:
        from load_data import load_trials  # teammate's loader
    except ImportError:  # her file isn't in this folder yet
        import pandas as pd

        def load_trials():
            return pd.read_parquet(f"{DATA_DIR}/trials.parquet")

    with open(f"{DATA_DIR}/mesh_names.json", encoding="utf-8") as f:
        names = json.load(f)
    trie = load_trie()

    df = load_trials().head(n)
    texts = trial_text(df)

    mentions, doc_freq = Counter(), Counter()
    for text in texts:
        counts = trie.scan_counts(text)
        mentions.update(counts)
        doc_freq.update(counts.keys())

    print(f"\nScanned {len(df)} trials, {len(mentions)} distinct concepts found")
    print("\nTop 50 concepts by number of trials containing them:")
    for cid, d in doc_freq.most_common(50):
        print(f"{d:4d} trials | {mentions[cid]:5d} mentions | {cid} {names.get(cid, '?')}")
    print("\nRead this list. Generic or junk concepts -> add to BLOCK_CONCEPTS; "
          "junk words -> BLOCK_TERMS.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dummy_only", action="store_true")
    args = ap.parse_args()

    run_dummy_tests()
    if not args.dummy_only:
        check_on_trials(args.n)
