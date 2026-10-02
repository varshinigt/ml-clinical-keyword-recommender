# parse_mesh.py
# Turns the MeSH descriptor XML into mesh_terms.json for the keyword extractor.
#
# Usage:
#   1. Download desc2026.xml from
#      https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/
#      and put it in data/
#   2. python3 parse_mesh.py                 (or: python3 parse_mesh.py path/to/desc2026.xml)
#
# Output (in data/):
#   mesh_terms.json  : {"lowercased term": "DescriptorUI", ...}  all synonyms -> same ID
#   mesh_names.json  : {"DescriptorUI": "Main heading name", ...}

import sys, json, glob, os
import xml.etree.ElementTree as ET

MIN_LEN = 3   # drop very short terms like "A" or "of" (too many false matches)

def parse(path):
    term_to_id, names = {}, {}
    ambiguous = set()

    # iterparse keeps memory low (the file is a few hundred MB)
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue
        ui = elem.findtext("DescriptorUI")
        name = elem.findtext("DescriptorName/String")
        if ui and name:
            names[ui] = name
            # every synonym (entry term) of every concept under this descriptor
            for s in elem.findall("ConceptList/Concept/TermList/Term/String"):
                t = (s.text or "").strip().lower()
                if len(t) < MIN_LEN:
                    continue
                if t in term_to_id and term_to_id[t] != ui:
                    ambiguous.add(t)          # same string used by 2 descriptors
                else:
                    term_to_id[t] = ui
        elem.clear()

    # drop strings that point to more than one descriptor (can't tell which is meant)
    for t in ambiguous:
        term_to_id.pop(t, None)
    return term_to_id, names, len(ambiguous)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        found = sorted(glob.glob("data/desc*.xml"))
        if not found:
            sys.exit("No data/desc*.xml found. Download the MeSH descriptor XML first.")
        path = found[-1]

    terms, names, n_amb = parse(path)
    os.makedirs("data", exist_ok=True)
    with open("data/mesh_terms.json", "w") as f:
        json.dump(terms, f)
    with open("data/mesh_names.json", "w") as f:
        json.dump(names, f)
    print(f"descriptors: {len(names)}")
    print(f"terms (synonyms): {len(terms)}")
    print(f"ambiguous terms dropped: {n_amb}")