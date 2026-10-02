# download_trials.py
# Install first:  pip install requests tqdm pandas pyarrow

import requests, pandas as pd, time
from tqdm import tqdm

URL = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,Condition,EligibilityCriteria"
MAX_DOCS = 50000          # paper used 337k; 50k is plenty for a one-week project

rows, token = [], None
pbar = tqdm(total=MAX_DOCS)

while len(rows) < MAX_DOCS:
    params = {"fields": FIELDS, "pageSize": 1000, "format": "json"}
    if token:
        params["pageToken"] = token

    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    for s in data["studies"]:
        p = s["protocolSection"]
        rows.append({
            "nct_id":         p["identificationModule"]["nctId"],
            "brief_title":    p["identificationModule"].get("briefTitle", ""),
            "official_title": p["identificationModule"].get("officialTitle", ""),
            "brief_summary":  p.get("descriptionModule", {}).get("briefSummary", ""),
            "detailed_desc":  p.get("descriptionModule", {}).get("detailedDescription", ""),
            "conditions":     "; ".join(p.get("conditionsModule", {}).get("conditions", [])),
            "eligibility":    p.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
        })

    pbar.update(len(data["studies"]))
    token = data.get("nextPageToken")
    if not token:
        break
    time.sleep(0.3)       # be polite to the server

pd.DataFrame(rows).to_parquet("trials.parquet")
print(len(rows), "studies saved")