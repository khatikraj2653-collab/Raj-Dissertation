"""
spotcheck_drafts.py
Prints a couple of drafts from each of the 4 new models (not yet manually
read) for a quick human sanity-check against the automated keyword scan.
"""

import json
from pathlib import Path

DRAFTS_FILE = Path(__file__).parent.parent / "results" / "goal_scope_drafts_multi_model.json"

with open(DRAFTS_FILE, "r", encoding="utf-8") as f:
    drafts = json.load(f)

MODELS_TO_CHECK = ["gemini", "llama", "gptoss", "gpt56terra"]
CASES_TO_CHECK = ["0", "5"]  # hard coal, paper wrap

for case_id in CASES_TO_CHECK:
    case = drafts[case_id]
    print(f"\n{'='*80}\nCASE {case_id}: {case['activity_name']}\n{'='*80}")
    for m in MODELS_TO_CHECK:
        print(f"\n--- {m.upper()} ---")
        print(case.get(f"{m}_draft", "MISSING"))