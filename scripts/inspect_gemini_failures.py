"""
inspect_gemini_failures.py
Shows the raw Gemini output on parse-failure cases, to check whether
these are genuine JSON-formatting failures or an extraction quality issue.
"""

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent / "results" / "multi_model_raw_responses.json"

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

count = 0
for case_id, case in sorted(results.items(), key=lambda x: int(x[0])):
    parsed = case.get("gemini_parsed")
    if isinstance(parsed, dict) and parsed.get("parse_error"):
        count += 1
        print(f"\n=== Case {case_id} ===")
        print(f"RAW GEMINI OUTPUT:\n{case.get('gemini_raw')}")
        print("-" * 60)
        if count >= 5:
            break

print(f"\nTotal parse failures shown: {count}")