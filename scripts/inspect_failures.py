"""
inspect_failures.py
Pulls out and displays every case where GPT's answer didn't exactly match
ground truth — for qualitative analysis in your discussion section.
"""

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent / "results" / "raw_responses.json"

FIELDS = ["activity_name", "geography", "reference_product_name", "reference_product_unit"]


def normalize(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("=== GPT-4o-mini FAILURE CASES ===\n")
    failure_count = 0

    for r in results:
        predicted = r["gpt_parsed"]
        truth = r["ground_truth"]

        if isinstance(predicted, dict) and predicted.get("parse_error"):
            continue

        mismatches = []
        for f in FIELDS:
            pred_val = predicted.get(f) if isinstance(predicted, dict) else None
            true_val = truth.get(f)
            if normalize(pred_val) != normalize(true_val):
                mismatches.append((f, pred_val, true_val))

        if mismatches:
            failure_count += 1
            print(f"--- Case {r['id']} ---")
            print(f"Description: {r['description']}")
            for field, pred, true in mismatches:
                print(f"  MISMATCH [{field}]:")
                print(f"    GPT said:   {pred}")
                print(f"    Actual:     {true}")
            print()

    print(f"Total failure cases: {failure_count}")


if __name__ == "__main__":
    main()