"""
score_results.py
Compares GPT and Claude's extracted fields against ground truth,
calculates accuracy, and saves a summary audit log for the dissertation.
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_FILE = Path(__file__).parent.parent / "results" / "raw_responses.json"
SUMMARY_FILE = Path(__file__).parent.parent / "results" / "summary.json"

FIELDS = ["activity_name", "geography", "reference_product_name", "reference_product_unit"]


def normalize(value):
    """Lowercase + strip so minor formatting differences don't count as wrong."""
    if value is None:
        return ""
    return str(value).strip().lower()


def score_model(results, model_key):
    """model_key is 'gpt_parsed' or 'claude_parsed'."""
    field_correct = {f: 0 for f in FIELDS}
    exact_match_count = 0
    parse_failures = 0
    total = len(results)

    for r in results:
        predicted = r[model_key]
        truth = r["ground_truth"]

        if isinstance(predicted, dict) and predicted.get("parse_error"):
            parse_failures += 1
            continue

        all_fields_correct = True
        for f in FIELDS:
            pred_val = normalize(predicted.get(f) if isinstance(predicted, dict) else None)
            true_val = normalize(truth.get(f))
            if pred_val == true_val and pred_val != "":
                field_correct[f] += 1
            else:
                all_fields_correct = False

        if all_fields_correct:
            exact_match_count += 1

    field_accuracy = {f: round(100 * field_correct[f] / total, 1) for f in FIELDS}
    overall_accuracy = round(100 * exact_match_count / total, 1)

    return {
        "total_cases": total,
        "parse_failures": parse_failures,
        "field_accuracy_pct": field_accuracy,
        "overall_exact_match_pct": overall_accuracy,
        "exact_match_count": exact_match_count,
    }


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    gpt_scores = score_model(results, "gpt_parsed")
    claude_scores = score_model(results, "claude_parsed")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "models_compared": {
            "gpt": "gpt-4o-mini",
            "claude": "claude-haiku-4-5-20251001",
        },
        "sample_size": len(results),
        "task": "structured field extraction from LCA activity descriptions",
        "gpt_results": gpt_scores,
        "claude_results": claude_scores,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTS SUMMARY ===")
    print(f"Sample size: {summary['sample_size']}")
    print(f"\nGPT-4o-mini:")
    print(f"  Overall exact-match accuracy: {gpt_scores['overall_exact_match_pct']}%")
    print(f"  Per-field accuracy: {gpt_scores['field_accuracy_pct']}")
    print(f"  Parse failures: {gpt_scores['parse_failures']}")
    print(f"\nClaude Haiku:")
    print(f"  Overall exact-match accuracy: {claude_scores['overall_exact_match_pct']}%")
    print(f"  Per-field accuracy: {claude_scores['field_accuracy_pct']}")
    print(f"  Parse failures: {claude_scores['parse_failures']}")
    print(f"\nSaved full summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()