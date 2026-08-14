"""
significance_test.py
McNemar's test to check if GPT vs Claude's accuracy difference
(90% vs 100% on n=50, same test cases) is statistically significant.
"""

import json
from pathlib import Path
from scipy.stats import binomtest

RESULTS_FILE = Path(__file__).parent.parent / "results" / "raw_responses.json"

FIELDS = ["activity_name", "geography", "reference_product_name", "reference_product_unit"]


def normalize(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def is_exact_match(predicted, truth):
    if isinstance(predicted, dict) and predicted.get("parse_error"):
        return False
    for f in FIELDS:
        pred_val = normalize(predicted.get(f) if isinstance(predicted, dict) else None)
        true_val = normalize(truth.get(f))
        if pred_val != true_val:
            return False
    return True


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # b = GPT wrong, Claude right ; c = GPT right, Claude wrong
    b = 0
    c = 0

    for r in results:
        gpt_correct = is_exact_match(r["gpt_parsed"], r["ground_truth"])
        claude_correct = is_exact_match(r["claude_parsed"], r["ground_truth"])

        if not gpt_correct and claude_correct:
            b += 1
        elif gpt_correct and not claude_correct:
            c += 1

    print(f"Discordant pairs: GPT wrong/Claude right = {b}, GPT right/Claude wrong = {c}")

    if b + c == 0:
        print("No discordant pairs — models agreed on every case, no test possible.")
        return

    # McNemar's exact test (binomial test on discordant pairs)
    result = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided")
    p_value = result.pvalue

    print(f"\nMcNemar's exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: statistically significant difference (p < 0.05)")
    else:
        print("Result: NOT statistically significant at the 0.05 level")
        print("→ With n=50, this difference could plausibly be due to chance.")


if __name__ == "__main__":
    main()