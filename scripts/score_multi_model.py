"""
score_multi_model.py
Scores all 5 models (Claude, GPT, Gemini, Llama, GPT-OSS) on the structured
extraction task against ground truth. Runs Cochran's Q test (overall
difference across all 5 models) plus Bonferroni-corrected pairwise
McNemar's tests, since comparing 5 models requires correcting for
multiple comparisons.
"""

import json
from pathlib import Path
from itertools import combinations
from scipy.stats import binomtest
import numpy as np

RESULTS_FILE = Path(__file__).parent.parent / "results" / "multi_model_raw_responses.json"
SUMMARY_FILE = Path(__file__).parent.parent / "results" / "multi_model_summary.json"

FIELDS = ["activity_name", "geography", "reference_product_name", "reference_product_unit"]
MODEL_NAMES = {
    "claude": "Claude Haiku 4.5",
    "gpt": "GPT-4o-mini",
    "gpt56terra": "GPT-5.6 Terra",
    "gemini": "Gemini 3.5 Flash",
    "llama": "Llama 3.3 70B",
    "gptoss": "GPT-OSS 120B",
}


def normalize(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def is_exact_match(predicted, truth):
    if not isinstance(predicted, dict) or predicted.get("parse_error"):
        return False
    for f in FIELDS:
        if normalize(predicted.get(f)) != normalize(truth.get(f)):
            return False
    return True


def field_correct(predicted, truth, field):
    if not isinstance(predicted, dict) or predicted.get("parse_error"):
        return False
    return normalize(predicted.get(field)) == normalize(truth.get(field))


def cochrans_q(binary_matrix):
    n, k = binary_matrix.shape
    col_sums = binary_matrix.sum(axis=0)
    row_sums = binary_matrix.sum(axis=1)
    grand_total = binary_matrix.sum()

    numerator = k * (k - 1) * np.sum((col_sums - grand_total / k) ** 2)
    denominator = k * grand_total - np.sum(row_sums ** 2)
    if denominator == 0:
        return None, None

    q_stat = numerator / denominator
    df = k - 1
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(q_stat, df)
    return q_stat, p_value


def mcnemar_pairwise(binary_matrix, model_keys, i, j):
    col_i = binary_matrix[:, i]
    col_j = binary_matrix[:, j]
    b = np.sum((col_i == 0) & (col_j == 1))
    c = np.sum((col_i == 1) & (col_j == 0))
    if b + c == 0:
        return b, c, 1.0
    result = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided")
    return b, c, result.pvalue


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results_dict = json.load(f)

    results = [results_dict[k] for k in sorted(results_dict.keys(), key=int)]
    model_keys = list(MODEL_NAMES.keys())
    n = len(results)
    print(f"Total cases: {n}")

    binary_matrix = np.zeros((n, len(model_keys)), dtype=int)
    field_accuracy = {m: {f: 0 for f in FIELDS} for m in model_keys}
    parse_failures = {m: 0 for m in model_keys}

    for row_idx, case in enumerate(results):
        truth = case["ground_truth"]
        for col_idx, m in enumerate(model_keys):
            parsed = case.get(f"{m}_parsed")
            if isinstance(parsed, dict) and parsed.get("parse_error"):
                parse_failures[m] += 1
            if is_exact_match(parsed, truth):
                binary_matrix[row_idx, col_idx] = 1
            for f in FIELDS:
                if field_correct(parsed, truth, f):
                    field_accuracy[m][f] += 1

    print("\n=== OVERALL ACCURACY ===")
    overall_accuracy = {}
    for col_idx, m in enumerate(model_keys):
        acc = binary_matrix[:, col_idx].sum() / n * 100
        overall_accuracy[m] = round(acc, 1)
        print(f"{MODEL_NAMES[m]}: {acc:.1f}%  (parse failures: {parse_failures[m]})")

    print("\n=== PER-FIELD ACCURACY ===")
    field_accuracy_pct = {}
    for m in model_keys:
        field_accuracy_pct[m] = {f: round(100 * field_accuracy[m][f] / n, 1) for f in FIELDS}
        print(f"{MODEL_NAMES[m]}: {field_accuracy_pct[m]}")

    print("\n=== COCHRAN'S Q TEST (overall difference across all 5 models) ===")
    q_stat, q_pvalue = cochrans_q(binary_matrix)
    print(f"Q statistic: {q_stat:.3f}, p-value: {q_pvalue:.6f}")
    if q_pvalue < 0.05:
        print("Result: statistically significant overall difference exists among the 5 models")
    else:
        print("Result: no statistically significant overall difference among the 5 models")

    print("\n=== PAIRWISE MCNEMAR'S TESTS (Bonferroni-corrected) ===")
    pairs = list(combinations(range(len(model_keys)), 2))
    n_comparisons = len(pairs)
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"Number of pairwise comparisons: {n_comparisons}, Bonferroni-corrected alpha: {bonferroni_alpha:.5f}")

    pairwise_results = []
    for i, j in pairs:
        m_i, m_j = model_keys[i], model_keys[j]
        b, c, p = mcnemar_pairwise(binary_matrix, model_keys, i, j)
        significant = p < bonferroni_alpha
        pairwise_results.append({
            "model_a": MODEL_NAMES[m_i],
            "model_b": MODEL_NAMES[m_j],
            "a_wrong_b_right": int(b),
            "a_right_b_wrong": int(c),
            "p_value": round(p, 6),
            "significant_after_bonferroni": bool(significant),
        })
        sig_marker = "***" if significant else ""
        print(f"{MODEL_NAMES[m_i]} vs {MODEL_NAMES[m_j]}: b={b}, c={c}, p={p:.6f} {sig_marker}")

    summary = {
        "sample_size": n,
        "overall_accuracy_pct": {MODEL_NAMES[m]: overall_accuracy[m] for m in model_keys},
        "field_accuracy_pct": {MODEL_NAMES[m]: field_accuracy_pct[m] for m in model_keys},
        "parse_failures": {MODEL_NAMES[m]: parse_failures[m] for m in model_keys},
        "cochrans_q": {"statistic": round(q_stat, 3) if q_stat is not None else None, "p_value": round(q_pvalue, 6) if q_pvalue is not None else None},
        "pairwise_mcnemar_bonferroni": pairwise_results,
        "bonferroni_alpha": round(bonferroni_alpha, 5),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()