"""
score_drafts_multi_model.py
Rubric-scans all 6 models' goal-and-scope drafts for ISO 14044 structural
elements and technical-precision markers (keyword-based automated screening,
to be confirmed/refined by manual reading).
"""

import json
import re
from pathlib import Path

DRAFTS_FILE = Path(__file__).parent.parent / "results" / "goal_scope_drafts_multi_model.json"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "task2_rubric_summary.json"

MODEL_NAMES = {
    "claude": "Claude Haiku 4.5",
    "gpt": "GPT-4o-mini",
    "gpt56terra": "GPT-5.6 Terra",
    "gemini": "Gemini 3.5 Flash",
    "llama": "Llama 3.3 70B",
    "gptoss": "GPT-OSS 120B",
}

STRUCTURAL_CHECKS = {
    "goal_purpose": [r"\bgoal\b", r"\bpurpose\b", r"\baim(s)?\b"],
    "intended_application": [r"intended application", r"intended use", r"application"],
    "target_audience": [r"target audience", r"audience", r"stakeholder"],
    "functional_unit": [r"functional unit"],
    "system_boundaries": [r"system boundar", r"cradle-to-", r"gate-to-gate"],
    "assumptions_limitations": [r"assumption", r"limitation"],
}

PRECISION_CHECKS = {
    "comparative_assertions": [r"comparative assertion"],
    "allocation_methodology": [r"allocation"],
    "capital_goods": [r"capital (equipment|goods)", r"infrastructure"],
    "iso_citation": [r"iso 14040", r"iso 14044"],
}


def check_patterns(text, patterns):
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def main():
    with open(DRAFTS_FILE, "r", encoding="utf-8") as f:
        drafts_dict = json.load(f)

    cases = [drafts_dict[k] for k in sorted(drafts_dict.keys(), key=int)]
    n = len(cases)
    print(f"Total cases: {n}\n")

    model_keys = list(MODEL_NAMES.keys())
    structural_counts = {m: {k: 0 for k in STRUCTURAL_CHECKS} for m in model_keys}
    precision_counts = {m: {k: 0 for k in PRECISION_CHECKS} for m in model_keys}
    word_counts = {m: [] for m in model_keys}
    missing_drafts = {m: 0 for m in model_keys}

    for case in cases:
        for m in model_keys:
            draft = case.get(f"{m}_draft")
            if not draft:
                missing_drafts[m] += 1
                continue
            word_counts[m].append(len(draft.split()))
            for check_name, patterns in STRUCTURAL_CHECKS.items():
                if check_patterns(draft, patterns):
                    structural_counts[m][check_name] += 1
            for check_name, patterns in PRECISION_CHECKS.items():
                if check_patterns(draft, patterns):
                    precision_counts[m][check_name] += 1

    print("=== STRUCTURAL COMPLETENESS (out of {} cases) ===".format(n))
    for m in model_keys:
        total_possible = len(STRUCTURAL_CHECKS) * n
        total_present = sum(structural_counts[m].values())
        pct = 100 * total_present / total_possible
        print(f"\n{MODEL_NAMES[m]}: {pct:.1f}% overall structural completeness")
        for check_name in STRUCTURAL_CHECKS:
            print(f"  {check_name}: {structural_counts[m][check_name]}/{n}")

    print("\n\n=== TECHNICAL PRECISION MARKERS (out of {} cases) ===".format(n))
    for m in model_keys:
        print(f"\n{MODEL_NAMES[m]}:")
        for check_name in PRECISION_CHECKS:
            print(f"  {check_name}: {precision_counts[m][check_name]}/{n}")

    print("\n\n=== AVERAGE WORD COUNT ===")
    for m in model_keys:
        avg = sum(word_counts[m]) / len(word_counts[m]) if word_counts[m] else 0
        print(f"{MODEL_NAMES[m]}: {avg:.0f} words (missing drafts: {missing_drafts[m]})")

    summary = {
        "sample_size": n,
        "structural_completeness": {MODEL_NAMES[m]: structural_counts[m] for m in model_keys},
        "precision_markers": {MODEL_NAMES[m]: precision_counts[m] for m in model_keys},
        "avg_word_count": {MODEL_NAMES[m]: round(sum(word_counts[m]) / len(word_counts[m]), 0) if word_counts[m] else 0 for m in model_keys},
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()