"""
expand_sample.py
Builds a 200-case test set: the original 100 (already tested on Claude/GPT)
plus 100 new, non-overlapping cases — so existing results are never wasted.
"""

import pandas as pd
import json
from pathlib import Path

SOURCE_FILE = Path(__file__).parent.parent / "data" / "RAJ_DISS.xlsx"
ORIGINAL_TEST_SET = Path(__file__).parent.parent / "data" / "test_set.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "test_set_200.json"
NEW_SAMPLE_SEED = 43  # different from original seed=42, ensures new rows
ADDITIONAL_SIZE = 100


def load_data():
    df = pd.read_excel(
        SOURCE_FILE, sheet_name="LCIA", header=3,
        usecols="A,C,D,E,F,G",
    )
    df.columns = ["activity_uuid", "activity_name", "geography",
                  "reference_product_name", "reference_product_unit", "reference_product_amount"]
    df = df.dropna(subset=["activity_name", "geography", "reference_product_name"])
    return df


def build_description(row):
    return (
        f"This life cycle inventory dataset covers the activity "
        f"'{row['activity_name']}', located in the region '{row['geography']}'. "
        f"It produces {row['reference_product_amount']} "
        f"{row['reference_product_unit']} of '{row['reference_product_name']}' "
        f"as its reference product."
    )


def main():
    with open(ORIGINAL_TEST_SET, "r", encoding="utf-8") as f:
        original_cases = json.load(f)
    original_uuids = set(c["activity_uuid"] for c in original_cases)
    print(f"Original test set: {len(original_cases)} cases")

    df = load_data()
    remaining = df[~df["activity_uuid"].isin(original_uuids)]
    print(f"Remaining eligible rows (excluding originals): {len(remaining)}")

    new_sample = remaining.sample(n=ADDITIONAL_SIZE, random_state=NEW_SAMPLE_SEED).reset_index(drop=True)

    new_cases = []
    next_id = len(original_cases)
    for _, row in new_sample.iterrows():
        new_cases.append({
            "id": next_id,
            "activity_uuid": row["activity_uuid"],
            "description": build_description(row),
            "ground_truth": {
                "activity_name": row["activity_name"],
                "geography": row["geography"],
                "reference_product_name": row["reference_product_name"],
                "reference_product_unit": row["reference_product_unit"],
            },
            "batch": "new"
        })
        next_id += 1

    # tag original cases too, for provenance
    for c in original_cases:
        c["batch"] = "original"

    full_set = original_cases + new_cases
    print(f"Final combined test set: {len(full_set)} cases")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_set, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()