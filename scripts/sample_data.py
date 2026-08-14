"""
sample_data.py
Pulls a reproducible random sample from the ecoinvent LCIA dataset
and builds a test set for the LLM extraction task (Objective 3).
"""

import pandas as pd
import json
from pathlib import Path

# ---- Config ----
SOURCE_FILE = Path(__file__).parent.parent / "data" / "RAJ_DISS.xlsx"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "test_set.json"
SAMPLE_SIZE = 100
RANDOM_SEED = 42  # fixed seed = reproducible sample, cite this in your methodology

def load_data():
    # Real column headers live on row 4 (index 3), data starts row 5
    df = pd.read_excel(
        SOURCE_FILE,
        sheet_name="LCIA",
        header=3,
        usecols="A,C,D,E,F,G",  # skip blank col B
    )
    df.columns = [
        "activity_uuid",
        "activity_name",
        "geography",
        "reference_product_name",
        "reference_product_unit",
        "reference_product_amount",
    ]
    df = df.dropna(subset=["activity_name", "geography", "reference_product_name"])
    return df

def build_description(row):
    """Turn clean structured fields into a natural-language description
    for the LLM to parse back apart."""
    return (
        f"This life cycle inventory dataset covers the activity "
        f"'{row['activity_name']}', located in the region '{row['geography']}'. "
        f"It produces {row['reference_product_amount']} "
        f"{row['reference_product_unit']} of '{row['reference_product_name']}' "
        f"as its reference product."
    )

def main():
    df = load_data()
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    test_set = []
    for i, row in sample.iterrows():
        test_set.append({
            "id": i,
            "activity_uuid": row["activity_uuid"],
            "description": build_description(row),
            "ground_truth": {
                "activity_name": row["activity_name"],
                "geography": row["geography"],
                "reference_product_name": row["reference_product_name"],
                "reference_product_unit": row["reference_product_unit"],
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(test_set)} test cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()