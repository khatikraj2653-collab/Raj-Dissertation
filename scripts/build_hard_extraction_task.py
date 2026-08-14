"""
build_hard_extraction_task.py
Builds a HARDER version of the extraction task: descriptions that do NOT
directly quote the ground-truth field values, forcing genuine extraction
from context/paraphrase rather than copy-matching or potential memorization.
"""

import pandas as pd
import json
from pathlib import Path

SOURCE_FILE = Path(__file__).parent.parent / "data" / "RAJ_DISS.xlsx"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "test_set_hard.json"
SAMPLE_SIZE = 50
RANDOM_SEED = 99  # new, independent sample


def load_data():
    df = pd.read_excel(
        SOURCE_FILE, sheet_name="LCIA", header=3,
        usecols="A,C,D,E,F,G",
    )
    df.columns = ["activity_uuid", "activity_name", "geography",
                  "reference_product_name", "reference_product_unit", "reference_product_amount"]
    df = df.dropna(subset=["activity_name", "geography", "reference_product_name"])
    return df


GEO_NAMES = {
    "GLO": "worldwide", "RoW": "the rest of the world", "RER": "the European region",
    "CH": "Switzerland", "US": "the United States", "JP": "Japan", "MA": "Morocco",
    "CN": "China", "DE": "Germany", "FR": "France", "GB": "the United Kingdom",
    "CA": "Canada", "IN": "India", "BR": "Brazil", "AU": "Australia",
}


def build_hard_description(row):
    """Paraphrased description WITHOUT directly quoting the exact field
    strings, so a correct answer requires genuine extraction/inference
    rather than copying quoted text or recalling memorized training data."""
    geo_readable = GEO_NAMES.get(row["geography"], f"the region coded {row['geography']}")
    amount = row["reference_product_amount"]
    unit = row["reference_product_unit"]

    return (
        f"An industrial process record describes operations related to "
        f"{row['activity_name'].lower()}. This process is documented as taking place "
        f"in {geo_readable}. The output of this process, when measured, comes to "
        f"{amount} units (specifically {unit}) of a product known as "
        f"{row['reference_product_name'].lower()}. Based on this record, identify the "
        f"underlying process/activity name (matching ecoinvent's standard naming), the "
        f"applicable ecoinvent geography code (not the plain-language region name given "
        f"above), the resulting product name, and its unit of measure."
    )


def main():
    df = load_data()
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    test_set = []
    for i, row in sample.iterrows():
        test_set.append({
            "id": i,
            "activity_uuid": row["activity_uuid"],
            "description": build_hard_description(row),
            "ground_truth": {
                "activity_name": row["activity_name"],
                "geography": row["geography"],
                "reference_product_name": row["reference_product_name"],
                "reference_product_unit": row["reference_product_unit"],
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(test_set)} HARD test cases to {OUTPUT_FILE}")
    print("\nExample description:")
    print(test_set[0]["description"])
    print("\nGround truth:", test_set[0]["ground_truth"])


if __name__ == "__main__":
    main()