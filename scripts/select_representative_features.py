"""
select_representative_features.py
Selects a small set of conceptually distinct climate-change indicators
(rather than all 68, which are ~90% redundant) for interpretable clustering.
"""

import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "clean_features.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "reduced_features.csv"

# Manually curated: one representative indicator per DISTINCT climate concept,
# using the most standard/common method (IPCC 2021, GWP100 with land-transformation)
# where available, to avoid the near-duplicate "no LT" variants and cross-method duplicates.
REPRESENTATIVE_KEYWORDS = [
    "climate change: total (incl. biogenic CO2): global warming potential (GWP100) [IPCC 2021 (incl. biogenic CO2)]",
    "climate change: fossil: global warming potential (GWP100) [IPCC 2021]",
    "climate change: biogenic (incl. CO2): global warming potential (GWP100) [IPCC 2021 (incl. biogenic CO2)]",
    "climate change: direct land use change: global warming potential (GWP100) [IPCC 2021]",
    "climate change: aircraft emissions: global warming potential (GWP100) [IPCC 2021]",
    "climate change: global warming potential (GWP20) [IPCC 2013]",
    "climate change: global temperature change potential (GTP100) [IPCC 2013]",
]

def main():
    df = pd.read_csv(DATA_FILE)
    meta_cols = ["activity_name", "geography", "reference_product_name",
                 "reference_product_unit", "reference_product_amount"]

    available = list(df.columns)
    selected = []
    for target in REPRESENTATIVE_KEYWORDS:
        if target in available:
            selected.append(target)
        else:
            # fuzzy fallback: find closest match containing key phrase
            matches = [c for c in available if target.split(":")[1].split("[")[0].strip() in c]
            print(f"Exact match not found for: {target}")
            print(f"  Closest candidates: {matches[:3]}")

    print(f"\nSelected {len(selected)} representative indicators:")
    for s in selected:
        print(" -", s)

    out_df = df[meta_cols + selected]
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved reduced feature set to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()