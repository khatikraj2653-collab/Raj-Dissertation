"""
interpret_clusters.py
Identifies which LCIA indicators most strongly characterize each cluster,
by comparing each cluster's mean indicator values against the dataset-wide mean.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "clean_features.csv"
CLUSTERS_FILE = Path(__file__).parent.parent / "results" / "clustering_anomaly_results.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "cluster_interpretation.txt"

TOP_N_INDICATORS = 5


def main():
    print("Loading data...")
    df_full = pd.read_csv(DATA_FILE)
    df_clusters = pd.read_csv(CLUSTERS_FILE)

    meta_cols = ["activity_name", "geography", "reference_product_name",
                 "reference_product_unit", "reference_product_amount"]
    indicator_cols = [c for c in df_full.columns if c not in meta_cols]

    # Merge cluster assignment back onto full indicator data
    df = df_full.copy()
    df["cluster"] = df_clusters["cluster"]
    df["is_anomaly"] = df_clusters["is_anomaly"]

    X = df[indicator_cols].apply(pd.to_numeric, errors="coerce")

    # Z-score each indicator across the WHOLE dataset (so we can compare
    # cluster means on a common, interpretable scale)
    X_z = (X - X.mean()) / X.std()

    output_lines = []
    print("\n=== CLUSTER PROFILES ===\n")

    for cluster_id in sorted(df["cluster"].unique()):
        mask = df["cluster"] == cluster_id
        size = mask.sum()
        cluster_z_means = X_z[mask].mean().sort_values(ascending=False)

        top_high = cluster_z_means.head(TOP_N_INDICATORS)
        top_low = cluster_z_means.tail(TOP_N_INDICATORS)

        # Example activities in this cluster
        examples = df[mask]["activity_name"].drop_duplicates().head(3).tolist()

        block = [f"--- Cluster {cluster_id} (n={size}) ---"]
        block.append(f"Example activities: {examples}")
        block.append("Elevated indicators (relative to dataset average):")
        for name, z in top_high.items():
            block.append(f"  {name}: z={z:+.2f}")
        block.append("Depressed indicators (relative to dataset average):")
        for name, z in top_low.items():
            block.append(f"  {name}: z={z:+.2f}")
        block.append("")

        text_block = "\n".join(block)
        print(text_block)
        output_lines.append(text_block)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved full cluster interpretation to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()