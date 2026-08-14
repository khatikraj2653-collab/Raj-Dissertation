"""
cross_validate_methods.py
Checks how much overlap exists between Isolation Forest anomalies and
the small/extreme KMeans clusters — two independent methods agreeing
is strong validation evidence.
"""

import pandas as pd
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent / "results" / "clustering_anomaly_results.csv"

def main():
    df = pd.read_csv(RESULTS_FILE)

    cluster_sizes = df["cluster"].value_counts().sort_index()
    print("Cluster sizes:")
    print(cluster_sizes)

    # Define "small/extreme" clusters as anything under 1% of the dataset
    threshold = len(df) * 0.01
    small_clusters = cluster_sizes[cluster_sizes < threshold].index.tolist()
    print(f"\nSmall/extreme clusters (< {threshold:.0f} rows): {small_clusters}")

    df["in_small_cluster"] = df["cluster"].isin(small_clusters)

    # Cross-tab: anomaly flag vs small-cluster membership
    crosstab = pd.crosstab(df["is_anomaly"], df["in_small_cluster"])
    print("\nCross-tabulation (rows=is_anomaly, cols=in_small_cluster):")
    print(crosstab)

    # Overlap rate
    total_small = df["in_small_cluster"].sum()
    total_anomaly = df["is_anomaly"].sum()
    overlap = ((df["is_anomaly"]) & (df["in_small_cluster"])).sum()

    print(f"\nTotal in small/extreme clusters: {total_small}")
    print(f"Total flagged as anomalies: {total_anomaly}")
    print(f"Overlap (both): {overlap}")
    print(f"% of small-cluster members also flagged as anomalies: {100*overlap/total_small:.1f}%")
    print(f"% of anomalies that are in small clusters: {100*overlap/total_anomaly:.1f}%")

if __name__ == "__main__":
    main()