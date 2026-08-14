"""
cluster_and_anomaly.py
Clusters ecoinvent activities by their environmental impact profile
(7 representative climate-change indicators) and flags statistical anomalies.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "reduced_features.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "clustering_anomaly_results.csv"

N_CLUSTERS = 8
RANDOM_STATE = 42


def main():
    print("Loading clean feature data...")
    df = pd.read_csv(DATA_FILE)

    meta_cols = ["activity_name", "geography", "reference_product_name",
                 "reference_product_unit", "reference_product_amount"]
    indicator_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Rows: {len(df)}, indicator features: {len(indicator_cols)}")

    X_raw = df[indicator_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = X_raw.notna().all(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    X_raw = X_raw[valid_mask].reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")

    power = PowerTransformer(method="yeo-johnson")
    X_power = power.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_power)

    print(f"\nRunning KMeans (k={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    sil_score = silhouette_score(X_scaled, clusters, sample_size=5000, random_state=RANDOM_STATE)
    print(f"Silhouette score: {sil_score:.3f}")

    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())

    print("\nRunning Isolation Forest anomaly detection...")
    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
    anomaly_flags = iso.fit_predict(X_scaled)
    df["is_anomaly"] = (anomaly_flags == -1)

    n_anomalies = df["is_anomaly"].sum()
    print(f"Flagged {n_anomalies} anomalies out of {len(df)} ({100*n_anomalies/len(df):.2f}%)")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    print(f"\nPCA explained variance (2 components): {pca.explained_variance_ratio_.sum():.1%}")

    output_cols = meta_cols + ["cluster", "is_anomaly", "pca_x", "pca_y"]
    df[output_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")

    print(f"File exists: {OUTPUT_FILE.exists()}")
    print(f"File size: {OUTPUT_FILE.stat().st_size} bytes")
    check_df = pd.read_csv(OUTPUT_FILE)
    print(f"Read-back shape: {check_df.shape}")
    print(f"Read-back cluster sizes:\n{check_df['cluster'].value_counts().sort_index()}")

    print("\n=== Sample flagged anomalies ===")
    print(df[df["is_anomaly"]][["activity_name", "geography", "reference_product_name", "cluster"]].head(10).to_string())


if __name__ == "__main__":
    main()