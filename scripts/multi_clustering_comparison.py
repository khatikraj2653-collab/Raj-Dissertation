"""
multi_clustering_comparison.py
Compares three clustering algorithms (KMeans, Agglomerative/Hierarchical,
DBSCAN) on the reduced 7-indicator feature set, using three independent
validity metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "reduced_features.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_STATE = 42


def load_and_prepare():
    df = pd.read_csv(DATA_FILE)
    meta_cols = ["activity_name", "geography", "reference_product_name",
                 "reference_product_unit", "reference_product_amount"]
    indicator_cols = [c for c in df.columns if c not in meta_cols]

    X_raw = df[indicator_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = X_raw.notna().all(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    X_raw = X_raw[valid_mask].reset_index(drop=True)

    power = PowerTransformer(method="yeo-johnson")
    X_power = power.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_power)

    return df, X_scaled


def evaluate_clustering(X, labels, method_name):
    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    valid_mask = labels != -1
    if valid_mask.sum() < len(labels):
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
    else:
        X_valid = X
        labels_valid = labels

    if n_clusters < 2 or len(set(labels_valid)) < 2:
        return {
            "method": method_name, "n_clusters": n_clusters,
            "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
            "noise_points": int((labels == -1).sum()),
        }

    sil = silhouette_score(X_valid, labels_valid, sample_size=5000, random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)

    return {
        "method": method_name, "n_clusters": n_clusters,
        "silhouette": round(sil, 3), "davies_bouldin": round(db, 3),
        "calinski_harabasz": round(ch, 1),
        "noise_points": int((labels == -1).sum()),
    }


def main():
    print("Loading and preparing data...")
    df, X_scaled = load_and_prepare()
    print(f"Rows: {len(df)}, features: {X_scaled.shape[1]}")

    all_results = []

    print("\nRunning KMeans (k=8)...")
    kmeans = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    df["kmeans_cluster"] = km_labels
    all_results.append(evaluate_clustering(X_scaled, km_labels, "KMeans (k=8)"))

    print("Running Agglomerative Clustering (k=8)...")
    agg = AgglomerativeClustering(n_clusters=8, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    df["agglomerative_cluster"] = agg_labels
    all_results.append(evaluate_clustering(X_scaled, agg_labels, "Agglomerative (k=8, ward)"))

    print("Running DBSCAN (tuning eps)...")
    best_eps, best_labels, best_noise_pct = None, None, 1.0
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        db_model = DBSCAN(eps=eps, min_samples=10)
        labels = db_model.fit_predict(X_scaled)
        n_noise = (labels == -1).sum()
        noise_pct = n_noise / len(labels)
        n_found_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  eps={eps}: {n_found_clusters} clusters, {noise_pct:.1%} noise")
        if n_found_clusters >= 2 and noise_pct < best_noise_pct:
            best_eps, best_labels, best_noise_pct = eps, labels, noise_pct

    if best_labels is not None:
        df["dbscan_cluster"] = best_labels
        all_results.append(evaluate_clustering(X_scaled, best_labels, f"DBSCAN (eps={best_eps})"))
        print(f"Selected eps={best_eps} ({best_noise_pct:.1%} noise)")
    else:
        print("DBSCAN: no eps value produced a valid multi-cluster result")

    print("\n=== CLUSTERING METHOD COMPARISON ===")
    print(f"{'Method':<28} {'k':>4} {'Silhouette':>11} {'Davies-Bouldin':>15} {'Calinski-Harabasz':>18} {'Noise':>7}")
    for r in all_results:
        print(f"{r['method']:<28} {r['n_clusters']:>4} {str(r['silhouette']):>11} "
              f"{str(r['davies_bouldin']):>15} {str(r['calinski_harabasz']):>18} {r['noise_points']:>7}")

    output_cols = ["activity_name", "geography", "reference_product_name",
                    "kmeans_cluster", "agglomerative_cluster"]
    if "dbscan_cluster" in df.columns:
        output_cols.append("dbscan_cluster")
    df[output_cols].to_csv(RESULTS_DIR / "multi_clustering_results.csv", index=False)

    import json
    with open(RESULTS_DIR / "clustering_comparison_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR / 'multi_clustering_results.csv'}")
    print(f"Saved summary to {RESULTS_DIR / 'clustering_comparison_summary.json'}")


if __name__ == "__main__":
    main()