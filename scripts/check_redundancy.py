"""
check_redundancy.py
Checks correlation between the 68 climate-change indicators to identify
redundancy, confirms PCA variance explained on the full 68-indicator set,
and quantifies exact-duplicate feature vectors on the reduced 7-indicator
set (the input to clustering/anomaly detection). Saves a summary JSON so
these figures are reproducible from a single results file rather than
console output.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FULL_FEATURES_FILE = DATA_DIR / "clean_features.csv"
REDUCED_FEATURES_FILE = DATA_DIR / "reduced_features.csv"
META_COLS = ["activity_name", "geography", "reference_product_name",
             "reference_product_unit", "reference_product_amount"]


def pca_two_component_variance(df, indicator_cols):
    X = df[indicator_cols].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    X = X[mask]
    X_power = PowerTransformer(method="yeo-johnson").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_power)
    pca = PCA(n_components=2, random_state=42)
    pca.fit_transform(X_scaled)
    return float(pca.explained_variance_ratio_.sum())


def main():
    df = pd.read_csv(FULL_FEATURES_FILE)
    indicator_cols = [c for c in df.columns if c not in META_COLS]

    X = df[indicator_cols].apply(pd.to_numeric, errors="coerce")
    corr = X.corr()

    # Find highly correlated pairs (>0.98, near-duplicates)
    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.98:
                high_corr_pairs.append((cols[i], cols[j], r))

    total_pairs = len(cols) * (len(cols) - 1) // 2
    pct_redundant = 100 * len(high_corr_pairs) / total_pairs
    avg_corr = float(corr.abs().values[np.triu_indices(len(cols), k=1)].mean())

    print(f"Total indicator pairs: {total_pairs}")
    print(f"Pairs with |correlation| > 0.98: {len(high_corr_pairs)}")
    print(f"Percentage highly redundant: {pct_redundant:.1f}%")
    print(f"\nAverage absolute pairwise correlation: {avg_corr:.3f}")

    print("\nSample of highly correlated (near-duplicate) pairs:")
    for a, b, r in high_corr_pairs[:10]:
        print(f"  r={r:.4f}  |  {a[:60]}  <->  {b[:60]}")

    corr.to_csv(RESULTS_DIR / "indicator_correlation_matrix.csv")
    print("\nFull correlation matrix saved.")

    # PCA on the full 68-indicator set, same Yeo-Johnson + StandardScaler
    # pipeline used for clustering (cluster_and_anomaly.py runs PCA on the
    # 7-indicator reduced set only, which is a different, smaller number).
    pca_variance_full68 = pca_two_component_variance(df, indicator_cols)
    print(f"\nPCA (full 68 indicators, Yeo-Johnson + standardised), "
          f"2-component variance explained: {pca_variance_full68:.1%}")

    # Exact-duplicate feature vectors on the reduced 7-indicator set (the
    # actual input to clustering/anomaly detection). Counted as "excess"
    # duplicates: rows beyond the first occurrence of each unique feature
    # vector, i.e. pd.DataFrame.duplicated(keep="first").
    df7 = pd.read_csv(REDUCED_FEATURES_FILE)
    indicator_cols_7 = [c for c in df7.columns if c not in META_COLS]
    dup_excess_mask = df7.duplicated(subset=indicator_cols_7, keep="first")
    n_dup_excess = int(dup_excess_mask.sum())
    pct_dup_excess = 100 * n_dup_excess / len(df7)
    print(f"\nExact-duplicate feature vectors (reduced 7-indicator set, "
          f"excess beyond first occurrence): {n_dup_excess} / {len(df7)} "
          f"({pct_dup_excess:.1f}%)")

    summary = {
        "correlation_redundancy": {
            "total_indicator_pairs": total_pairs,
            "pairs_abs_corr_gt_0.98": len(high_corr_pairs),
            "pct_pairs_abs_corr_gt_0.98": round(pct_redundant, 1),
            "mean_abs_pairwise_correlation": round(avg_corr, 3),
        },
        "pca_full_68_indicators": {
            "n_components": 2,
            "preprocessing": "Yeo-Johnson PowerTransformer + StandardScaler",
            "explained_variance_ratio_sum": round(pca_variance_full68, 4),
        },
        "duplicate_feature_vectors_reduced_7": {
            "definition": "rows beyond first occurrence of each unique "
                           "feature vector (duplicated(keep='first'))",
            "n_rows": len(df7),
            "n_excess_duplicates": n_dup_excess,
            "pct_excess_duplicates": round(pct_dup_excess, 1),
        },
    }
    summary_file = RESULTS_DIR / "redundancy_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")


if __name__ == "__main__":
    main()