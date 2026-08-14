"""
multi_anomaly_comparison.py
Compares two anomaly detection methods (Isolation Forest, Local Outlier
Factor) on the reduced 7-indicator feature set, and cross-validates
their agreement.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "reduced_features.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_STATE = 42
CONTAMINATION = 0.02


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


def main():
    print("Loading and preparing data...")
    df, X_scaled = load_and_prepare()
    print(f"Rows: {len(df)}, features: {X_scaled.shape[1]}")

    print("\nRunning Isolation Forest...")
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    iso_flags = iso.fit_predict(X_scaled)
    df["iso_forest_anomaly"] = (iso_flags == -1)
    n_iso = df["iso_forest_anomaly"].sum()
    print(f"Isolation Forest flagged: {n_iso} ({100*n_iso/len(df):.2f}%)")

    print("\nRunning Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
    lof_flags = lof.fit_predict(X_scaled)
    df["lof_anomaly"] = (lof_flags == -1)
    n_lof = df["lof_anomaly"].sum()
    print(f"LOF flagged: {n_lof} ({100*n_lof/len(df):.2f}%)")

    print("\n=== CROSS-VALIDATION BETWEEN METHODS ===")
    both = ((df["iso_forest_anomaly"]) & (df["lof_anomaly"])).sum()
    only_iso = ((df["iso_forest_anomaly"]) & (~df["lof_anomaly"])).sum()
    only_lof = ((~df["iso_forest_anomaly"]) & (df["lof_anomaly"])).sum()
    neither = ((~df["iso_forest_anomaly"]) & (~df["lof_anomaly"])).sum()

    print(f"Flagged by BOTH methods: {both}")
    print(f"Flagged by Isolation Forest ONLY: {only_iso}")
    print(f"Flagged by LOF ONLY: {only_lof}")
    print(f"Flagged by neither: {neither}")

    overlap_pct_of_iso = 100 * both / n_iso if n_iso > 0 else 0
    overlap_pct_of_lof = 100 * both / n_lof if n_lof > 0 else 0
    print(f"\n% of Isolation Forest anomalies also flagged by LOF: {overlap_pct_of_iso:.1f}%")
    print(f"% of LOF anomalies also flagged by Isolation Forest: {overlap_pct_of_lof:.1f}%")

    union = both + only_iso + only_lof
    jaccard = both / union if union > 0 else 0
    print(f"Jaccard similarity (agreement index): {jaccard:.3f}")

    print("\n=== EXAMPLES FLAGGED BY BOTH METHODS (highest-confidence anomalies) ===")
    both_mask = (df["iso_forest_anomaly"]) & (df["lof_anomaly"])
    print(df[both_mask][["activity_name", "geography", "reference_product_name"]].head(10).to_string())

    output_cols = ["activity_name", "geography", "reference_product_name",
                    "iso_forest_anomaly", "lof_anomaly"]
    df[output_cols].to_csv(RESULTS_DIR / "multi_anomaly_results.csv", index=False)

    import json
    summary = {
        "isolation_forest_flagged": int(n_iso),
        "lof_flagged": int(n_lof),
        "flagged_by_both": int(both),
        "flagged_by_iso_only": int(only_iso),
        "flagged_by_lof_only": int(only_lof),
        "pct_iso_confirmed_by_lof": round(overlap_pct_of_iso, 1),
        "pct_lof_confirmed_by_iso": round(overlap_pct_of_lof, 1),
        "jaccard_similarity": round(jaccard, 3),
    }
    with open(RESULTS_DIR / "anomaly_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR / 'multi_anomaly_results.csv'}")
    print(f"Saved summary to {RESULTS_DIR / 'anomaly_comparison_summary.json'}")


if __name__ == "__main__":
    main()