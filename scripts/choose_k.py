"""
choose_k.py
Tests multiple values of k for KMeans clustering to justify the final
choice of k, using silhouette score and inertia (elbow method).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "reduced_features.csv"
RANDOM_STATE = 42

def main():
    df = pd.read_csv(DATA_FILE)
    meta_cols = ["activity_name", "geography", "reference_product_name",
                 "reference_product_unit", "reference_product_amount"]
    indicator_cols = [c for c in df.columns if c not in meta_cols]

    X_raw = df[indicator_cols].apply(pd.to_numeric, errors="coerce")
    power = PowerTransformer(method="yeo-johnson")
    X_power = power.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_power)

    print(f"{'k':>3} | {'Inertia':>12} | {'Silhouette':>10}")
    print("-" * 32)
    for k in range(2, 13):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=RANDOM_STATE)
        print(f"{k:>3} | {km.inertia_:>12.1f} | {sil:>10.3f}")

if __name__ == "__main__":
    main()