"""
inspect_low_k.py
Inspects what k=2 and k=3 clusters actually contain, to check whether
the high silhouette scores reflect a meaningful split or just isolate
a handful of extreme outliers.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
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

    for k in [2, 3]:
        print(f"\n=== k={k} ===")
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        df[f"cluster_k{k}"] = labels
        print(df[f"cluster_k{k}"].value_counts().sort_index())

        # show examples from the smallest cluster
        smallest = df[f"cluster_k{k}"].value_counts().idxmin()
        print(f"Examples from smallest cluster ({smallest}):")
        print(df[df[f"cluster_k{k}"] == smallest]["activity_name"].head(10).tolist())

if __name__ == "__main__":
    main()