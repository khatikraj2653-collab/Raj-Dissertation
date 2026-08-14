"""
explore_missingness.py
Full-dataset missingness analysis across all ~25,400 rows and 72 LCIA indicator columns.
"""

import pandas as pd
from pathlib import Path

SOURCE_FILE = Path(__file__).parent.parent / "data" / "RAJ_DISS.xlsx"

def main():
    print("Loading full dataset (this may take a minute)...")
    df = pd.read_excel(SOURCE_FILE, sheet_name="LCIA", header=3)

    # First 6 columns are metadata; rest are LCIA indicator scores
    meta_cols = df.columns[:6]
    indicator_cols = df.columns[6:]

    print(f"\nTotal rows: {len(df)}")
    print(f"Metadata columns: {list(meta_cols)}")
    print(f"Total indicator columns: {len(indicator_cols)}")

    # Overall missingness
    total_cells = len(df) * len(indicator_cols)
    missing_cells = df[indicator_cols].isna().sum().sum()
    print(f"\nOverall missingness: {missing_cells} / {total_cells} cells ({100*missing_cells/total_cells:.2f}%)")

    # Per-column missingness - show worst and best 10
    col_missing_pct = (df[indicator_cols].isna().sum() / len(df) * 100).sort_values(ascending=False)
    print("\nTop 10 indicators with MOST missing data:")
    print(col_missing_pct.head(10))
    print("\nTop 10 indicators with LEAST missing data:")
    print(col_missing_pct.tail(10))

    # Save full column missingness report
    output_path = Path(__file__).parent.parent / "results" / "missingness_report.csv"
    col_missing_pct.to_csv(output_path, header=["pct_missing"])
    print(f"\nFull per-column report saved to {output_path}")

if __name__ == "__main__":
    main()