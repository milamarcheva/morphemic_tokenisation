#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Print rows whose mor column contains a substring.")
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., data/dfs/brown_aggregate_df.csv)")
    ap.add_argument("--pattern", default="xxx", help="Substring to search for in mor (default: xxx)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    hits = df[df["mor"].fillna("").str.contains(args.pattern, regex=False)]
    if hits.empty:
        print("No rows found.")
        return
    # Print all columns for matching rows
    print(hits.to_string(index=False))

if __name__ == "__main__":
    main()
