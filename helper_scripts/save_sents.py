import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Extract a column from a CSV and save one value per line to a text file"
)

parser.add_argument(
    "--csv",
    required=True,
    help="Path to input CSV file"
)

parser.add_argument(
    "--col",
    help="Name of the column to extract"
)

parser.add_argument(
    "--out",
    help="Output text file path"
)

parser.add_argument(
    "--max-len",
    type=int,
    default=None,
    help="Optional: keep only rows whose column has <= max-len whitespace-separated tokens.",
)

parser.add_argument(
    "--using-engall",
    action="store_true",
    help="Extract non-empty ctb_tree_morph parses and matching sent_morphtok sentences.",
)

parser.add_argument(
    "--engall-out-dir",
    default="data/filtered_ctb",
    help="Output directory for engall exports (default: data/filtered_ctb).",
)

args = parser.parse_args()

# Read CSV
try:
    df = pd.read_csv(args.csv, low_memory=False)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

if args.using_engall:
    for col in ["ctb_tree_morph", "sent_morphtok"]:
        if col not in df.columns:
            print(f"Error: column '{col}' not found.")
            print("Available columns:", ", ".join(df.columns))
            sys.exit(1)

    mask = (
        df["ctb_tree_morph"].notna()
        & (df["ctb_tree_morph"].astype(str).str.strip() != "")
        & df["sent_morphtok"].notna()
        & (df["sent_morphtok"].astype(str).str.strip() != "")
    )
    out_dir = args.engall_out_dir
    parse_path = f"{out_dir}/filtered_ctb_parses.txt"
    sent_path = f"{out_dir}/filtered_ctb_sents_morphtok.txt"

    df.loc[mask, "ctb_tree_morph"].to_csv(parse_path, index=False, header=False)
    df.loc[mask, "sent_morphtok"].to_csv(sent_path, index=False, header=False)

    print(f"Saved {mask.sum()} parses to {parse_path}")
    print(f"Saved {mask.sum()} sents to {sent_path}")
else:
    if not args.col or not args.out:
        print("Error: --col and --out are required unless --using-engall is set.")
        sys.exit(1)

    if args.col not in df.columns:
        print(f"Error: column '{args.col}' not found.")
        print("Available columns:", ", ".join(df.columns))
        sys.exit(1)

    series = df[args.col].dropna().astype(str)
    if args.max_len is not None:
        mask = series.str.split().str.len() <= args.max_len
        series = series[mask]

    series.to_csv(
        args.out,
        index=False,
        header=False
    )

    print(f"Saved {len(series)} lines to {args.out}")
