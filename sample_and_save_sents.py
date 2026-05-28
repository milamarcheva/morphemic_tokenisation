#!/usr/bin/env python3
"""
Randomly sample rows from a CSV and save sentences built from a token-list column.

Usage:
  python sample_and_save_sents.py \
    --input data/dfs/child_utterances_vocab_morphtok_filtered.csv \
    --column sent_morphtok \
    --n 1000 \
    --output-csv data/dfs/child_utterances_vocab_morphtok_filtered_sampled_1000.csv \
    --output-sents data/dfs/child_utterances_vocab_morphtok_filtered_sampled_1000.raw
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import Counter
from pathlib import Path
import re
import statistics
import pandas as pd


def save_sents(df: pd.DataFrame, column_name: str, filepath: str | Path) -> None:
    sents = []
    for u in df[column_name]:
        if pd.isna(u):
            sents.append("\n")
            continue
        if isinstance(u, (list, tuple)):
            tokens = list(u)
            sents.append(" ".join(tokens) + "\n")
            continue
        if isinstance(u, str):
            text = u.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    tokens = literal_eval(text)
                except Exception as exc:
                    raise SystemExit(
                        f"Failed to parse tokens from column '{column_name}': {u!r}"
                    ) from exc
                sents.append(" ".join(tokens) + "\n")
            else:
                sents.append(text + "\n")
            continue
        sents.append(str(u).strip() + "\n")

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(sents)


def parse_age_to_months(age: str) -> float | None:
    if age is None:
        return None
    text = str(age).strip()
    if not text:
        return None
    if text.endswith("."):
        text = text[:-1]
    m = re.match(r"^(?P<years>\d+);(?P<months>\d+)(?:\.(?P<days>\d+))?$", text)
    if not m:
        return None
    years = int(m.group("years"))
    months = int(m.group("months"))
    days = int(m.group("days")) if m.group("days") else 0
    return years * 12 + months + (days / 30.0)


def print_age_summary(series: pd.Series) -> None:
    ages = [a for a in series if not pd.isna(a)]
    counts = Counter(str(a).strip() for a in ages if str(a).strip())
    parsed = [parse_age_to_months(a) for a in ages]
    parsed = [p for p in parsed if p is not None]

    print("Age summary (sampled rows)")
    print(f"  total_rows: {len(series):,}")
    print(f"  missing_age: {len(series) - len(ages):,}")
    print(f"  unique_age_strings: {len(counts):,}")
    if parsed:
        print(f"  mean_age_months: {statistics.mean(parsed):.2f}")
        print(f"  median_age_months: {statistics.median(parsed):.2f}")
        print(f"  min_age_months: {min(parsed):.2f}")
        print(f"  max_age_months: {max(parsed):.2f}")
    if counts:
        top = counts.most_common(10)
        formatted = ", ".join([f"{age}:{cnt}" for age, cnt in top])
        print(f"  top_age_strings: {formatted}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--column", default="sent_morphtok", help="Column with token lists.")
    p.add_argument("--age-column", default="age", help="Column with child age values.")
    p.add_argument("--n", type=int, default=1000, help="Number of rows to sample.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--output-csv", default=None, help="Output CSV path for sampled rows.")
    p.add_argument("--output-sents", default=None, help="Output path for sentence-per-line file.")
    args = p.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {input_path}.")

    n = min(args.n, len(df))
    if n == 0:
        raise SystemExit("Input CSV has no rows to sample.")

    if args.seed is None:
        sampled = df.sample(n=n)
    else:
        sampled = df.sample(n=n, random_state=args.seed)

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_sampled_{n}{input_path.suffix}")
    )
    output_sents = (
        Path(args.output_sents)
        if args.output_sents
        else input_path.with_name(f"{input_path.stem}_sampled_{n}.raw")
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False, encoding="utf-8")
    save_sents(sampled, args.column, output_sents)

    print(f"Wrote {len(sampled):,} rows to {output_csv}")
    print(f"Wrote sentences to {output_sents}")
    if args.age_column in sampled.columns:
        print_age_summary(sampled[args.age_column])
    else:
        print(f"Age summary skipped: column '{args.age_column}' not found.")


if __name__ == "__main__":
    main()
