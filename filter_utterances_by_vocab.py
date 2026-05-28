#!/usr/bin/env python3
"""
Filter a CSV to keep only rows whose utterance tokens are all in a vocab.

Usage:
  python filter_utterances_by_vocab.py \
    --input data/dfs/child_utterances.csv \
    --output data/dfs/child_utterances_vocab_filtered.csv \
    --vocab data/vocabs/filtered_ctb_sents_morphtok.vocab \
    --column utt_cleaned

  python filter_utterances_by_vocab.py \
    --input data/dfs/child_utterances.csv \
    --output data/dfs/child_utterances_vocab_filtered.csv \
    --lexicon data/lexicons/bg_lexicon_filtered.txt \
    --column utt_cleaned
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

WORD_RE = re.compile(r"\b\w+\b")
LEXICON_RE = re.compile(r"^\s*(?:\S+\s+){2}\S+\s+-->\s+(.+?)\s*$")


def load_vocab(vocab_path: str | Path) -> set[str]:
    vocab_path = Path(vocab_path)
    with vocab_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_lexicon(lexicon_path: str | Path) -> set[str]:
    lexicon_path = Path(lexicon_path)
    vocab: set[str] = set()
    with lexicon_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            match = LEXICON_RE.match(line)
            if not match:
                raise SystemExit(
                    f"Could not parse lexicon line {lineno} in {lexicon_path}: {line}"
                )
            vocab.add(match.group(1))
    return vocab


def utterance_in_vocab(utt: str, vocab: set[str]) -> bool:
    tokens = WORD_RE.findall(utt or "")
    if not tokens:
        return False
    return all(tok in vocab for tok in tokens)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--vocab", help="Vocab file path (one token per line).")
    source.add_argument(
        "--lexicon",
        help="Lexicon file path with grammar-style lexical rules, e.g. '1.0 0.1 X --> token'.",
    )
    p.add_argument(
        "--column",
        default="utt_cleaned",
        help="CSV column to check against vocab (default: utt_cleaned).",
    )
    args = p.parse_args()

    vocab = load_vocab(args.vocab) if args.vocab else load_lexicon(args.lexicon)
    df = pd.read_csv(args.input)

    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found in {args.input}.")

    keep_mask = df[args.column].astype(str).apply(lambda s: utterance_in_vocab(s, vocab))
    filtered = df[keep_mask].copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Loaded {len(vocab):,} types")
    print(f"Kept {len(filtered):,} / {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
