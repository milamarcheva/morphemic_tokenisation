#!/usr/bin/env python3
"""
Attach CHILDES Treebank parses (normal + morphemic) to rows in an aggregate CSV.

Assumes the treebank files in data/childes_treebank/ctb-data_normal and
ctb-data_morph have matching line order. Sentences are matched to the CSV by
normalised surface text (lowercased, punctuation stripped).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from nltk import Tree  # type: ignore

SPECIAL_REPLACEMENTS = {
    "hafta": ["have", "to"],
    "needta": ["need", "to"],
    "lets": ["let", "'s"],
}


def canonicalize(text: str) -> tuple[str, str]:
    """
    Return (canon_with_apostrophes, canon_without_apostrophes).
    Keeps alphanumerics and apostrophes; second variant drops apostrophes entirely.
    """
    # Tokenize with simple regex, but split common contractions n't and 's
    raw_tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    tokens = []
    for tok in raw_tokens:
        if tok.endswith("n't") and len(tok) > 3:
            tokens.append(tok[:-3])
            tokens.append("n't")
        elif tok.endswith("'s") and len(tok) > 2:
            tokens.append(tok[:-2])
            tokens.append("s")
        else:
            tokens.append(tok)
    canon = " ".join(tokens)
    canon_noapos = canon.replace("'", "")
    return canon, canon_noapos


def apply_special_replacements(text: str) -> str:
    tokens = []
    for tok in str(text).split():
        rep = SPECIAL_REPLACEMENTS.get(tok.lower())
        if rep:
            tokens.extend(rep)
        else:
            tokens.append(tok)
    return " ".join(tokens)


def tree_leaves(tree_str: str) -> str:
    """Return a space-joined string of tree leaves."""
    try:
        t = Tree.fromstring(tree_str)
        leaves = t.leaves()
        return " ".join(leaves)
    except Exception:
        return ""


def load_treebank_pairs(normal_paths: List[Path], morph_paths: List[Path]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for n_path, m_path in zip(normal_paths, morph_paths):
        with n_path.open() as fn, m_path.open() as fm:
            for n_line, m_line in zip(fn, fm):
                n_line = n_line.strip()
                m_line = m_line.strip()
                if not n_line:
                    continue
                pairs.append((n_line, m_line))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach CHILDES Treebank parses to an aggregate CSV.")
    ap.add_argument("--csv", required=True, help="Input aggregate CSV (e.g., data/dfs/engna_aggregate_df.csv)")
    ap.add_argument("--output", help="Output CSV path (default: overwrite input when --inplace is set)")
    ap.add_argument("--inplace", action="store_true", help="Write output back to the input CSV")
    ap.add_argument(
        "--treebank-dir",
        default="data/childes_treebank/",
        help="Base directory containing ctb-data_normal and ctb-data_morph",
    )
    ap.add_argument(
        "--text-col",
        default="spacy_normtok",
        help="Column name with utterance text for normal matching (default: spacy_normtok)",
    )
    ap.add_argument(
        "--match-mode",
        choices=["normal", "morph"],
        default="normal",
        help="Match mode: normal (ctb-data_normal) or morph (ctb-data_morph).",
    )
    ap.add_argument(
        "--morph-text-col",
        default="sent_morphtok",
        help="Column name with utterance text for morph matching (default: sent_morphtok).",
    )
    ap.add_argument(
        "--unmatched-output",
        help="Optional path to write sentences whose parses could not be matched to the CSV.",
    )
    args = ap.parse_args()

    base = Path(args.treebank_dir)
    normal_files = sorted((base / "ctb-data_normal").glob("ctb-*_normal.txt"))
    morph_files = sorted((base / "ctb-data_morph").glob("ctb-*.txt"))
    if not normal_files or len(normal_files) != len(morph_files):
        raise SystemExit("Expected matching normal/morph treebank files.")

    df = pd.read_csv(args.csv)
    # if "ctb_tree_normal" not in df.columns:
    df["ctb_tree_normal"] = ""
    # if "ctb_tree_morph" not in df.columns:
    df["ctb_tree_morph"] = ""
    output_path: Path | None = None
    if args.output:
        output_path = Path(args.output)
    elif args.inplace:
        output_path = Path(args.csv)
    else:
        raise SystemExit("Specify --output or use --inplace to overwrite the input CSV.")

    # Helper to build index from canonical utt to row indices with optional path prefixes
    def build_map(prefixes: List[str] | None, text_col: str) -> Dict[str, List[int]]:
        canon_map: Dict[str, List[int]] = defaultdict(list)
        for idx, (utt, path) in enumerate(zip(df[text_col].fillna(""), df.get("path", []))):
            if prefixes:
                p = str(path) if isinstance(path, str) else ""
                if not any(p.startswith(pref) for pref in prefixes):
                    continue
            utt_rewritten = utt #apply_special_replacements
            canon, canon_noapos = canonicalize(utt_rewritten)
            if canon:
                canon_map[canon].append(idx)
            if canon_noapos and canon_noapos != canon:
                canon_map[canon_noapos].append(idx)
        return canon_map

    matched = 0
    unmatched: List[str] = []
    # Define allowed prefixes per split
    test_prefixes = ["CHILDES_Eng-NA/Brown/Adam"]
    train_valid_prefixes = ["CHILDES_Eng-NA/Brown/Eve", "CHILDES_Eng-NA/Brown/Sarah", "CHILDES_Eng-NA/HSLLD", "CHILDES_Eng-NA/Soderstrom", "CHILDES_Eng-NA/Suppes", "CHILDES_Eng-NA/Valian"]

    for n_path, m_path in zip(normal_files, morph_files):
        fname = n_path.name
        if "test" in fname:
            prefixes = test_prefixes
        else:
            prefixes = train_valid_prefixes
        if args.match_mode == "normal":
            text_col = args.text_col
        else:
            text_col = args.morph_text_col
        if text_col not in df.columns:
            raise SystemExit(f"Column not found for {args.match_mode} match: {text_col}")
        canon_map = build_map(prefixes, text_col)
        file_pairs = load_treebank_pairs([n_path], [m_path])
        for n_tree, m_tree in file_pairs:
            sent = tree_leaves(n_tree) if args.match_mode == "normal" else tree_leaves(m_tree)
            if len(sent.split()) <= 1:
                continue  # skip single-token yields
            canon, canon_noapos = canonicalize(sent)
            if not canon and not canon_noapos:
                continue
            idx_list = canon_map.get(canon)
            if idx_list:
                idx = idx_list.pop(0)
                df.at[idx, "ctb_tree_normal"] = n_tree
                df.at[idx, "ctb_tree_morph"] = m_tree
                matched += 1
            elif canon_noapos:
                idx_list_alt = canon_map.get(canon_noapos)
                if idx_list_alt:
                    idx = idx_list_alt.pop(0)
                    df.at[idx, "ctb_tree_normal"] = n_tree
                    df.at[idx, "ctb_tree_morph"] = m_tree
                    matched += 1
                else:
                    unmatched.append(f"{n_path.name}\t{sent}")
            else:
                unmatched.append(f"{n_path.name}\t{sent}")

    assert output_path is not None
    df.to_csv(output_path, index=False)
    print(f"Matched {matched} treebank sentences; wrote {len(df)} rows to {output_path}")
    if unmatched:
        if args.unmatched_output:
            Path(args.unmatched_output).write_text("\n".join(unmatched))
            print(f"Wrote {len(unmatched)} unmatched parses to {args.unmatched_output}")
        else:
            print(f"Unmatched parses ({len(unmatched)}):")
            # for s in unmatched:
            #     print(s)


if __name__ == "__main__":
    main()
