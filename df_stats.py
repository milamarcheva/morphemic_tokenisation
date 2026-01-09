#!/usr/bin/env python3
import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_age_to_months(age_str: str) -> Optional[float]:
    # Handles formats like "4;10.06" or "4;10"
    if not isinstance(age_str, str) or not age_str.strip():
        return None
    m = re.match(r"(?P<y>\d+);(?P<m>\d+)(?:\.(?P<d>\d+))?", age_str.strip())
    if not m:
        return None
    years = int(m.group("y"))
    months = int(m.group("m"))
    days = int(m.group("d")) if m.group("d") else 0
    return years * 12 + months + days / 30.0


def tokenize_len(text: str) -> int:
    if not isinstance(text, str):
        return 0
    toks = text.split()
    return sum(1 for t in toks if not re.fullmatch(r"[^\w]+", t))


def extract_tags(mor_entry: str) -> list[str]:
    if not isinstance(mor_entry, str):
        return []
    tags = []
    for tok in mor_entry.split():
        head = tok.split("|", 1)[0]
        if head and not re.fullmatch(r"[^\w]+", head):
            tags.append(head)
    return tags


def extract_nouns(mor_entry: str) -> list[str]:
    """
    Return lemma/surface for tokens tagged as noun|... in %mor.
    """
    if not isinstance(mor_entry, str):
        return []
    nouns = []
    for tok in mor_entry.split():
        if "|" not in tok:
            continue
        head, rest = tok.split("|", 1)
        if head != "noun":
            continue
        lemma = rest.split("-", 1)[0]
        lemma = re.sub(r"[^\w]+", "", lemma.lower())
        if lemma:
            nouns.append(lemma)
    return nouns


def extract_verbs(mor_entry: str) -> list[str]:
    """
    Return lemma/surface for tokens tagged as verb|... in %mor.
    """
    if not isinstance(mor_entry, str):
        return []
    verbs = []
    for tok in mor_entry.split():
        if "|" not in tok:
            continue
        head, rest = tok.split("|", 1)
        if head != "verb":
            continue
        lemma = rest.split("-", 1)[0]
        lemma = re.sub(r"[^\w]+", "", lemma.lower())
        if lemma:
            verbs.append(lemma)
    return verbs


def summarize_lengths(lengths: list[int]) -> dict:
    s = pd.Series(lengths)
    return {
        "count": int(s.size),
        "gt1": int((s > 1).sum()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "min": int(s.min()) if len(s) else 0,
        "max": int(s.max()) if len(s) else 0,
        "p10": float(s.quantile(0.1)),
        "p90": float(s.quantile(0.9)),
    }


def main():
    ap = argparse.ArgumentParser(description="Compute stats for a CHAT aggregate CSV.")
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., data/dfs/brown_aggregate_df.csv)")
    ap.add_argument("--text-col", default="utt", help="Column with utterance text (default: utt)")
    ap.add_argument("--mor-col", default="mor", help="Column with %mor annotations (default: mor)")
    ap.add_argument(
        "--pos-col",
        default=None,
        help="Optional column with POS tuples (e.g., postags_morphtok/postags_normtok). When set, use POS for counts.",
    )
    ap.add_argument(
        "--tok-col",
        default=None,
        help="Optional tokens column to align with --pos-col (defaults: postags_morphtok->sent_morphtok, postags_normtok->spacy_normtok).",
    )
    ap.add_argument("--speaker-col", default="speaker_role", help="Column with speaker role (default: speaker_role)")
    ap.add_argument("--age-col", default="child_age", help="Column with child age string (default: child_age)")
    ap.add_argument("--plots-outdir", default=None, help="Directory to save plots (optional)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    dataset_label = Path(args.csv).stem

    lengths = [tokenize_len(x) for x in df[args.text_col]]
    total_tokens = sum(lengths)
    length_stats = summarize_lengths(lengths)

    # Speaker role distribution
    speaker_counts = Counter(df[args.speaker_col].fillna(""))

    # Tag stats
    tag_counter = Counter()
    tag_len = []
    noun_counter = Counter()
    verb_counter = Counter()
    using_pos = False

    if args.pos_col:
        if args.pos_col not in df.columns:
            raise SystemExit(f"--pos-col {args.pos_col} not found in CSV columns")

        import ast
        using_pos = True

        # Choose token column
        if args.tok_col:
            tok_col = args.tok_col
        elif args.pos_col == "postags_morphtok":
            tok_col = "sent_morphtok"
        elif args.pos_col == "postags_normtok":
            tok_col = "spacy_normtok"
        else:
            tok_col = None

        if tok_col not in df.columns:
            raise SystemExit(f"Tokens column '{tok_col}' (for --pos-col {args.pos_col}) not found. Use --tok-col to specify.")

        noun_upos = {"NOUN", "PROPN"}
        noun_detailed = {"NN", "NNS", "NNP", "NNPS"}
        verb_upos = {"VERB"}

        def parse_pos(val):
            try:
                parsed = ast.literal_eval(str(val))
            except Exception:
                return []
            if not isinstance(parsed, list):
                return []
            out = []
            for p in parsed:
                if isinstance(p, (list, tuple)) and len(p) >= 1:
                    upos = str(p[0])
                    detailed = str(p[1]) if len(p) > 1 and p[1] is not None else ""
                    out.append((upos, detailed))
            return out

        for _, row in df.iterrows():
            parsed = parse_pos(row[args.pos_col])
            tags = [p[0] for p in parsed]
            tag_counter.update(tags)
            tag_len.append(len(tags))

            if tok_col and tok_col in df.columns:
                tokens = str(row[tok_col]).split()
                limit = min(len(parsed), len(tokens))
                for i in range(limit):
                    pos = parsed[i]
                    tok = tokens[i].lower()
                    upos = pos[0].upper()
                    detailed = pos[1].upper() if pos[1] else ""
                    if upos in noun_upos or detailed in noun_detailed:
                        noun_counter.update([tok])
                    if upos in verb_upos or detailed.startswith("V"):
                        verb_counter.update([tok])
            else:
                noun_counter.update(
                    [
                        p[0].lower()
                        for p in parsed
                        if (p[0].upper() in noun_upos or (p[1].upper() in noun_detailed if p[1] else False))
                    ]
                )
                verb_counter.update(
                    [
                        p[0].lower()
                        for p in parsed
                        if (p[0].upper() in verb_upos or ((p[1].upper().startswith("V") if p[1] else False)))
                    ]
                )
    else:
        for mor in df[args.mor_col]:
            tags = extract_tags(mor)
            tag_counter.update(tags)
            tag_len.append(len(tags))
            noun_counter.update(extract_nouns(mor))
            verb_counter.update(extract_verbs(mor))
    tag_len_stats = summarize_lengths(tag_len)


    print("=== Sentence length stats (tokens, punctuation excluded) ===")
    for k, v in length_stats.items():
        print(f"{k}: {v}")
    print(f"total_tokens: {total_tokens}")
    if "spacy_normtok" in df.columns:
        spacy_len = df["spacy_normtok"].fillna("").str.split().str.len()
        short_mask = spacy_len < 2
        print("\nRows with spacy_normtok length < 2 (head 20):")
        print(df.loc[short_mask, ["utt", "spacy_normtok", "sent_morphtok"]].head(20))
    print("\n=== Tag count per sentence (from %mor) ===")
    for k, v in tag_len_stats.items():
        print(f"{k}: {v}")
    print("\n=== Speaker role distribution ===")
    for role, cnt in speaker_counts.most_common():
        print(f"{role or '<blank>'}: {cnt}")
    tag_total = sum(tag_counter.values())
    noun_total = sum(noun_counter.values())
    verb_total = sum(verb_counter.values())

    print("\n=== Top 20 tags ({}) ===".format("POS" if using_pos else "%mor"))
    for tag, cnt in tag_counter.most_common(20):
        pct = (cnt / tag_total * 100) if tag_total else 0.0
        print(f"{tag}: {cnt} ({pct:.2f}%)")
    print("\n=== Top 50 nouns (from %mor noun|...) ===")
    for noun, cnt in noun_counter.most_common(50):
        pct = (cnt / noun_total * 100) if noun_total else 0.0
        print(f"{noun}: {cnt} ({pct:.2f}%)")
    print("\n=== Top 50 verbs (from %mor verb|...) ===")
    for verb, cnt in verb_counter.most_common(50):
        pct = (cnt / verb_total * 100) if verb_total else 0.0
        print(f"{verb}: {cnt} ({pct:.2f}%)")

    # Plots
    if args.plots_outdir:
        outdir = Path(args.plots_outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        pd.Series(lengths).hist(bins=50)
        plt.title(f"Utterance length (tokens) – {dataset_label}")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outdir / f"{dataset_label}_utt_length_hist.png")
        plt.close()

        plt.figure()
        pd.Series(tag_len).hist(bins=50)
        plt.title(f"Number of %mor tags per utterance – {dataset_label}")
        plt.xlabel("Tags")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outdir / f"{dataset_label}_mor_tag_count_hist.png")
        plt.close()

        # Speaker role bar chart
        plt.figure(figsize=(8, 4))
        roles, counts = zip(*speaker_counts.most_common()) if speaker_counts else ([], [])
        plt.bar(roles, counts)
        plt.title(f"Speaker role distribution – {dataset_label}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"{dataset_label}_speaker_roles.png")
        plt.close()

        # Top tag frequency (top 50)
        if tag_counter:
            plt.figure(figsize=(10, 4))
            top_tags = tag_counter.most_common(50)
            tags, tcounts = zip(*top_tags)
            plt.bar(tags, tcounts)
            plt.title(f"Top %mor tags (counts) – {dataset_label}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(outdir / f"{dataset_label}_mor_tags_top50.png")
            plt.close()

        # Age vs length scatter (if available)
        if not age_len_df.empty:
            plt.figure(figsize=(6, 4))
            plt.scatter(age_len_df["age_months"], age_len_df["utt_len"], alpha=0.3, s=8)
            plt.title(f"Age (months) vs utterance length – {dataset_label}")
            plt.xlabel("Age (months)")
            plt.ylabel("Utterance length (tokens)")
            plt.tight_layout()
            plt.savefig(outdir / f"{dataset_label}_age_vs_length.png")
            plt.close()


if __name__ == "__main__":
    main()
