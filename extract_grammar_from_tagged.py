#!/usr/bin/env python3
"""
Extract a grammar from PTB-style trees or tagged sentences.

Modes:
  tree   - input is one bracketed tree per line (Penn Treebank style)
  tagged - input is one sentence per line with tokens like word/TAG or word_TAG

Output formats:
  io   - "prob bias  LHS --> RHS"
  pcfg - "LHS -> RHS [prob]" with terminals quoted
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def escape_terminal(tok: str) -> str:
    tok = tok.replace("\\", "\\\\")
    if "'" in tok:
        inner = tok.replace('"', '\\"')
        return f"\"{inner}\""
    return f"'{tok}'"


def iter_lines(paths: Sequence[Path]) -> Iterable[str]:
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                yield line


def collect_rules_from_tree(tree, counts: Dict[Tuple[str, Tuple[str, ...]], int]) -> None:
    from nltk import Tree  # type: ignore

    if not isinstance(tree, Tree):
        return
    lhs = tree.label()
    rhs: List[str] = []
    for child in tree:
        if isinstance(child, Tree):
            rhs.append(child.label())
        else:
            rhs.append(str(child))
    if rhs:
        counts[(lhs, tuple(rhs))] += 1
    for child in tree:
        if isinstance(child, Tree):
            collect_rules_from_tree(child, counts)


def build_from_trees(lines: Iterable[str]) -> Tuple[Dict[Tuple[str, Tuple[str, ...]], int], str]:
    try:
        from nltk import Tree  # type: ignore
    except ImportError as e:
        raise SystemExit("nltk is required for tree mode. Install with: pip install nltk") from e

    counts: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
    root_counts: Counter[str] = Counter()

    for line in lines:
        try:
            tree = Tree.fromstring(line)
        except Exception:
            continue
        root_counts[tree.label()] += 1
        collect_rules_from_tree(tree, counts)

    root = root_counts.most_common(1)[0][0] if root_counts else "ROOT"
    return counts, root


def build_from_tagged(
    lines: Iterable[str],
    sep: str,
    add_sent_rules: bool,
    start_symbol: str,
    strict: bool,
) -> Tuple[Dict[Tuple[str, Tuple[str, ...]], int], str]:
    counts: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
    for line in lines:
        toks = [t for t in line.split() if t]
        tags: List[str] = []
        for tok in toks:
            if sep not in tok:
                if strict:
                    raise SystemExit(f"Token missing separator '{sep}': {tok}")
                continue
            word, tag = tok.rsplit(sep, 1)
            if not word or not tag:
                continue
            counts[(tag, (word,))] += 1
            tags.append(tag)
        if add_sent_rules and tags:
            counts[(start_symbol, tuple(tags))] += 1
    return counts, start_symbol


def normalize_counts(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
    min_count: int,
) -> Dict[str, List[Tuple[Tuple[str, ...], float]]]:
    by_lhs: Dict[str, List[Tuple[Tuple[str, ...], int]]] = defaultdict(list)
    totals: Dict[str, int] = Counter()
    for (lhs, rhs), c in counts.items():
        if c < min_count:
            continue
        by_lhs[lhs].append((rhs, c))
        totals[lhs] += c

    probs: Dict[str, List[Tuple[Tuple[str, ...], float]]] = defaultdict(list)
    for lhs, rhs_list in by_lhs.items():
        total = totals[lhs]
        for rhs, c in rhs_list:
            probs[lhs].append((rhs, c / total if total else 0.0))
        probs[lhs].sort(key=lambda x: x[1], reverse=True)
    return probs


def group_counts(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
    min_count: int,
) -> Dict[str, List[Tuple[Tuple[str, ...], int]]]:
    by_lhs: Dict[str, List[Tuple[Tuple[str, ...], int]]] = defaultdict(list)
    for (lhs, rhs), c in counts.items():
        if c < min_count:
            continue
        by_lhs[lhs].append((rhs, c))
    for lhs in by_lhs:
        by_lhs[lhs].sort(key=lambda x: (-x[1], x[0]))
    return by_lhs


def drop_unary_nt_rules(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
    root: str,
) -> Dict[Tuple[str, Tuple[str, ...]], int]:
    nonterminals = {lhs for (lhs, _) in counts.keys()}
    filtered: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
    removed = 0

    for (lhs, rhs), c in counts.items():
        if len(rhs) == 1 and rhs[0] in nonterminals:
            removed += 1
            continue
        filtered[(lhs, rhs)] = c

    if removed:
        print(f"Dropped {removed} unary NT->NT rules (root={root}).")
    return filtered


def prune_undefined_nts(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
) -> Dict[Tuple[str, Tuple[str, ...]], int]:
    filtered = counts
    while True:
        nonterminals = {lhs for (lhs, _) in filtered.keys()}
        next_counts: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
        removed = 0
        for (lhs, rhs), c in filtered.items():
            if len(rhs) == 1 and rhs[0] not in nonterminals:
                next_counts[(lhs, rhs)] = c
                continue
            if any(sym not in nonterminals for sym in rhs):
                removed += 1
                continue
            next_counts[(lhs, rhs)] = c
        if removed == 0:
            return next_counts
        print(f"Pruned {removed} rules with RHS symbols missing from LHS set.")
        filtered = next_counts


def filter_productions_by_count(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
    min_prod_count: int,
) -> Dict[Tuple[str, Tuple[str, ...]], int]:
    if min_prod_count <= 1:
        return counts
    nonterminals = {lhs for (lhs, _) in counts.keys()}
    filtered: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
    removed = 0
    for (lhs, rhs), c in counts.items():
        is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
        if not is_lex and c < min_prod_count:
            removed += 1
            continue
        filtered[(lhs, rhs)] = c
    if removed:
        print(f"Dropped {removed} production rules with count < {min_prod_count}.")
    return filtered


def write_grammar(
    out_path: Path,
    probs: Dict[str, List[Tuple[Tuple[str, ...], float]]],
    counts_by_lhs: Dict[str, List[Tuple[Tuple[str, ...], int]]],
    root: str,
    fmt: str,
    bias: float,
    weight_mode: str,
) -> None:
    with out_path.open("w", encoding="utf-8") as fout:
        lhs_keys = set(probs.keys()) | set(counts_by_lhs.keys())
        nonterminals = set(lhs_keys)
        if root in lhs_keys:
            order = [root] + [lhs for lhs in sorted(lhs_keys) if lhs != root]
        else:
            order = sorted(lhs_keys)
        prod_lines: List[str] = []
        lex_lines: List[str] = []
        for lhs in order:
            if weight_mode == "prob":
                rules = [(rhs, prob, 0) for rhs, prob in probs.get(lhs, [])]
            else:
                rules = [(rhs, 0.0, c) for rhs, c in counts_by_lhs.get(lhs, [])]
            for rhs, prob, count in rules:
                is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
                if fmt == "pcfg":
                    if len(rhs) == 1 and rhs[0].isupper() is False:
                        rhs_str = escape_terminal(rhs[0])
                    else:
                        rhs_str = " ".join(rhs)
                    if weight_mode == "none":
                        line = f"{lhs} -> {rhs_str}\n"
                    else:
                        if weight_mode == "prob":
                            weight = prob
                        elif weight_mode == "counts":
                            weight = count
                        elif weight_mode in {"uniform", "uniform_vb"}:
                            weight = 1.0
                        else:
                            weight = prob
                        line = f"{lhs} -> {rhs_str} [{weight}]\n"
                else:
                    rhs_str = " ".join(rhs)
                    if weight_mode == "none":
                        line = f"{lhs} --> {rhs_str}\n"
                    else:
                        if weight_mode == "prob":
                            line = f"{prob}  {lhs} --> {rhs_str}\n"
                        elif weight_mode == "counts":
                            line = f"{count}  {lhs} --> {rhs_str}\n"
                        elif weight_mode == "uniform":
                            line = f"1.0  {lhs} --> {rhs_str}\n"
                        elif weight_mode == "uniform_vb":
                            line = f"1.0 0.1  {lhs} --> {rhs_str}\n"
                        else:
                            line = f"{prob} {bias}  {lhs} --> {rhs_str}\n"

                if is_lex:
                    lex_lines.append(line)
                else:
                    prod_lines.append(line)

        fout.writelines(prod_lines)
        fout.writelines(lex_lines)


def write_counts_split(
    out_dir: Path,
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
    bias: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nonterminals = {lhs for (lhs, _) in counts.keys()}
    preterminals = {
        lhs
        for (lhs, rhs), _ in counts.items()
        if len(rhs) == 1 and rhs[0] not in nonterminals
    }
    prod_path = out_dir / "productions.txt"
    lex_path = out_dir / "lexicon.txt"
    nt_path = out_dir / "nonterminals.txt"
    pre_path = out_dir / "preterminals.txt"

    with prod_path.open("w", encoding="utf-8") as fprod, lex_path.open("w", encoding="utf-8") as flex:
        grouped: Dict[str, List[Tuple[Tuple[str, ...], int, bool]]] = {}
        for (lhs, rhs), c in counts.items():
            is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
            grouped.setdefault(lhs, []).append((rhs, c, is_lex))

        for lhs in sorted(grouped.keys()):
            prod_rules = [(rhs, c) for (rhs, c, is_lex) in grouped[lhs] if not is_lex]
            lex_rules = [(rhs, c) for (rhs, c, is_lex) in grouped[lhs] if is_lex]

            prod_rules.sort(key=lambda x: (-x[1], x[0]))
            lex_rules.sort(key=lambda x: (-x[1], x[0]))

            for rhs, c in prod_rules:
                rhs_str = " ".join(rhs)
                fprod.write(f"{c}  {lhs} --> {rhs_str}\n")
            for rhs, c in lex_rules:
                rhs_str = " ".join(rhs)
                flex.write(f"{c}  {lhs} --> {rhs_str}\n")

    nt_totals = {
        lhs: sum(c for (l, _), c in counts.items() if l == lhs)
        for lhs in nonterminals
    }
    pre_totals = {lhs: nt_totals[lhs] for lhs in preterminals}

    with nt_path.open("w", encoding="utf-8") as fnt:
        for lhs, total in sorted(nt_totals.items(), key=lambda x: (-x[1], x[0])):
            fnt.write(f"{total}\t{lhs}\n")

    with pre_path.open("w", encoding="utf-8") as fpre:
        for lhs, total in sorted(pre_totals.items(), key=lambda x: (-x[1], x[0])):
            fpre.write(f"{total}\t{lhs}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a grammar from PTB trees or tagged sentences.")
    ap.add_argument("--input", help="Input file (one tree or tagged sentence per line).")
    ap.add_argument(
        "--input-dir",
        help="Directory containing input files (each with one tree or tagged sentence per line).",
    )
    ap.add_argument("--output", required=True, help="Output grammar file.")
    ap.add_argument("--mode", choices=["tree", "tagged"], default="tree", help="Input mode (default: tree).")
    ap.add_argument("--format", choices=["io", "pcfg"], default="io", help="Output format (default: io).")
    ap.add_argument("--bias", type=float, default=0.0, help="Bias for io format (default: 0.0).")
    ap.add_argument("--min-count", type=int, default=1, help="Minimum count to keep a rule (default: 1).")
    ap.add_argument(
        "--no-weights",
        action="store_true",
        help="Write rules without probabilities/bias (e.g., 'LHS --> RHS').",
    )
    ap.add_argument(
        "--weight-mode",
        "--parametrisation",
        dest="weight_mode",
        choices=["prob", "percentage", "counts", "uniform", "uniform_vb", "none"],
        default="prob",
        help="Weight mode for rules: prob/percentage, counts, uniform, uniform_vb, or none.",
    )
    ap.add_argument(
        "--drop-unary-nt",
        action="store_true",
        help="Drop unary nonterminal->nonterminal rules to avoid unary cycles.",
    )
    ap.add_argument(
        "--prune-undefined-nts",
        action="store_true",
        help="Remove rules whose RHS uses symbols that never appear on the LHS.",
    )
    ap.add_argument(
        "--min-prod-count",
        type=int,
        default=1,
        help="Minimum count for non-lexical productions (default: 1).",
    )
    ap.add_argument(
        "--split-output",
        action="store_true",
        help="Create output directory and write productions.txt + lexicon.txt with counts.",
    )

    # tagged mode options
    ap.add_argument("--sep", default="/", help="Token/tag separator for tagged mode (default: /).")
    ap.add_argument("--add-sent-rules", action="store_true", help="Add sentence-level rules (S -> TAG TAG ...).")
    ap.add_argument("--start-symbol", default="S", help="Start symbol for sentence rules (default: S).")
    ap.add_argument("--strict", action="store_true", help="Fail on tokens missing the separator.")
    args = ap.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"--input-dir is not a directory: {input_dir}")
        paths = sorted([p for p in input_dir.iterdir() if p.is_file()])
        if not paths:
            raise SystemExit(f"No input files found in directory: {input_dir}")
    elif args.input:
        paths = [Path(args.input)]
    else:
        raise SystemExit("Specify --input or --input-dir.")

    lines = iter_lines(paths)
    if args.mode == "tree":
        counts, root = build_from_trees(lines)
    else:
        counts, root = build_from_tagged(lines, args.sep, args.add_sent_rules, args.start_symbol, args.strict)

    if not counts:
        raise SystemExit("No rules extracted. Check input format and mode.")

    if args.drop_unary_nt:
        counts = drop_unary_nt_rules(counts, root)
    if args.prune_undefined_nts:
        counts = prune_undefined_nts(counts)
    counts = filter_productions_by_count(counts, args.min_prod_count)

    if args.split_output:
        out_dir = Path(args.output)
        write_counts_split(out_dir, counts, args.bias)
        print(f"Wrote productions/lexicon counts to {out_dir}")
    else:
        weight_mode = "none" if args.no_weights else args.weight_mode
        if weight_mode == "percentage":
            weight_mode = "prob"
        probs = normalize_counts(counts, args.min_count)
        counts_by_lhs = group_counts(counts, args.min_count)
        write_grammar(Path(args.output), probs, counts_by_lhs, root, args.format, args.bias, weight_mode)
        print(f"Wrote {sum(len(v) for v in probs.values())} rules to {args.output}")


if __name__ == "__main__":
    main()
