#!/usr/bin/env python3
"""
Extract a grammar from PTB-style trees.
Input is one bracketed tree per line (Penn Treebank style).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


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


def write_yields(
    lines: Iterable[str],
    out_path: Path,
    skip_bad: bool,
) -> None:
    try:
        from nltk import Tree  # type: ignore
    except ImportError as e:
        raise SystemExit("nltk is required for yield extraction. Install with: pip install nltk") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for line in lines:
            try:
                tree = Tree.fromstring(line)
            except Exception:
                if skip_bad:
                    bad += 1
                    continue
                raise
            tokens = tree.leaves()
            tokens = [t.lower() for t in tokens]
            fout.write(" ".join(tokens) + "\n")
    if bad:
        print(f"Skipped {bad} unparsable lines.")
    print(f"Wrote yields to {out_path}")


def normalize_counts(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
    min_count: int,
) -> Dict[str, List[Tuple[Tuple[str, ...], float]]]:
    by_lhs: Dict[str, List[Tuple[Tuple[str, ...], int]]] = defaultdict(list)
    totals: Dict[str, int] = Counter()
    nonterminals = {lhs for (lhs, _) in counts.keys()}
    for (lhs, rhs), c in counts.items():
        is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
        if not is_lex and c < min_count:
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
    nonterminals = {lhs for (lhs, _) in counts.keys()}
    for (lhs, rhs), c in counts.items():
        is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
        if not is_lex and c < min_count:
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


def drop_self_unary_rules(
    counts: Dict[Tuple[str, Tuple[str, ...]], int],
) -> Dict[Tuple[str, Tuple[str, ...]], int]:
    filtered: Dict[Tuple[str, Tuple[str, ...]], int] = Counter()
    removed = 0
    for (lhs, rhs), c in counts.items():
        if len(rhs) == 1 and rhs[0] == lhs:
            removed += 1
            continue
        filtered[(lhs, rhs)] = c
    if removed:
        print(f"Dropped {removed} self-unary rules (A -> A).")
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
    weight: float,
    pseudocount: float,
    weight_mode: str,
    rule_type: str,
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
            if weight_mode == "percentage":
                rules = [(rhs, prob, 0) for rhs, prob in probs.get(lhs, [])]
            else:
                rules = [(rhs, 0.0, c) for rhs, c in counts_by_lhs.get(lhs, [])]
            for rhs, prob, count in rules:
                is_lex = len(rhs) == 1 and rhs[0] not in nonterminals
                rhs_str = " ".join(rhs)
                if weight_mode == "none":
                    line = f"{lhs} --> {rhs_str}\n"
                else:
                    if weight_mode == "percentage":
                        line = f"{prob}  {lhs} --> {rhs_str}\n"
                    elif weight_mode == "counts":
                        line = f"{count}  {lhs} --> {rhs_str}\n"
                    elif weight_mode == "uniform":
                        line = f"{weight}  {lhs} --> {rhs_str}\n"
                    elif weight_mode == "uniform_vb":
                        line = f"{weight} {pseudocount} {lhs} --> {rhs_str}\n"
                    else:
                        line = f"{prob} {pseudocount} {lhs} --> {rhs_str}\n"

                if is_lex:
                    lex_lines.append(line)
                else:
                    prod_lines.append(line)

        if rule_type in {"full", "productions"}:
            fout.writelines(prod_lines)
        if rule_type in {"full", "lexicon"}:
            fout.writelines(lex_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a grammar from PTB trees.")
    ap.add_argument("--input", help="Input file (one PTB-style tree per line).")
    ap.add_argument(
        "--input-dir",
        help="Directory containing input files (each with one tree or tagged sentence per line).",
    )
    ap.add_argument("--output", help="Output grammar file.")
    ap.add_argument(
        "--extract-yields",
        action="store_true",
        help="Write yields (tokenized sentences) instead of a grammar.",
    )
    ap.add_argument(
        "--yields-output",
        help="Output yields file (required with --extract-yields).",
    )
    ap.add_argument("--skip-bad", action="store_true", help="Skip lines that fail to parse (with --extract-yields).")

    ap.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help=(
            "Weight value for VB [Weight [Pseudocount]] Parent --> Child1 ... Childn (default: 1.0)."
        ),
    )
    ap.add_argument(
        "--pseudocount",
        type=float,
        default=0.1,
        help=(
            "Pseudocount value for VB [Weight [Pseudocount]] Parent --> Child1 ... Childn (default: 0.1)."
        ),
    )

    ap.add_argument(
        "--weight-mode",
        dest="weight_mode",
        choices=["percentage", "counts", "uniform", "uniform_vb", "none"],
        default="percentage",
        help="Weight mode for rules: percentage, counts, uniform, uniform_vb, or none.",
    )
    ap.add_argument(
        "--drop-unary-nt",
        action="store_true",
        help="Drop unary nonterminal->nonterminal rules to avoid unary cycles.",
    )
    ap.add_argument(
        "--remove-self-unary",
        action="store_true",
        help="Drop self-unary rules of the form A -> A.",
    )
    ap.add_argument(
        "--min-freq",
        dest="min_count",
        type=int,
        default=1,
        help="Minimum count to keep a rule (default: 1).",
    )
    ap.add_argument(
        "--rule-type",
        choices=["full", "productions", "lexicon"],
        default="full",
        help="Output either the full grammar, productions only, or lexicon only (default: full).",
    )

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
    if args.extract_yields:
        if not args.yields_output:
            raise SystemExit("Specify --yields-output when using --extract-yields.")
        write_yields(lines, Path(args.yields_output), args.skip_bad)
        return
    if not args.output:
        raise SystemExit("Specify --output for grammar extraction.")

    counts, root = build_from_trees(lines)

    if not counts:
        raise SystemExit("No rules extracted. Check input format and mode.")

    if args.drop_unary_nt:
        counts = drop_unary_nt_rules(counts, root)
        counts = prune_undefined_nts(counts)
    if args.remove_self_unary:
        counts = drop_self_unary_rules(counts)
    counts = filter_productions_by_count(counts, args.min_count)

    weight_mode = args.weight_mode

    probs = normalize_counts(counts, args.min_count)
    counts_by_lhs = group_counts(counts, args.min_count)
    write_grammar(
        Path(args.output),
        probs,
        counts_by_lhs,
        root,
        args.weight,
        args.pseudocount,
        weight_mode,
        args.rule_type,
    )
    print(f"Wrote {sum(len(v) for v in probs.values())} rules to {args.output}")


if __name__ == "__main__":
    main()
