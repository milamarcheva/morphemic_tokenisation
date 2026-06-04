#!/usr/bin/env python3
"""
Write top lexicalisations per preterminal from a grammar with counts.
Input grammar can be a full grammar or a lexicon-only file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_count(token: str) -> float | None:
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return None


def _format_count(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_grammar_with_counts(
    lines: Iterable[str],
) -> List[Tuple[float, str, Tuple[str, ...]]]:
    rules: List[Tuple[float, str, Tuple[str, ...]]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if "-->" not in tokens:
            continue
        arrow_idx = tokens.index("-->")
        if arrow_idx < 2 or arrow_idx >= len(tokens) - 1:
            continue
        count = _parse_count(tokens[0])
        if count is None:
            continue
        lhs = tokens[arrow_idx - 1]
        rhs = tuple(tokens[arrow_idx + 1 :])
        rules.append((count, lhs, rhs))
    return rules


def top_lexicalisations(
    rules: Iterable[Tuple[float, str, Tuple[str, ...]]],
    top_n: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    rules_list = list(rules)
    nonterminals = {lhs for _, lhs, _ in rules_list}
    lex_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for count, lhs, rhs in rules_list:
        if len(rhs) != 1:
            continue
        if rhs[0] in nonterminals:
            continue
        lex_counts[lhs][rhs[0]] += count
    top_by_lhs: Dict[str, List[Tuple[str, float]]] = {}
    for lhs, counts in lex_counts.items():
        sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top_by_lhs[lhs] = sorted_items[:top_n]
    return top_by_lhs


def write_top_lexicalisations(
    grammar_path: Path,
    output_path: Path | None = None,
    top_n: int = 5,
) -> Path:
    with grammar_path.open(encoding="utf-8") as f:
        rules = parse_grammar_with_counts(f)
    output_path = output_path or grammar_path.with_name("top_lexicalisations.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_by_lhs = top_lexicalisations(rules, top_n=top_n)
    with output_path.open("w", encoding="utf-8") as fout:
        for lhs in sorted(top_by_lhs):
            items = top_by_lhs[lhs]
            limit = 50 if lhs in {"VB", "NN"} else top_n
            for lex, count in items[:limit]:
                fout.write(f"{_format_count(count)}  {lhs} --> {lex}\n")
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write top lexicalisations per preterminal from a grammar with counts."
    )
    ap.add_argument("--input", required=True, help="Input grammar file with counts.")
    ap.add_argument(
        "--output",
        help="Output file path (default: top_lexicalisations.txt next to input).",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of lexicalisations to keep per preterminal (default: 5).",
    )
    args = ap.parse_args()

    output_path = Path(args.output) if args.output else None
    out = write_top_lexicalisations(Path(args.input), output_path=output_path, top_n=args.top_n)
    print(f"Wrote top lexicalisations to {out}")


if __name__ == "__main__":
    main()
