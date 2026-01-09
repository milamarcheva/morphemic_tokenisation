#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, List

from nltk import Tree


def iter_lines(paths: List[Path]) -> Iterable[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield line


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract yields from PTB trees (one tree per line).")
    ap.add_argument("--input", help="Input file (one PTB tree per line).")
    ap.add_argument("--input-dir", help="Directory of input files.")
    ap.add_argument("--output", required=True, help="Output yields file.")
    ap.add_argument("--lower", action="store_true", help="Lowercase yields.")
    ap.add_argument("--skip-bad", action="store_true", help="Skip lines that fail to parse.")
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bad = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for line in iter_lines(paths):
            try:
                tree = Tree.fromstring(line)
            except Exception:
                if args.skip_bad:
                    bad += 1
                    continue
                raise
            tokens = tree.leaves()
            if args.lower:
                tokens = [t.lower() for t in tokens]
            fout.write(" ".join(tokens) + "\n")

    if bad:
        print(f"Skipped {bad} unparsable lines.")
    print(f"Wrote yields to {out_path}")


if __name__ == "__main__":
    main()
