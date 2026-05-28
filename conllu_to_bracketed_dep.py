#!/usr/bin/env python3
import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert CoNLL-U dependency parses to bracketed dependency trees."
    )
    p.add_argument("--input_conllu", required=True, help="Path to input CoNLL-U file")
    p.add_argument(
        "--output",
        default=None,
        help="Optional output file. If omitted, writes to stdout.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print trees with indentation (one tree per block).",
    )
    p.add_argument(
        "--with_comments",
        action="store_true",
        help="Include # sent_id / # text comment lines above each tree.",
    )
    return p.parse_args()


def _sanitize_label(text):
    if text is None:
        return "X"
    s = str(text).strip()
    if not s or s == "_":
        return "X"
    return s


def _sanitize_word(text):
    if text is None:
        return "_"
    s = str(text)
    if s == "":
        return "_"
    # Avoid raw parentheses which break bracketed tree readers.
    s = s.replace("(", "-LRB-").replace(")", "-RRB-")
    return s


def _node_to_brackets(node, pretty=False, indent=0):
    label = _sanitize_label(node.get("deprel", "dep"))
    pos = _sanitize_label(node.get("upos", "X"))
    word = _sanitize_word(node.get("form", "_"))
    children = node.get("children", [])

    if not pretty:
        if children:
            kids = " ".join(_node_to_brackets(c, pretty=False) for c in children)
            return f"({label} ({pos} {word}) {kids})"
        return f"({label} ({pos} {word}))"

    ind = "  " * indent
    if not children:
        return f"{ind}({label} ({pos} {word}))"
    lines = [f"{ind}({label} ({pos} {word})"]
    for c in children:
        lines.append(_node_to_brackets(c, pretty=True, indent=indent + 1))
    lines.append(f"{ind})")
    return "\n".join(lines)


def _roots_to_brackets(roots, pretty=False):
    if not roots:
        return "(ROOT)"
    if not pretty:
        kids = " ".join(_node_to_brackets(r, pretty=False) for r in roots)
        return f"(ROOT {kids})"
    lines = ["(ROOT"]
    for r in roots:
        lines.append(_node_to_brackets(r, pretty=True, indent=1))
    lines.append(")")
    return "\n".join(lines)


def _parse_conllu_sentences(path):
    sent_lines = []
    comments = []

    def flush():
        nonlocal sent_lines, comments
        if not sent_lines and not comments:
            return None
        lines = sent_lines
        comms = comments
        sent_lines = []
        comments = []
        return comms, lines

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                item = flush()
                if item:
                    yield item
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            sent_lines.append(line)
        item = flush()
        if item:
            yield item


def _build_tree(sent_lines):
    nodes = {}
    heads = {}

    for line in sent_lines:
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        tid = fields[0]
        # Skip multi-word tokens (e.g., 1-2) and empty nodes (e.g., 3.1)
        if "-" in tid or "." in tid:
            continue
        try:
            tid = int(tid)
        except ValueError:
            continue

        form = fields[1]
        upos = fields[3]
        head = fields[6]
        deprel = fields[7]
        try:
            head = int(head)
        except ValueError:
            head = 0

        node = {
            "id": tid,
            "form": form,
            "upos": upos,
            "deprel": deprel if deprel and deprel != "_" else "dep",
            "children": [],
        }
        nodes[tid] = node
        heads[tid] = head

    # Attach children
    roots = []
    for tid, node in nodes.items():
        head = heads.get(tid, 0)
        if head == 0 or head not in nodes:
            roots.append(node)
        else:
            nodes[head]["children"].append(node)

    # Deterministic ordering
    for node in nodes.values():
        node["children"].sort(key=lambda n: n["id"])
    roots.sort(key=lambda n: n["id"])
    return roots


def main():
    args = parse_args()

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        first = True
        for comments, sent_lines in _parse_conllu_sentences(args.input_conllu):
            roots = _build_tree(sent_lines)
            tree = _roots_to_brackets(roots, pretty=args.pretty)
            if not first:
                out_fh.write("\n\n" if args.pretty else "\n")
            first = False
            if args.with_comments and comments:
                for c in comments:
                    out_fh.write(c + "\n")
            out_fh.write(tree)
        if args.output:
            out_fh.write("\n")
    finally:
        if args.output and out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
