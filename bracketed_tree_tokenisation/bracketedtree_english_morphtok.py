{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
"""\
Utilities for preprocessing CHILDES Treebank parses:\
- Strip animacy/theta annotations and traces\
- Normalize punctuation/terminals\
- (Optionally) transform morphology into base+affix trees\
- Split datasets (train/valid/test) in a CHILDES-compatible way\
"""\
\
from __future__ import annotations\
\
import argparse\
import logging\
import re\
from pathlib import Path\
from typing import Dict, Iterable, List, Sequence, Tuple\
\
import nltk\
import pandas as pd\
import spacy\
from nltk import Tree\
\
# --------------------------- #\
# Constants & precompiled REs #\
# --------------------------- #\
\
ED_EXCEPTIONS = frozenset(\{\
    'aran_reed', 'bed', 'birdfeed', 'birdseed', 'bled', 'bleed',\
    'bunkbed', 'captain_mildred', 'ed','feed', 'fireman_fred', 'flatbed',\
    'fred', 'get_into_bed', 'go_to_bed', 'hundred', 'indeed', 'jed',\
    'knockkneed', 'led', 'mildred', 'mister_reed', 'ned', 'need',\
    'old_macreed', 'playbed', 'red', 'reed', 'reseed',\
    'robber_red', 'shed', 'sled', 'sofabed', 'ted', 'thoroughbred',\
    'wilfred'\
\})\
\
ING_EXCEPTIONS = frozenset(\{\
    'aboing', 'anything', 'bing', 'blingie^bling^bling', 'boing',\
    'boing^boing', 'boingaboingaboingaboing', 'boingboingboing',\
    'boingeyeeyboingeyboing', 'bring', 'burger_king', 'building',\
    'ceiling', 'chi_ling', 'cunning', 'diddle_diddle_dumpling',\
    'ding', 'ding^ding', 'dingading', 'dingaling', 'dingalingaling',\
    'dingding', 'dingdingdingding', 'dingdingdingdingding', 'during',\
    'earling', 'everything', 'herring', 'i_spy_everything', 'keyring',\
    'king', 'lion_king', 'ming', 'morning', 'nothing', 'ping', 'ring',\
    'ring^ring', 'ring^ring^ring', 'ringading', 'ruby_ring', 'sing',\
    'sling', 'something', 'spring', 'sting', 'swing', 'thing', 'ting', 'wing'\
\})\
\
PLURAL_ONLY = frozenset(\{\
    'binoculars', 'headphones', 'sunglasses', 'glasses', 'scissors', 'tweezers',\
    'jeans', 'pyjamas', 'tights', 'knickers', 'shorts', 'trousers', 'pants',\
    'belongings', 'outskirts', 'clothes', 'premises', 'congratulations',\
    'savings', 'earnings', 'stairs', 'goods', 'surroundings', 'thanks', 'yours'\
\})\
\
IRREGULAR_3RD_PERSON_VERBS = frozenset(\{'be', 'have', 'go', 'do'\})\
\
CORPORA_WITH_THEMATIC_ANNOTATION = frozenset(\{\
    'ctb_brown-adam4up+animacy+theta',\
    'ctb_brown-adam3to4+animacy+theta',\
    'ctb_brown-eve+animacy+theta',\
    'ctb_valian+animacy+theta',\
\})\
\
PLURAL_PTS = frozenset(\{"NNS", "NNPS"\})\
VERB_3RD_PTS = frozenset(\{"VBZ", "AUX", "VBD"\})\
\
# Regexes\
RE_TRACE_INDEX = re.compile(r'-\\d+')\
RE_TAG_WITH_ANN = re.compile(r'\\((\\w+)(-[^ ]+)?\\s')  # (TAG-<...>  -> (TAG\
RE_PUNCT_CONSTIT = re.compile(r'\\((,|\\.|:|``|\\'\\'|\\'|\\?|!|;)\\s[^\\)]+\\)')\
RE_TERMINAL = re.compile(r'(\\([A-Z$]+ )([^\\(\\) ]+)\\)')  # (TAG token)\
RE_FIND_CONSTIT = re.compile(r'\\(([^()]+)\\)')\
\
\
# --------------------- #\
# Lazy spaCy model load #\
# --------------------- #\
\
_nlp = None\
\
\
def nlp() -> "spacy.language.Language":\
    """Load spaCy English model once (lazy)."""\
    global _nlp\
    if _nlp is None:\
        _nlp = spacy.load("en_core_web_lg")\
    return _nlp\
\
\
# ------------------------------- #\
# Tree/annotation cleaning utils  #\
# ------------------------------- #\
\
def remove_unary_subtrees_with_T_terminals(tree: Tree) -> Tree | None:\
    """\
    Remove unary subtrees whose single child contains a terminal starting with '*'\
    (e.g., traces like *T*).\
    """\
    if not isinstance(tree, Tree):\
        return tree\
\
    new_children: List[Tree] = []\
    for child in tree:\
        processed = remove_unary_subtrees_with_T_terminals(child)\
        if processed is not None:\
            new_children.append(processed)\
\
    if (\
        len(new_children) == 1\
        and isinstance(new_children[0], Tree)\
        and any(str(t).startswith('*') for t in new_children[0].leaves())\
    ):\
        return None\
\
    return Tree(tree.label(), new_children)\
\
\
def clear_trace(parse: str) -> str:\
    """\
    Remove '-1' style indices from traces and drop unary subtrees headed by a trace.\
    Returns a pretty-printed tree string or a single space on failure.\
    """\
    cleaned = RE_TRACE_INDEX.sub('', parse)\
    try:\
        nltk_tree = Tree.fromstring(cleaned)\
        pruned = remove_unary_subtrees_with_T_terminals(nltk_tree)\
        return pruned.pformat() if pruned is not None else " "\
    except Exception:\
        # Only a small number of problematic parses; skip gracefully.\
        return " "\
\
\
def clear_animacy_and_theta(text: str) -> str:\
    """\
    Remove animacy/theta annotations like -<ANIM>-<PATIENT-V1> from node labels.\
    Keeps the base tag, e.g. (NP-<ANIM> ... -> (NP ...).\
    """\
    def repl(m: re.Match) -> str:\
        return f'(\{m.group(1)\} '\
    return RE_TAG_WITH_ANN.sub(repl, text)\
\
\
def remove_punctuation_constituents(parse_string: str) -> str:\
    """Remove punctuation-only constituents and collapse extra whitespace."""\
    cleaned = RE_PUNCT_CONSTIT.sub('', str(parse_string))\
    return re.sub(r'\\s+', ' ', cleaned).strip()\
\
\
def lowercase_terminals(parse_string: str) -> str:\
    """Lowercase tokens at terminal positions without touching POS tags."""\
    def to_lower(m: re.Match) -> str:\
        return f'\{m.group(1)\}\{m.group(2).lower()\})'\
    return RE_TERMINAL.sub(to_lower, parse_string)\
\
\
def collapse_to_one_line(text: str) -> str:\
    """Collapse whitespace and newlines to a single space."""\
    return re.sub(r'\\s+', ' ', text).strip()\
\
\
# -------------------------------- #\
# Morphological tree transformation #\
# -------------------------------- #\
\
def _get_lemma(doc: "spacy.tokens.Doc", token_text: str) -> str | None:\
    for t in doc:\
        if t.lower_ == token_text.lower():\
            return t.lemma_\
    return None\
\
\
def transform_tree(tree_str: str, conservative: bool = False) -> str:\
    """\
    Expand certain terminals into base+affix structures using spaCy lemmas:\
      - VBG/VBN/ed/ing -> (VB base) (ASP/T ing/ed)\
      - Regular plural NNS/NNPS -> (NN base) (DIV s)\
      - Regular VBZ/AUX/VBD with -s (except irregular set) -> (VB base) (PRS s)\
\
    If conservative=True, only change tokens whose lemma differs from surface.\
    """\
    if not tree_str.strip():\
        return " "\
\
    nltk_tree = Tree.fromstring(tree_str)\
    tree_yield = ' '.join(nltk_tree.leaves())\
    doc = nlp()(tree_yield)\
\
    def walk(sub: Tree) -> Tree:\
        if isinstance(sub, Tree):\
            # pre-terminal\
            if len(sub) == 1 and isinstance(sub[0], str):\
                word = sub[0]\
                lemma = _get_lemma(doc, word) or word\
\
                if lemma == word and conservative:\
                    return sub\
\
                # -ing\
                if word.endswith('ing') and word not in ING_EXCEPTIONS:\
                    base = lemma\
                    if base.endswith('ing') and base not in ING_EXCEPTIONS:\
                        base = base[:-3]\
                    return Tree(sub.label(), [Tree('VB', [base]), Tree('ASP', ['ing'])])\
\
                # -ed\
                if word.endswith('ed') and word not in ED_EXCEPTIONS:\
                    base = lemma\
                    if base.endswith('ed') and base not in ED_EXCEPTIONS:\
                        base = base[:-2]\
                    return Tree(sub.label(), [Tree('VB', [base]), Tree('T', ['ed'])])\
\
                # -s plurals / 3sg\
                if word.endswith('s'):\
                    if sub.label() in PLURAL_PTS and word not in PLURAL_ONLY:\
                        base_n = lemma\
                        return Tree(sub.label(), [Tree(sub.label()[:-1], [base_n]), Tree('DIV', ['s'])])\
                    if sub.label() in VERB_3RD_PTS and lemma not in IRREGULAR_3RD_PERSON_VERBS:\
                        base_v = lemma\
                        return Tree(sub.label(), [Tree('VB', [base_v]), Tree('PRS', ['s'])])\
\
            return Tree(sub.label(), [walk(c) for c in sub])\
        return sub\
\
    transformed = walk(nltk_tree)\
    return ' '.join(str(transformed).split())\
\
\
def transform_tree_conservative(tree_str: str) -> str:\
    return transform_tree(tree_str, conservative=True)\
\
\
# ------------------------- #\
# Constituents search utils #\
# ------------------------- #\
\
def find_constituents_with_ending(tree: str, ending: str) -> List[str]:\
    """Return constituents where a word ends with the given suffix (simple regex-based)."""\
    constituents = RE_FIND_CONSTIT.findall(tree)\
    pattern = re.compile(r'\\b\\w+' + re.escape(ending) + r'\\b')\
\
    out: List[str] = []\
    for c in constituents:\
        if pattern.search(c):\
            out.append(c)\
    return out\
\
\
def find_ing_ed_constituents(tree: str) -> Tuple[List[str], List[str]]:\
    """Return (ing_constituents, ed_constituents) based on simple regex scans."""\
    constituents = RE_FIND_CONSTIT.findall(tree)\
    ings, eds = [], []\
    for c in constituents:\
        if re.search(r'\\b\\w+ing\\b', c):\
            ings.append(c)\
        elif re.search(r'\\b\\w+ed\\b', c):\
            eds.append(c)\
    return ings, eds\
\
\
def find_preterminals(parses: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:\
    """Return dicts of \{PRETERMINAL_LABEL: parse_string\} for -ing and -ed (excluding exceptions)."""\
    ing_pre, ed_pre = \{\}, \{\}\
    for p in parses:\
        ings, eds = find_ing_ed_constituents(p)\
        for ing in ings:\
            label, token = ing.split()[0], ing.split()[1].lower()\
            if token not in ING_EXCEPTIONS and label not in ing_pre:\
                ing_pre[label] = p\
        for ed in eds:\
            label, token = ed.split()[0], ed.split()[1].lower()\
            if token not in ED_EXCEPTIONS and label not in ed_pre:\
                ed_pre[label] = p\
    return ing_pre, ed_pre\
\
\
def find_preterminals_with_ending(parses: Iterable[str], ending: str) -> Dict[str, str]:\
    """Return \{PRETERMINAL_LABEL: parse_string\} for a given word ending (excluding ED exceptions)."""\
    out: Dict[str, str] = \{\}\
    for p in parses:\
        for c in find_constituents_with_ending(p, ending):\
            label, token = c.split()[0], c.split()[1].lower()\
            if token not in ED_EXCEPTIONS and label not in out:\
                out[label] = p\
    return out\
\
\
# ---------------------- #\
# I/O & dataset splitting #\
# ---------------------- #\
\
def save_file(lines: Sequence[str], name: str, output_dir: Path) -> None:\
    output_dir.mkdir(parents=True, exist_ok=True)\
    path = output_dir / name\
    with path.open("w", encoding="utf-8") as f:\
        for l in lines:\
            f.write(l + "\\n")\
\
\
def train_val_test_parses(\
    df: pd.DataFrame,\
    col: str = "morph_tok_parses_lower_nopunct",\
    extension: str = "",\
    output_dir: Path = Path("resources"),\
) -> None:\
    """\
    Split as:\
      - Train/Valid: all non Brown-Adam parses (85/15)\
      - Test: all Brown-Adam parses\
    """\
    brown_adam = df[df.corpus.str.startswith("ctb_brown-adam")][col].astype(str).tolist()\
    non_adam = df[~df.corpus.str.startswith("ctb_brown-adam")][col].astype(str).tolist()\
\
    train_end = int(0.85 * len(non_adam))\
    train = non_adam[:train_end]\
    valid = non_adam[train_end:]\
\
    save_file(train, f"ctb-train\{extension\}.txt", output_dir)\
    save_file(valid, f"ctb-valid\{extension\}.txt", output_dir)\
    save_file(brown_adam, f"ctb-test\{extension\}.txt", output_dir)\
\
\
# -------------- #\
# CLI / main run #\
# -------------- #\
\
def parse_args() -> argparse.Namespace:\
    p = argparse.ArgumentParser(description="Preprocess CHILDES TB parses.")\
    p.add_argument(\
        "-p", "--path",\
        required=True,\
        help="Path to CSV (e.g., resources/df_ctb.csv)",\
    )\
    p.add_argument(\
        "-o", "--output-dir",\
        default="resources",\
        help="Directory to write output splits (default: resources)",\
    )\
    p.add_argument(\
        "--spacy-model",\
        default="en_core_web_lg",\
        help="spaCy English model to load (default: en_core_web_lg)",\
    )\
    p.add_argument(\
        "--run-tests",\
        action="store_true",\
        help="Run lightweight transformation sanity tests.",\
    )\
    return p.parse_args()\
\
\
def run_sanity_tests() -> None:\
    """Minimal assertions mirroring the original examples."""\
    t1 = "(ROOT (S (NP (PRP i)) (VP (AUX was) (VP (VBG crossing) (NP (DT the) (NN street)))) ))"\
    t1_morph_tok  = "(ROOT (S (NP (PRP i)) (VP (AUX was) (VP (VBG (VB cross) (ASP ing)) (NP (DT the) (NN street))))))"\
\
    t2 = "(ROOT (S (NP (PRP you)) (VP (AUX 've) (VP (VBN used) (PRT (RP up)) (ADJP (JJ all) (PP (IN of) (NP (DT the) (NN tape)))))) ))"\
    t2_morph_tok = "(ROOT (S (NP (PRP you)) (VP (AUX 've) (VP (VBN (VB use) (T ed)) (PRT (RP up)) (ADJP (JJ all) (PP (IN of) (NP (DT the) (NN tape))))))))"\
\
    t3 = "(ROOT (FRAG (NP (NNS gifts)) ))"\
    t3_morph_tok = "(ROOT (FRAG (NP (NNS (NN gift) (DIV s)))))"\
\
    t4 = "(ROOT (S (NP (NNP goldilocks)) (VP (VBZ runs) (PP (ADVP (RB away)) (IN from) (NP (DT the) (CD three) (NNS bears)))) ))"\
    t4_morph_tok = "(ROOT (S (NP (NNP goldilocks)) (VP (VBZ (VB run) (PRS s)) (PP (ADVP (RB away)) (IN from) (NP (DT the) (CD three) (NNS (NN bear) (DIV s)))))))"\
\
    t5 = "(ROOT (FRAG (VP (AUX does) (NP (PRP she)) (ADVP (RB ever))) (. .)))"\
\
    t6 = "(ROOT (INTJ (UH well)) (, ,) (S (VP (AUX do) (NOT n't) (VP (COP be) (ADJP (JJ scared)))) (. .)))"\
    t6_morph_tok = "(ROOT (INTJ (UH well)) (, ,) (S (VP (AUX do) (NOT n't) (VP (COP be) (ADJP (JJ (VB scar) (T ed))))) (. .)))"\
\
    assert transform_tree(t1) == t1_morph_tok\
    assert transform_tree(t2) == t2_morph_tok\
    assert transform_tree(t3) == t3_morph_tok\
    assert transform_tree(t4) == t4_morph_tok\
    assert transform_tree(t5) == t5\
    assert transform_tree(t6, True) == t6\
    assert transform_tree_conservative(t6) == t6\
    assert transform_tree(t6) == t6_morph_tok\
\
\
def main() -> None:\
    args = parse_args()\
\
    # Configure logging\
    logging.basicConfig(\
        level=logging.INFO,\
        format="%(levelname)s: %(message)s"\
    )\
\
    # Load spaCy model (allow override)\
    if args.spacy_model != "en_core_web_lg":\
        # Replace the lazy loader globally if a different model is specified\
        global _nlp\
        _nlp = spacy.load(args.spacy_model)\
\
    csv_path = Path(args.path)\
    output_dir = Path(args.output_dir)\
\
    logging.info("Reading %s", csv_path)\
    df = pd.read_csv(csv_path)\
\
    # Core preprocessing pipeline\
    logging.info("Cleaning animacy/theta annotations...")\
    df["morph_tok_parses_lower_nopunct"] = df["gra"].astype(str).apply(clear_animacy_and_theta)\
\
    logging.info("Removing punctuation constituents...")\
    df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(remove_punctuation_constituents)\
\
    logging.info("Lowercasing terminals...")\
    df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(lowercase_terminals)\
\
    logging.info("Clearing traces...")\
    df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(clear_trace)\
\
    # Keep a collapsed, single-line variant\
    df["normal_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(collapse_to_one_line)\
\
    logging.info("Creating train/valid/test splits...")\
    train_val_test_parses(df, col="normal_tok_parses_lower_nopunct", extension="_normal", output_dir=output_dir)\
\
    # Optional sanity checks\
    if args.run_tests:\
        logging.info("Running sanity tests...")\
        run_sanity_tests()\
        logging.info("Sanity tests passed.")\
\
    # Peek\
    logging.info("Columns available: %s", list(df.columns))\
    logging.info("Done.")\
\
\
if __name__ == "__main__":\
    main()\
}