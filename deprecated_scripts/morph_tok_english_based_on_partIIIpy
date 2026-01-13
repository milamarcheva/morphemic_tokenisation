#!/usr/bin/env python3
"""
Morphemic tokeniser + Brown-order feature extractor.

Input:  .txt (one sentence per line)
Output: .txt (one tokenised sentence per line)
Optional: CSV with Brown-order feature counts per line.

Brown features approximated with spaCy POS/morph/deps:
  1. Present progressive "-ing"                 -> morpheme_ing
  2. Prepositions "in", "on"                   -> prep_in, prep_on
  3. Plural -s (regular)                       -> plural_s
  4. Possessive 's                             -> possessive_s
  5. Irregular past (VBD not ending in "ed")   -> irregular_past
  6. Regular past -ed                          -> regular_past
  7. Articles (a/an/the)                       -> articles
  8. 3rd person present regular -s             -> third_person_s
  9. 3rd person present irregular              -> third_person_irregular  (do/have/say)
 10. Contractible auxiliary (am/is/are + VBG)  -> contractible_aux
 11. Contractible copula (am/is/are as cop)    -> contractible_cop
 12. Uncontractible auxiliary (was/were + VBG) -> uncontractible_aux
 13. Uncontractible copula (was/were as cop)   -> uncontractible_cop

Usage:
    python morph_tokenize_brown.py -i input.txt -o output.txt \
        --brown-features outputs/brown_features.csv
Options:
    --include-punct   Keep punctuation in the tokenised text (default: drop)
    --case keep       Preserve original casing (default: lower)
    --spacy-model     spaCy model name (default: en_core_web_lg)

Requires:
    pip install spacy pandas
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import spacy

# --------------------------- #
# Exception lists (from NB)   #
# --------------------------- #

ED_EXCEPTIONS = {
    'aran_reed', 'bed', 'birdfeed', 'birdseed', 'bled', 'bleed',
    'bunkbed', 'captain_mildred', 'ed', 'feed', 'fireman_fred', 'flatbed',
    'fred', 'get_into_bed', 'go_to_bed', 'hundred', 'indeed', 'jed',
    'knockkneed', 'led', 'mildred', 'mister_reed', 'ned', 'need',
    'old_macreed', 'playbed', 'red', 'reed', 'reseed',
    'robber_red', 'shed', 'sled', 'sofabed', 'ted', 'thoroughbred',
    'wilfred'
}

ING_EXCEPTIONS = {
    'aboing', 'anything', 'bing', 'blingie^bling^bling', 'boing',
    'boing^boing', 'boingaboingaboingaboing', 'boingboingboing',
    'boingeyeeyboingeyboing', 'bring', 'burger_king', 'building',
    'ceiling', 'chi_ling', 'cunning',
    'diddle_diddle_dumpling', 'ding', 'ding^ding', 'dingading',
    'dingaling', 'dingalingaling', 'dingding', 'dingdingdingding',
    'dingdingdingdingding', 'during', 'earling', 'everything', 'herring',
    'i_spy_everything', 'keyring', 'king', 'lion_king', 'ming', 'morning',
    'nothing', 'ping', 'ring', 'ring^ring', 'ring^ring^ring',
    'ringading', 'ruby_ring', 'sing', 'sling', 'something',
    'spring', 'sting', 'swing', 'thing', 'ting', 'wing'
}

S_EXCEPTIONS = {
    'binoculars', 'headphones', 'sunglasses', 'glasses',
    'scissors', 'tweezers', 'jeans', 'pyjamas', 'tights',
    'knickers', 'shorts', 'trousers', 'pants', 'belongings',
    'outskirts', 'clothes', 'premises', 'congratulations',
    'savings', 'earnings', 'stairs', 'goods', 'surroundings',
    'thanks', 'yours'
}

IRREGULAR_3RD_PERSON_VERBS = {'be', 'have', 'go', 'do'}
BROWN_3SG_IRREGULAR = {'do', 'have', 'say'}  # Brown's typical set for irregular 3sg

# --------------------------- #
# spaCy load                  #
# --------------------------- #

def load_spacy(model: str):
    try:
        return spacy.load(model)
    except OSError as e:
        raise SystemExit(
            f"Could not load spaCy model '{model}'. "
            "Install with: python -m spacy download en_core_web_lg"
        ) from e

# --------------------------- #
# Core processing per line    #
# --------------------------- #

FeatureDict = Dict[str, int]

BROWN_FEATURE_KEYS = [
    "morpheme_ing",
    "prep_in", "prep_on",
    "plural_s",
    "possessive_s",
    "irregular_past",
    "regular_past",
    "articles",
    "third_person_s",
    "third_person_irregular",
    "contractible_aux",
    "contractible_cop",
    "uncontractible_aux",
    "uncontractible_cop",
]

def process_line(
    text: str,
    nlp,
    include_punct: bool,
    lowercase: bool,
) -> Tuple[str, FeatureDict]:
    """
    Return (morphemically_tokenised_sentence, brown_features).
    Implements token splitting AND Brown features in one spaCy pass.
    """
    feats: FeatureDict = {k: 0 for k in BROWN_FEATURE_KEYS}

    if not text.strip():
        return "", feats

    doc = nlp(text)
    out_tokens: List[str] = []

    # Pre-scan for dependency-based Brown features
    for tok in doc:
        low = tok.text.lower()
        lem = tok.lemma_.lower()

        # Articles
        if low in {"a", "an", "the"}:
            feats["articles"] += 1

        # Prepositions in/on
        if low == "in" and tok.pos_ == "ADP":
            feats["prep_in"] += 1
        if low == "on" and tok.pos_ == "ADP":
            feats["prep_on"] += 1

        # Possessive 's
        if tok.tag_ == "POS" or low.endswith("'s") or low.endswith("’s"):
            feats["possessive_s"] += 1

        # Irregular past: VBD and not ending in "ed"
        if tok.tag_ == "VBD" and not low.endswith("ed"):
            feats["irregular_past"] += 1

        # Contractible / Uncontractible AUX & Copula (approximate patterns)
        is_be = lem == "be"
        has_vbg_child = any(ch.tag_ == "VBG" for ch in tok.subtree)  # rough
        # If tok itself is the BE form, simpler checks:
        if is_be:
            tense = tok.morph.get("Tense")
            # Auxiliary if governing/providing support to a VBG verb
            if any(ch.tag_ == "VBG" for ch in tok.children):
                if "Pres" in tense:
                    feats["contractible_aux"] += 1
                if "Past" in tense:
                    feats["uncontractible_aux"] += 1
            # Copula if it's 'cop' or links to predicate ADJ/NOUN/PRON
            head = tok.head
            if tok.dep_ == "cop" or (head and head.pos_ in {"ADJ", "NOUN", "PRON"} and not has_vbg_child):
                if "Pres" in tense:
                    feats["contractible_cop"] += 1
                if "Past" in tense:
                    feats["uncontractible_cop"] += 1

        # 3sg irregular (Brown): does/has/says
        if (
            lem in BROWN_3SG_IRREGULAR
            and tok.morph.get("Tense") == ["Pres"]
            and tok.morph.get("Person") == ["3"]
            and tok.morph.get("Number") == ["Sing"]
        ):
            feats["third_person_irregular"] += 1

    # Build morphemic tokens + counts best tied to surface/lemma
    for tok in doc:
        if tok.is_punct:
            if include_punct:
                out_tokens.append(tok.text if not lowercase else tok.text.lower())
            continue

        surface = tok.text if not lowercase else tok.text.lower()
        lemma = tok.lemma_ if not lowercase else tok.lemma_.lower()

        # Progressive -ing
        if surface.endswith("ing") and surface not in ING_EXCEPTIONS:
            base = lemma
            if base.endswith("ing") and base not in ING_EXCEPTIONS:
                base = base[:-3]
            out_tokens.extend([base, "ing"])
            feats["morpheme_ing"] += 1
            continue

        # Regular past -ed
        if surface.endswith("ed") and surface not in ED_EXCEPTIONS:
            base = lemma
            if base.endswith("ed") and base not in ED_EXCEPTIONS:
                base = base[:-2]
            out_tokens.extend([base, "ed"])
            feats["regular_past"] += 1
            continue

        # Plural -s (regular)
        if tok.tag_ in {"NNS", "NNPS"} and surface.endswith("s") and surface not in PLURAL_ONLY:
            out_tokens.extend([lemma, "s"])
            feats["plural_s"] += 1
            continue

        # 3rd person present regular -s
        if (
            tok.morph.get('Tense') == ['Pres']
            and tok.morph.get('Person') == ['3']
            and tok.morph.get('Number') == ['Sing']
            and lemma not in IRREGULAR_3RD_PERSON_VERBS
            and surface.endswith("s")
        ):
            out_tokens.extend([lemma, "s"])
            feats["third_person_s"] += 1
            continue

        # Default: keep
        out_tokens.append(surface)

    return " ".join(out_tokens), feats


def morph_tokens_from_mor(mor_line: str, lowercase: bool = True) -> List[str]:
    """
    Split a CHAT %mor line into base + affix tokens.
    Example: v|do&3S&PAST -> ['do', '+3S', '+PAST']
    """
    tokens: List[str] = []
    if not mor_line:
        return tokens
    for word in mor_line.strip().split():
        word = word.strip()
        if not word:
            continue
        if "|" in word:
            _, rest = word.split("|", 1)
        else:
            rest = word
        parts = rest.split("&")
        if not parts:
            continue
        base = parts[0]
        if base:
            tokens.append(base.lower() if lowercase else base)
        for aff in parts[1:]:
            if aff:
                tokens.append(("+" + aff.lower()) if lowercase else ("+" + aff))
    return tokens


def iter_chat_utterances(
    chat_path: Path,
    include_participants: Optional[Iterable[str]] = None,
    exclude_participants: Optional[Iterable[str]] = ("CHI",),
) -> Iterator[Tuple[str, str, Optional[str]]]:
    """
    Yield (speaker, utt_text, mor_line) for each utterance in a CHAT file.
    Defaults to child-directed speech (excludes CHI).
    """
    include_set = set(include_participants) if include_participants else None
    exclude_set = set(exclude_participants) if exclude_participants else set()

    current_speaker = None
    current_utt = None
    current_mor = None

    with chat_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("*"):
                # flush previous
                if current_speaker and current_utt is not None:
                    if (include_set is None or current_speaker in include_set) and current_speaker not in exclude_set:
                        yield current_speaker, current_utt, current_mor
                # reset
                current_mor = None
                current_speaker = line[1:].split(":", 1)[0].strip()
                if "\t" in line:
                    current_utt = line.split("\t", 1)[1].strip()
                else:
                    current_utt = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            elif line.startswith("%mor:"):
                current_mor = line.split(":", 1)[1].strip()
            else:
                continue

    # flush tail
    if current_speaker and current_utt is not None:
        if (include_set is None or current_speaker in include_set) and current_speaker not in exclude_set:
            yield current_speaker, current_utt, current_mor

# --------------------------- #
# CLI & file I/O              #
# --------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Morphemically tokenise a text file and (optionally) export Brown-order features."
    )
    p.add_argument("-i", "--input", required=True, help="Input .txt or .cha (or directory of .cha)")
    p.add_argument("-o", "--output", required=True, help="Output .txt (tokenised, one per line)")
    p.add_argument("--brown-features", help="Optional path to write a CSV of Brown features per line")
    p.add_argument("--include-punct", action="store_true", help="Keep punctuation tokens in tokenised text")
    p.add_argument("--case", choices={"lower", "keep"}, default="lower", help="Output casing (default: lower)")
    p.add_argument("--spacy-model", default="en_core_web_lg", help="spaCy model to load")
    p.add_argument(
        "--mode",
        choices={"text", "mor", "mor_spacy"},
        default="text",
        help="Tokenisation source: raw text (default), %mor tokens only, or %mor tokens with spaCy features.",
    )
    p.add_argument(
        "--participants",
        help="Comma-separated participant codes to include (chat mode). Default: all except CHI.",
    )
    p.add_argument(
        "--exclude-participants",
        help="Comma-separated participant codes to exclude (chat mode). Default: CHI.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bf_path = Path(args.brown_features) if args.brown_features else None
    if bf_path:
        bf_path.parent.mkdir(parents=True, exist_ok=True)

    lowercase = args.case == "lower"
    participants = (
        [p.strip() for p in args.participants.split(",")] if args.participants else None
    )
    exclude_participants = (
        [p.strip() for p in args.exclude_participants.split(",")] if args.exclude_participants else ["CHI"]
    )

    need_spacy = args.mode in {"text", "mor_spacy"}
    nlp = load_spacy(args.spacy_model) if need_spacy else None

    logging.info("Reading: %s", in_path)

    # Determine input type
    chat_files: List[Path] = []
    if in_path.is_dir():
        chat_files = sorted(in_path.rglob("*.cha"))
    elif in_path.suffix.lower() == ".cha":
        chat_files = [in_path]

    csv_writer = None
    csvfile = None
    if bf_path:
        csvfile = bf_path.open("w", encoding="utf-8", newline="")
        fieldnames = ["line_number", "original_sentence", "tokenised_sentence"] + BROWN_FEATURE_KEYS
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        with out_path.open("w", encoding="utf-8") as fout:
            if chat_files:
                idx = 0
                for chat in chat_files:
                    for _, utt, mor in iter_chat_utterances(
                        chat, include_participants=participants, exclude_participants=exclude_participants
                    ):
                        idx += 1
                        if args.mode == "mor":
                            tokens = morph_tokens_from_mor(mor or "", lowercase=lowercase)
                            tokenised = " ".join(tokens)
                            feats: FeatureDict = {k: 0 for k in BROWN_FEATURE_KEYS}
                        elif args.mode == "mor_spacy":
                            tokens = morph_tokens_from_mor(mor or "", lowercase=lowercase)
                            spacy_input = " ".join(t for t in tokens if not t.startswith("+")) or utt
                            _, feats = process_line(
                                spacy_input, nlp=nlp, include_punct=args.include_punct, lowercase=lowercase
                            )
                            tokenised = " ".join(tokens)
                        else:  # text
                            tokenised, feats = process_line(
                                utt, nlp=nlp, include_punct=args.include_punct, lowercase=lowercase
                            )

                        fout.write(tokenised + "\n")
                        if csv_writer:
                            row = {
                                "line_number": idx,
                                "original_sentence": utt,
                                "tokenised_sentence": tokenised,
                            }
                            row.update(feats)
                            csv_writer.writerow(row)
            else:
                with in_path.open("r", encoding="utf-8") as fin:
                    for idx, raw in enumerate(fin, start=1):
                        original = raw.rstrip("\n")
                        tokenised, feats = process_line(
                            original, nlp=nlp, include_punct=args.include_punct, lowercase=lowercase
                        )
                        fout.write(tokenised + "\n")
                        if csv_writer:
                            row = {"line_number": idx, "original_sentence": original, "tokenised_sentence": tokenised}
                            row.update(feats)
                            csv_writer.writerow(row)
    finally:
        if bf_path and csvfile:
            csvfile.close()

    logging.info("Wrote tokenised text: %s", out_path)
    if bf_path:
        logging.info("Wrote Brown features: %s", bf_path)

if __name__ == "__main__":
    main()
