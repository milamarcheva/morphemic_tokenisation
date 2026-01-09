#!/usr/bin/env python3
"""
Aggregate child-available (non-CHI) utterances from CHAT files into a CSV.

Columns:
  corpus      -> corpus name from @ID
  path        -> path to the .cha file relative to the supplied root
  child_age   -> age of the target child (from CHI/Target_Child @ID)
  speaker_id  -> speaker code (e.g., MOT)
  speaker_role-> speaker role (e.g., Mother)
  utt         -> original utterance text
  gra         -> %gra tier (if present)
  mor         -> %mor tier (if present)

Usage:
    python aggregate_chat_cds.py --root data/CHILDES_Eng-NA --output outputs/cds.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Silence LibreSSL warning emitted by urllib3 (triggered via pylangacq).
warnings.filterwarnings("ignore", message=".*OpenSSL.*", module="urllib3")
try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:
    NotOpenSSLWarning = None
else:
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import pandas as pd

try:
    import pylangacq
except ImportError as e:
    raise SystemExit("pylangacq is required. Install with `pip install pylangacq`.") from e


CHAT_TIMECODE_RE = re.compile(r"\u0015\d+_\d+\u0015")
PAREN_PUNCT_RE = re.compile(r"\(\s*[.?!,-]+\s*\)")
ANGLE_BRACKET_RE = re.compile(r"<[^>]*>")
SQUARE_BRACKET_RE = re.compile(r"\[[^\]]*\]")
BRACKET_REPAIRS = (
    ("[//]", " "),
    ("[/]", " "),
)
# Default: keep ASCII letters plus basic punctuation; strip other symbols.
NON_ALPHA_KEEP_PUNCT_ASCII = re.compile(r"[^A-Za-z\s.,;:!?\"'-]")
# Bulgarian flag will switch to this to keep any-letter characters (Unicode word chars)
NON_ALPHA_KEEP_PUNCT_UNI = re.compile(r"[^\w\s.,;:!?\"'-]", flags=re.UNICODE)

ALLOW_UNICODE_ALPHA = False

VARIANT_TO_STANDARD = {

    # --------------------
    # Cliticizations
    # --------------------
    "coulda": "could have",
    "mighta": "might have",
    "musta": "must have",
    "shoulda": "should have",
    "woulda": "would have",

    "gotta": "got to",
    "hadta": "had to",
    "hafta": "have to",
    "hasta": "has to",
    "oughta": "ought to",


    "wanna": "want to",
    "needa": "need to",
    "needta": "need to",
    "gonna": "going to",
    "sposta": "supposed to",
    "useta": "used to",

    "'cos": "because", 
    "cos": "because",
    "'em": "them",
    "em": "them",

    # --------------------
    # Assimilations
    # --------------------
    "dunno": "don't know",
    "dyou": "do you",
    "gimme": "give me",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
    "lotsa": "lots of",
    "wassup": "what's up",
    "whaddya": "what did you",
    "whyntcha": "why didn't you",
    "gotchu": "got you", 
    "gotcha": "got you",

    # --------------------
    # Dialectal Variants
    # --------------------
    "caint": "can't",
    "da": "the",
    "dan": "than",
    "dat": "that",
    "de": "the",
    "dese": "these",
    "deir": "their",
    "deirselves": "themselves",
    "dem": "them",
    "demselves": "themselves",
    "den": "then",
    "dere": "there",
    "dey": "they",
    "dis": "this",
    "dose": "those",
    "fer": "for",
    "git": "get",
    "gon": "going",
    "hisself": "himself",
    "hows about": "how about",
    "nutin": "nothing",
    "sumpin": "something",
    "tagether": "together",
    "tamorrow": "tomorrow",
    "ta": "to",
    "weunz": "we",
    "whad": "what",
    "wif": "with",
    "ya": "you",
    "yall": "you all",
    "yer": "you",
    "yinz": "you all",
    "younz": "your",
    "youse": "you all",
    "ze": "the",
    "zis": "this",
    "zat": "that",


    # --------------------
    # Other
    # --------------------
    "lets": "let's",
    "mummy": "mommy",
}

def _get_speaker_code(utt) -> str:
    """
    Try to recover speaker/participant code across pylangacq versions and data shapes.
    """
    speaker = getattr(utt, "speaker", None) or getattr(utt, "participant", None)
    if not speaker and isinstance(utt, dict):
        speaker = utt.get("speaker") or utt.get("participant")
    return speaker or ""


# def _maybe_clean_with_pylangacq(text: str) -> str:
    """
    Use pylangacq's CHAT cleaner when available; fall back silently on failure.
    """
    clean_fn = getattr(pylangacq, "clean_utterance", None)
    if callable(clean_fn):
        try:
            return clean_fn(text)
        except Exception:
            pass
    try:
        from pylangacq.chat import clean_utterance as chat_clean_utterance
    except Exception:
        return text
    try:
        return chat_clean_utterance(text)
    except Exception:
        return text


def clean_utterance_text(text: str) -> str:
    """
    Strip CHAT markup characters while keeping basic punctuation needed at the morphemic level.
    """
    if not text:
        return ""

    cleaned = text
    cleaned = CHAT_TIMECODE_RE.sub(" ", cleaned)  # remove \x15 time codes early
    cleaned = PAREN_PUNCT_RE.sub(" ", cleaned)  # drop parenthesized pauses like (.) before alignment
    cleaned = re.sub(r"\b0\S+\b", " ", cleaned)  # drop omitted tokens (0prefixed) before alignment
    cleaned = re.sub(r"&\S*", " ", cleaned)  # drop &-marked words (e.g., &~k's)

    # Strip CHAT address markers and plus-prefixed fragments
    cleaned = re.sub(r"@[\w-]+", "", cleaned)   #remove @ and everyhting that follows until the end of the sentence
    cleaned = cleaned.replace("+", "") # remove + so that baby+sitter --> babysitter
    # cleaned = re.sub(r"\+\S*", " ", cleaned)  # remove +, +..., +" etc.

    cleaned = ANGLE_BRACKET_RE.sub(" ", cleaned)  # remove <...> markup
    for bad, repl in BRACKET_REPAIRS:
        cleaned = cleaned.replace(bad, repl)
    cleaned = SQUARE_BRACKET_RE.sub(" ", cleaned)  # remove [/] etc.

    # # Run pylangacq cleaning after we have stripped pauses/0-markers to avoid misalignment
    # cleaned = _maybe_clean_with_pylangacq(cleaned)
    # # Re-drop any 0-prefixed tokens that pylangacq may have preserved
    # cleaned = re.sub(r"\b0\S+\b", " ", cleaned)
    # # Remove CHAT fillers like &-um -> um (strip the markup prefix)
    # cleaned = re.sub(r"&-?", "", cleaned)

    # Remove parentheses but keep enclosed letters glued (d(o) -> do)
    cleaned = cleaned.replace("(", "").replace(")", "")

    # Remove colons inside words or immediately before punctuation
    cleaned = re.sub(r"(?<=\w):(?=\w)", "", cleaned)
    cleaned = re.sub(r":(?=[?.!,;])", "", cleaned)

    # Strip digits and other non alphabetic characters while keeping punctuation
    cleaned = re.sub(r"\d+", " ", cleaned)
    regex = NON_ALPHA_KEEP_PUNCT_UNI if ALLOW_UNICODE_ALPHA else NON_ALPHA_KEEP_PUNCT_ASCII
    cleaned = regex.sub(" ", cleaned)

    # Collapse multiple spaces but keep single spaces even before punctuation.
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def replace_variants(text: str) -> str:
    """
    Replace any token that exactly matches a key in VARIANT_TO_STANDARD
    with its mapped value.
    """
    if not text:
        return text
    tokens = text.split()
    replaced = [VARIANT_TO_STANDARD.get(tok, tok) for tok in tokens]
    return " ".join(replaced)


def tokens_from_utterance(utt) -> List:
    if hasattr(utt, "tokens"):
        try:
            toks = utt.tokens  # type: ignore[attr-defined]
            if toks:
                return list(toks)
        except Exception:
            pass
    if hasattr(utt, "tokens"):
        try:
            return list(utt.tokens())  # type: ignore[call-arg]
        except Exception:
            pass
    if isinstance(utt, dict):
        toks = utt.get("tokens")
        if toks:
            return list(toks)
    return []


def build_token_annotations(
    utt_text: str, mor: str, gra: str, tokens: Optional[List] = None
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    token_ann: List[Tuple[str, Optional[str], Optional[str]]] = []
    toks = tokens or []
    if toks:
        for t in toks:
            word = getattr(t, "word", None) or getattr(t, "tok", None)
            if isinstance(word, str) and word.startswith("0"):
                continue  # drop omitted tokens marked with leading 0
            mor_tok = getattr(t, "mor", None)
            gra_tok = getattr(t, "gra", None)
            if word is None and mor_tok is None and gra_tok is None:
                continue
            token_ann.append((word or "", mor_tok or None, gra_tok or None))
        if token_ann:
            return token_ann

    # Fallback alignment using whitespace tokenization of utt/mor/gra
    words = (utt_text or "").split()
    mor_parts = mor.split() if mor else []
    gra_parts = gra.split() if gra else []
    maxlen = max(len(words), len(mor_parts), len(gra_parts))
    for i in range(maxlen):
        w = words[i] if i < len(words) else ""
        m = mor_parts[i] if i < len(mor_parts) else None
        g = gra_parts[i] if i < len(gra_parts) else None
        token_ann.append((w, m or None, g or None))
    return token_ann


def parse_id_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse an @ID line into (corpus, code, age, role).
    Example: "@ID:\teng|Brent|MOT||female|||Mother|||"
    """
    after = line.split(":", 1)[1].strip()
    parts = after.split("|")
    corpus = parts[1] if len(parts) > 1 else None
    code = parts[2] if len(parts) > 2 else None
    age = parts[3] if len(parts) > 3 else None
    role = parts[7] if len(parts) > 7 else None
    return corpus, code, age, role


def extract_participants(reader) -> Tuple[Dict[str, Dict[str, Optional[str]]], Optional[str]]:
    participants: Dict[str, Dict[str, Optional[str]]] = {}
    corpus_name: Optional[str] = None
    child_age: Optional[str] = None

    def add_from_pid(pid: str) -> None:
        nonlocal corpus_name, child_age
        line = f"@ID:\t{pid}"
        corpus, code, age, role = parse_id_line(line)
        if corpus and not corpus_name:
            corpus_name = corpus
        if code:
            participants[code] = {"role": role, "age": age}
        if role and "Target_Child" in (role or "") and age:
            child_age = age
        if code == "CHI" and age:
            child_age = age

    # pylangacq API compatibility: ids(), headers()['IDs'], or participants()
    if hasattr(reader, "ids"):
        for pid in reader.ids():
            add_from_pid(pid)
    else:
        headers = []
        try:
            headers = reader.headers()
        except Exception:
            headers = []
        if headers:
            for h in headers:
                if corpus_name is None and isinstance(h, dict):
                    corpus_name = h.get("Corpus") or corpus_name
                for pid in h.get("IDs", []):
                    add_from_pid(pid)
        else:
            try:
                pmap = reader.participants()
            except Exception:
                pmap = {}
            if isinstance(pmap, dict):
                for code, meta in pmap.items():
                    role = (meta or {}).get("role")
                    age = (meta or {}).get("age")
                    participants[code] = {"role": role, "age": age}
                    if role and "Target_Child" in (role or "") and age:
                        child_age = age
                    if code == "CHI" and age:
                        child_age = age

    return participants, child_age or None if child_age else None, corpus_name


def rows_from_reader(chat_path: Path, root: Path, reader, debug: bool = False) -> List[Dict[str, str]]:
    participants, child_age, corpus_name = extract_participants(reader)
    rows: List[Dict[str, str]] = []
    skipped_child = 0
    total_utts = 0
    missing_speaker = 0
    missing_samples: List[str] = []

    for utt in reader.utterances():
        total_utts += 1
        speaker = _get_speaker_code(utt)
        if not speaker:
            missing_speaker += 1
            if debug and len(missing_samples) < 3:
                if isinstance(utt, dict):
                    missing_samples.append(f"dict keys={list(utt.keys())}")
                else:
                    missing_samples.append(f"type={type(utt)} attrs={dir(utt)[:5]}")
            continue
        info = participants.get(speaker, {})
        role = info.get("role")
        if speaker == "CHI" or role == "Target_Child":
            skipped_child += 1
            continue

        tiers = getattr(utt, "tiers", {}) or {}
        utt_text = tiers.get("main", "") or tiers.get("utterance", "") or ""
        gra = tiers.get("gra", "") or ""
        mor = tiers.get("mor", "") or ""
        tokens = tokens_from_utterance(utt)

        # If tiers missing, try tokens
        if not utt_text:
            words = [
                getattr(t, "word", None) or getattr(t, "tok", None)
                for t in tokens
                if getattr(t, "word", None) or getattr(t, "tok", None)
            ]
            utt_text = " ".join(words)
        if not mor:
            mors = [getattr(t, "mor", None) for t in tokens if getattr(t, "mor", None)]
            mor = " ".join(mors)
        if not gra:
            gras = [getattr(t, "gra", None) for t in tokens if getattr(t, "gra", None)]
            gra = " ".join(gras)
        cleaned_utt = replace_variants(clean_utterance_text(utt_text))
        token_ann = build_token_annotations(cleaned_utt, mor, gra, tokens)

        rows.append(
            {
                "corpus": corpus_name or "",
                "path": str(chat_path.relative_to(root)),
                "child_age": child_age or "",
                "speaker_id": speaker,
                "speaker_role": role or "",
                "utt": cleaned_utt,
                "gra": gra,
                "mor": mor,
                "tok_annotations": json.dumps(token_ann, ensure_ascii=False),
            }
        )
    if debug:
        print(
            f"[DEBUG] rows_from_reader {chat_path}: total_utts={total_utts}, missing_speaker={missing_speaker}, "
            f"skipped_child={skipped_child}, added={len(rows)}, missing_examples={missing_samples}"
        )
    return rows


def fallback_parse(chat_path: Path, root: Path) -> List[Dict[str, str]]:
    participants: Dict[str, Dict[str, Optional[str]]] = {}
    corpus_name: Optional[str] = None
    child_age: Optional[str] = None
    rows: List[Dict[str, str]] = []

    current_speaker = None
    current_utt = None
    current_mor = None
    current_gra = None

    lines = chat_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if line.startswith("@ID:"):
            corpus, code, age, role = parse_id_line(line)
            if corpus and not corpus_name:
                corpus_name = corpus
            if code:
                participants[code] = {"role": role, "age": age}
            if role and "Target_Child" in (role or "") and age:
                child_age = age
            if code == "CHI" and age:
                child_age = age
        elif line.startswith("*"):
            if current_speaker and current_utt is not None:
                info = participants.get(current_speaker, {})
                role = info.get("role")
                if current_speaker != "CHI" and role != "Target_Child":
                    cleaned_current = replace_variants(clean_utterance_text(current_utt))
                    rows.append(
                        {
                            "corpus": corpus_name or "",
                            "path": str(chat_path.relative_to(root)),
                            "child_age": child_age or "",
                            "speaker_id": current_speaker,
                            "speaker_role": role or "",
                            "utt": cleaned_current,
                            "gra": current_gra or "",
                            "mor": current_mor or "",
                            "tok_annotations": json.dumps(
                                build_token_annotations(cleaned_current or "", current_mor or "", current_gra or ""),
                                ensure_ascii=False,
                            ),
                        }
                    )
            current_mor = None
            current_gra = None
            current_speaker = line[1:].split(":", 1)[0].strip()
            if "\t" in line:
                current_utt = line.split("\t", 1)[1].strip()
            else:
                current_utt = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        elif line.startswith("%mor:"):
            current_mor = line.split(":", 1)[1].strip()
        elif line.startswith("%gra:"):
            current_gra = line.split(":", 1)[1].strip()

    if current_speaker and current_utt is not None:
        info = participants.get(current_speaker, {})
        role = info.get("role")
        if current_speaker != "CHI" and role != "Target_Child":
            cleaned_current = replace_variants(clean_utterance_text(current_utt))
            rows.append(
                {
                    "corpus": corpus_name or "",
                    "path": str(chat_path.relative_to(root)),
                    "child_age": child_age or "",
                    "speaker_id": current_speaker,
                    "speaker_role": role or "",
                    "utt": cleaned_current,
                    "gra": current_gra or "",
                    "mor": current_mor or "",
                    "tok_annotations": json.dumps(
                        build_token_annotations(cleaned_current or "", current_mor or "", current_gra or ""),
                        ensure_ascii=False,
                    ),
                }
            )
    return rows


def parse_chat(
    chat_path: Path, root: Path, quiet: bool = False, debug: bool = False, force_fallback: bool = False
) -> List[Dict[str, str]]:
    last_exc: Optional[Exception] = None
    reader = None
    if force_fallback:
        rows = fallback_parse(chat_path, root)
        if debug:
            print(f"[DEBUG] force_fallback {chat_path}: added={len(rows)}")
        return rows
    # Try standard pylangacq read, then relax options if available.
    try:
        reader = pylangacq.read_chat(str(chat_path))
    except Exception as exc:
        last_exc = exc
        for kwargs in [{"strict": False}, {"check": False}, {"clean": False}]:
            try:
                reader = pylangacq.read_chat(str(chat_path), **kwargs)
                last_exc = None
                break
            except TypeError:
                continue
            except Exception as exc2:
                last_exc = exc2
                continue

    if reader is None:
        if not quiet:
            print(f"Warning: pylangacq failed on {chat_path} ({last_exc}); using fallback parser.")
        rows = fallback_parse(chat_path, root)
        if debug:
            print(f"[DEBUG] fallback_parse {chat_path}: added={len(rows)}")
        return rows

    try:
        rows = rows_from_reader(chat_path, root, reader, debug=debug)
        if debug:
            print(f"[DEBUG] pylangacq rows {chat_path}: added={len(rows)}")
        return rows
    except Exception as exc:
        if not quiet:
            print(f"Warning: pylangacq failed on {chat_path} ({exc}); using fallback parser.")
        rows = fallback_parse(chat_path, root)
        if debug:
            print(f"[DEBUG] fallback_parse {chat_path}: added={len(rows)}")
        return rows


def aggregate(
    root: Path, quiet: bool = False, verbose: bool = False, debug: bool = False, force_fallback: bool = False
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for chat_path in sorted(root.rglob("*.cha")):
        new_rows = parse_chat(chat_path, root, quiet=quiet, debug=debug, force_fallback=force_fallback)
        rows.extend(new_rows)
        if verbose:
            print(f"Processed {chat_path} -> {len(new_rows)} rows")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate CHAT child-available utterances into CSV.")
    p.add_argument("--root", required=True, help="Directory containing .cha files (e.g., data/CHILDES_Eng-NA).")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--quiet", action="store_true", help="Suppress pylangacq warnings.")
    p.add_argument("--verbose", action="store_true", help="Print each .cha file as it is processed.")
    p.add_argument("--debug", action="store_true", help="Print debug stats on parsing/row counts.")
    p.add_argument("--force-fallback", action="store_true", help="Bypass pylangacq and use the fallback parser.")
    p.add_argument("--bg", action="store_true", help="Allow Unicode letters (e.g., Bulgarian/Cyrillic) during cleaning.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global ALLOW_UNICODE_ALPHA
    ALLOW_UNICODE_ALPHA = bool(args.bg)
    root = Path(args.root)
    rows = aggregate(
        root,
        quiet=args.quiet,
        verbose=args.verbose,
        debug=args.debug,
        force_fallback=args.force_fallback,
    )
    df = pd.DataFrame(
        rows,
        columns=["corpus", "path", "child_age", "speaker_id", "speaker_role", "utt", "gra", "mor", "tok_annotations"],
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
