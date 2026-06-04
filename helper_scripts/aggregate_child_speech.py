from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

# --------------------
# Regex patterns
# --------------------

CHI_ID_RE = re.compile(r"^@ID:\s*.*\|CHI\|([^|]*)\|")
CHI_UTT_RE = re.compile(r"^\*CHI:\s*(.*)$")

# Word-only tokenizer (punctuation excluded)
WORD_RE = re.compile(r"\b\w+\b")

CHAT_TIMECODE_RE = re.compile(r"\u0015\d+_\d+\u0015")
PAREN_PUNCT_RE = re.compile(r"\(\s*[.?!,-]+\s*\)")
ANGLE_BRACKET_RE = re.compile(r"<[^>]*>")
SQUARE_BRACKET_RE = re.compile(r"\[[^\]]*\]")
BRACKET_REPAIRS = (
    ("[//]", " "),
    ("[/]", " "),
)
NON_ALPHA_KEEP_PUNCT_ASCII = re.compile(r"[^A-Za-z\s.,;:!?\"'-]")
NON_ALPHA_KEEP_PUNCT_UNI = re.compile(r"[^\w\s.,;:!?\"'-]", flags=re.UNICODE)
PUNCT_ONLY_RE = re.compile(r"[.,;:!?\"-]")

ALLOW_UNICODE_ALPHA = False

VARIANT_TO_STANDARD = {
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
    "lets": "let's",
    "mummy": "mommy",
}


# --------------------
# Config
# --------------------
ROOT_DIR = Path(
    "/Users/milamarcheva/Desktop/morphemic_tokenisation/data/"
    "CHILDES_ENG/CHILDES_Eng-NA"
)
ALLOWED_PREFIXES = {
    "Brown/Adam",
    "Brown/Eve",
    "Brown/Sarah",
    "HSLLD/HV1-ER",
    "HSLLD/HV1-MT",
    "Soderstrom",
    "Suppes",
    "Valian",
}


# --------------------
# Helpers
# --------------------

def load_vocab(vocab_path: str | Path) -> set[str]:
    vocab_path = Path(vocab_path)
    with vocab_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def extract_age_from_lines(lines: list[str]) -> str | None:
    for line in lines:
        m = CHI_ID_RE.match(line)
        if m:
            age = m.group(1).strip()
            return age if age else None
    return None


def extract_chi_utts_from_lines(lines: list[str]) -> list[str]:
    utts = []
    for line in lines:
        m = CHI_UTT_RE.match(line)
        if m:
            utt = m.group(1).strip()
            if utt:
                utts.append(utt)
    return utts


def clean_utterance_text(text: str) -> str:
    """
    Strip CHAT markup while keeping basic punctuation for downstream cleaning.
    """
    if not text:
        return ""
    cleaned = text
    cleaned = CHAT_TIMECODE_RE.sub(" ", cleaned)
    cleaned = PAREN_PUNCT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\b0\S+\b", " ", cleaned)
    cleaned = re.sub(r"&\S*", " ", cleaned)
    cleaned = re.sub(r"@[\w-]+", "", cleaned)
    cleaned = cleaned.replace("+", "")
    cleaned = ANGLE_BRACKET_RE.sub(" ", cleaned)
    for bad, repl in BRACKET_REPAIRS:
        cleaned = cleaned.replace(bad, repl)
    cleaned = SQUARE_BRACKET_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = re.sub(r"(?<=\w):(?=\w)", "", cleaned)
    cleaned = re.sub(r":(?=[?.!,;])", "", cleaned)
    cleaned = re.sub(r"\d+", " ", cleaned)
    regex = NON_ALPHA_KEEP_PUNCT_UNI if ALLOW_UNICODE_ALPHA else NON_ALPHA_KEEP_PUNCT_ASCII
    cleaned = regex.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def replace_variants(text: str) -> str:
    if not text:
        return text
    tokens = text.split()
    replaced = [VARIANT_TO_STANDARD.get(tok, tok) for tok in tokens]
    return " ".join(replaced)


def clean_child_utterance(text: str) -> str:
    cleaned = replace_variants(clean_utterance_text(text))
    cleaned = PUNCT_ONLY_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def utterance_in_vocab(utt: str, vocab: set[str]) -> bool:
    """
    Keep utterance iff ALL WORD tokens (no punctuation) are in vocab.
    """
    tokens = WORD_RE.findall(utt)
    if not tokens:
        return False
    return all(tok in vocab for tok in tokens)


def is_allowed_path(path: Path) -> bool:
    """
    Check whether path starts with one of the allowed corpus headers.
    """
    path_str = path.as_posix()
    return any(path_str.startswith(prefix) for prefix in ALLOWED_PREFIXES)


# --------------------
# Main extraction
# --------------------

def extract_child_utterances(
    root_dir: str | Path,
) -> pd.DataFrame:
    root_dir = Path(root_dir)

    rows = []

    for cha_path in sorted(root_dir.rglob("*.cha")):
        # restrict to allowed corpora
        rel_path = cha_path.relative_to(root_dir)
        if not is_allowed_path(rel_path):
            continue

        try:
            text = cha_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = cha_path.read_text(encoding="utf-8", errors="replace")

        lines = text.splitlines()
        age = extract_age_from_lines(lines)

        for utt in extract_chi_utts_from_lines(lines):
            cleaned = clean_child_utterance(utt)
            rows.append(
                {
                    "source": str(rel_path),
                    "age": age,
                    "utt": utt,
                    "utt_cleaned": cleaned,
                }
            )

    return pd.DataFrame(rows, columns=["source", "age", "utt", "utt_cleaned"])


# --------------------
# Run
# --------------------

if __name__ == "__main__":
    # Point at the CHILDES_Eng-NA root so rel_paths start with Brown/, HSLLD/, etc.
    root = ROOT_DIR
    df = extract_child_utterances(root)

    out_csv = Path("data/dfs/child_utterances_vocab_filtered.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Wrote {len(df):,} rows to {out_csv}")
    print(df.head())
