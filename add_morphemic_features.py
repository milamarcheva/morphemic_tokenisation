import argparse
from collections import Counter
import io
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import spacy

DEFAULT_ENGLISH_MORPHSCORE_V1_URL = (
    "https://github.com/catherinearnett/morphscore/raw/v1/data/english_morph_data.csv"
)
DEFAULT_ENGLISH_MORPHSCORE_V2_URL = (
    "https://huggingface.co/datasets/catherinearnett/morphscore/"
    "resolve/main/english_data.csv?download=true"
)
MORPHSCORE_V2_URLS = {
    "eng_latn": DEFAULT_ENGLISH_MORPHSCORE_V2_URL,
}
ENGLISH_DETAIL_LABEL_ORDER = (
    "reg_past_choice",
    "progressive_choice",
    "s_suffix_choice",
    "contraction_choice",
    "reg_past_boundary",
    "progressive_boundary",
    "s_suffix_boundary",
    "contraction_boundary",
    "reg_past_single_token",
    "progressive_single_token",
    "s_suffix_single_token",
    "contraction_single_token",
    "skipped_single_token",
    "boundary_mismatch",
    "surface_mismatch",
    "single_token",
)
ENGLISH_CHOICE_LABELS = {
    "reg_past_choice",
    "progressive_choice",
    "s_suffix_choice",
    "contraction_choice",
}

def load_spacy(model: str):
    try:
        return spacy.load(model)
    except OSError as e:
        raise SystemExit(
            f"Could not load spaCy model '{model}'. Install with: python -m spacy download {model}"
        ) from e


def get_lemma(doc, token):
    for t in doc:
        if t.lower_ == token.lower():
            return t.lemma_

# --- Feature predicates (mirror tests/test_partIII_notebook_asserts.py) ---

brown_morphs = [ "progressive", "prep_on", "prep_in", "reg_plural",  "irreg_past",
    "poss", "uncontr_copula", "articles", "reg_past", "reg_3rd", "irreg_3rd",
    "uncontr_aux", "contr_copula", "contr_aux",
]
def is_progressive(token) -> bool:
    return token.morph.get("Aspect") == ["Prog"] and token.lower_[-3:] == "ing"


def is_in(token) -> bool:
    return token.tag_ == "IN" and token.lower_ == "in"


def is_on(token) -> bool:
    return token.tag_ == "IN" and token.lower_ == "on"


ed_exceptions = {'aran_reed', 'bed', 'birdfeed', 'birdseed', 'bled', 'bleed',
                 'bunkbed', 'captain_mildred', 'ed', 'embed','exceed','feed', 'fireman_fred', 
                 'flatbed',
                 'fred', 'get_into_bed', 'go_to_bed', 'heed', 'hundred', 'indeed', 'jed',
                 'knockkneed', 'led', 'mildred', 'mister_reed', 'ned', 'need',
                 'old_macreed', 'playbed', 'proceed', 'red', 'reed', 'reseed',
                 'robber_red', 'shed', 'sled', 'sofabed', 'ted', 'thoroughbred',
                 'wilfred'
                 }

ing_exceptions = {'aboing', 'anything',  'bing', 'blingie^bling^bling', 'boing',  #building, clothing, stocking? have currently not included them as exceptions
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
s_exceptions = {'binoculars', 'headphones', 'sunglasses', 'glasses',
                 'scissors', 'tweezers', 'jeans', 'pyjamas', 'tights',
                 'knickers', 'shorts', 'trousers', 'pants', 'belongings',
                 'outskirts', 'clothes', 'premises', 'congratulations',
                 'savings', 'earnings', 'stairs', 'goods', 'surroundings',
                 'thanks', 'yours', 
                 'actress', 'illness', 'weakness', 'sinus', 'success', 
                 'dress', 'boss', 'bypass', 'encompass', 'ass',"'s",
                 } #yours added because for some reason it is tagged as NNS by spacy repeatedly
   
irregular_3rd_person_verbs = {'be', 'have', 'go', 'do'}
IRREGULAR_3RD_PERSON_VERBS = {"have", "go", "do", "be"}

CONTRACTIONS = ("n't", "'s", "'re", "'ll", "'d", "'m", "'ve", )

plural_tags = {"NNS", "NNPS"}
verb_3rd_tags = {"VBZ", "AUX", "VBD", }


def is_reg_plural(token) -> bool:
    return token.tag_ == "NNS" and token.lower_.endswith("s") and token.lower_ not in PLURAL_ONLY


def is_irregular_past(token) -> bool:
    return token.morph.get("Tense") == ["Past"] and not token.lower_.endswith("ed") and token.lemma_ != "be"


def is_poss(token) -> bool:
    return token.tag_ == "POS"


def is_article(token) -> bool:
    return token.tag_ == "DT" and token.lower_ in {"a", "the"}


def is_regular_past(token) -> bool:
    return token.morph.get("Tense") == ["Past"] and token.lower_.endswith("ed")





def is_regular_3rd_present(token) -> bool:
    return (
        token.morph.get("Tense") == ["Pres"]
        and token.morph.get("Person") == ["3"]
        and token.morph.get("Number") == ["Sing"]
        and token.lemma_ not in IRREGULAR_3RD_PERSON_VERBS | {"be"}
        and token.lower_.endswith("s")
    )


def is_irregular_3rd_present(token) -> bool:
    return (
        token.morph.get("Tense") == ["Pres"]
        and token.morph.get("Person") == ["3"]
        and token.morph.get("Number") == ["Sing"]
        and token.lemma_ in IRREGULAR_3RD_PERSON_VERBS
    )


def is_uncontr_aux(token) -> bool:
    return token.tag_.startswith("V") and token.dep_ == "aux" and token.lemma_ == "be" and not token.text.startswith("'")


def is_uncontr_copula(token) -> bool:
    return token.tag_.startswith("V") and token.dep_ == "ROOT" and token.lemma_ == "be" and not token.text.startswith("'")


def is_contr_copula(token) -> bool:
    return token.tag_.startswith("V") and token.dep_ == "ROOT" and token.lemma_ == "be" and token.text.startswith("'")


def is_contr_aux(token) -> bool:
    return token.tag_.startswith("V") and token.lemma_ == "be" and token.text.startswith("'") and token.dep_ in {"aux", "case"}


# --- Helpers ---
def clean_word(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.lower())


def token_contains_contraction(token: str) -> Optional[Tuple[str, str]]:
    """
    If a lemma ends with a known contraction, split into (base, contraction).
    Returns None when no contraction match is found.
    """
    for contr in sorted(CONTRACTIONS, key=len, reverse=True):
        if token.endswith(contr) and len(token) > len(contr):
            return contr
    return ""


def parse_ann_feats(mor: str) -> Dict[str, str]:
    feats: Dict[str, str] = {}
    parts = mor.split("|", 1)
    if len(parts) != 2:
        return feats
    slots = parts[1].split("-")
    for slot in slots:
        s = slot.lower()
        if s == "pres":
            feats["Tense"] = "Pres"
        if s == "past":
            feats["Tense"] = "Past"
        if s == "prog":
            feats["Aspect"] = "Prog"
        if s in {"s1", "s2", "s3"}:
            feats["Person"] = s[1]
            feats["Number"] = "Sing"
        if s == "plur":
            feats["Number"] = "Plur"
    return feats


def build_ann_token(tup: Sequence) -> Optional[Dict[str, object]]:
    if not tup or len(tup) < 2:
        return None
    word = clean_word(tup[0] or "")
    mor = tup[1] or ""
    if "|" not in mor:
        return None
    lemma = clean_word(mor.split("|", 1)[1].split("-", 1)[0])
    feats = parse_ann_feats(mor)
    print(word, lemma, feats)
    return {"word": word, "lemma": lemma, "feats": feats}




def tokens_aligned(tok_ann: Sequence[Sequence], utt: str = "") -> bool:
    if not tok_ann:
        return False
    EXCEPTION_EQUIV = {
        "me": "i",
        "i": "me",
        "us": "we",
        "em": "they", 
        "dat":"that",
        "better": "good",
        "best":"good", 
        "went": "go", 
        "ate": "eat"
    }
    for i, tup in enumerate(tok_ann):
        if not tup or len(tup) < 2:
            #print(f"[tokens_aligned] idx={i} fail: empty/short tuple")
            return False
        word, mor = tup[0], tup[1]
        # Skip placeholder/empty tokens (often punctuation placeholders)
        if (word is None or str(word).strip() == "") and (mor is None or str(mor).strip() == ""):
            continue
        # Skip punctuation tokens for alignment
        if re.fullmatch(r"\W+", str(word)):
            continue
        if not word or not mor or "|" not in mor:
            # print(f"[tokens_aligned] idx={i} fail: missing word/mor or no '|' -> word={word}, mor={mor} utt={utt} tok_ann={tok_ann}")
            continue

        # Only consider the first sub-morpheme (before '~') for alignment
        mor_first = mor.split("~", 1)[0]
        mor_word = mor_first.split("|", 1)[1].split("-")[0].lower() if "|" in mor_first else ""
        w_clean = clean_word(word)
        m_clean = clean_word(mor_word)

        # Looser check: require first letter to match after cleaning
        if not w_clean or not m_clean:
            # print(f"[tokens_aligned] idx={i} fail: empty cleaned forms w='{w_clean}' m='{m_clean}' utt='{utt}' tok_ann={tok_ann}")
            return False
        if w_clean[0] == m_clean[0]:
            continue
        # Allow common copula/aux forms where surface differs from lemma (is/are/am -> be)
        if w_clean in {"is", "are", "am", "was", "were", "s", "isn't", "wasn't",   "weren't", "aren't", "r"} and m_clean.startswith("be"):
            continue
        # Allow exception mappings
        if EXCEPTION_EQUIV.get(w_clean) == m_clean or EXCEPTION_EQUIV.get(m_clean) == w_clean:
            continue
        # print(f"[tokens_aligned] idx={i} fail: first letter mismatch w='{w_clean}' m='{m_clean}'") # utt='{utt}' tok_ann={tok_ann}")
        return False

    return True


def morph_tokens_from_annotations(tok_ann: Sequence[Sequence]) -> List[str]:
    tokens: List[str] = []
    for tup in tok_ann:
        ann = build_ann_token(tup)
        if not ann:
            continue
        surface = ann["word"]
        base = ann["lemma"]
        # Heuristics for splitting common morphemes
        if surface.endswith("ing") and base:
            stem = base[:-3] if base.endswith("ing") else base
            if stem:
                tokens.extend([stem, "ing"])
            else:
                tokens.append("ing")
            continue
        if surface.endswith("ed") and base:
            stem = base[:-2] if base.endswith("ed") else base
            if stem:
                tokens.extend([stem, "ed"])
            else:
                tokens.append("ed")
            continue
        if surface.endswith("s") and base and base != surface:
            stem = base[:-1] if base.endswith("s") else base
            if stem:
                tokens.extend([stem, "s"])
            else:
                tokens.append("s")
            continue

        if base:
            tokens.append(base)
    return tokens


def split_surface_for_morph(word: str) -> List[str]:
    """Split a surface form to align with morph parts (handles n't and apostrophes)."""
    w = (word or "").lower()
    if not w:
        return []
    if w.endswith("n't") and len(w) > 3:
        return [w[:-3], "n't"]
    if "'" in w:
        base, suffix = w.split("'", 1)
        return [base, f"'{suffix}"]
    return [w]


def spacy_tokenise(doc, tok_ann=None, pytest=False):
  sent = []
  postags_morphtok = [] 

  if tok_ann:
    surfaces = [item[0] for item in tok_ann]
    mors = [item[1] for item in tok_ann]
    surfaces_split = []
    for s_idx, surf in enumerate(surfaces):
      parts = split_surface_for_morph(surf) or [str(surf).lower()]
      for p in parts:
        surfaces_split.append((p, s_idx))
    mors_split = []
    for m_idx, mor in enumerate(mors):
      if not mor:
        continue
      for part in str(mor).split("~"):
        mors_split.append((part, m_idx))
  
  mor_index = 0
  for idx, t in enumerate(doc):

    #progressive ing
    if t.lower_[-3:] == 'ing' and t.lower_ not in ing_exceptions: #token.morph.get('Aspect') == ['Prog'] and
      morpheme = t.lower_[-3:]
      lemma = ""
      if t.lemma_[-3:] == 'ing' and  t.lemma_ not in ing_exceptions:
        lemma = t.lemma_[:-3]
      else:
        lemma = t.lemma_

      sent.append(lemma.lower())
      sent.append(morpheme)

      postags_morphtok.append((t.pos_, t.tag_))
      postags_morphtok.append(("ing", "ing"))

    #regular plurals (-s)
    elif t.tag_ in plural_tags and t.lower_[-1] == 's' and t.tag_[0]!="'" and t.lower_ not in s_exceptions:
      if pytest:
        print(f"{t.tag_} in reg plural s")
      morpheme = t.lower_[-1:]
      lemma = ""
      if t.lemma_[-1:] == 's' and t.lemma_ not in s_exceptions:
        lemma = t.lemma_[:-1]
      else:
        lemma = t.lemma_

      sent.append(lemma.lower())
      sent.append(morpheme)

      postags_morphtok.append((t.pos_, t.tag_))
      postags_morphtok.append(("plu", "plu"))

    #regular past tense
    elif t.lower_[-2:] == 'ed' and  t.lower_ not in ed_exceptions: #token.morph.get('Tense') == ['Past'] and  and token.morph.get('VerbForm')!=['Part']
      morpheme = t.lower_[-2:]
      lemma = ""

      if t.lemma_[-2:] == 'ed' and  t.lemma_ not in ed_exceptions:
        lemma = t.lemma_[:-2]
      else:
        lemma = t.lemma_
      sent.append(lemma.lower())
      sent.append(morpheme)

      postags_morphtok.append((t.pos_, t.tag_))
      postags_morphtok.append(("ed", "ed"))
    
    #3rd person present regular (-s)
    elif t.morph.get('Tense') == ['Pres'] and t.morph.get('Person') == ['3'] and t.morph.get('Number') == ['Sing'] and t.lemma_ not in irregular_3rd_person_verbs and t.lower_[-1] == 's' and t.lower_ not in s_exceptions:
      if pytest:
        print(f"{t.tag_} 3rd person s")
      morpheme = t.lower_[-1:]
      lemma = ""
      #   sent.append(t.lemma_)
      if t.lemma_[-1:] == 's' and  t.lemma_ not in s_exceptions:
        lemma = t.lemma_[:-1]
      else:
        lemma = t.lemma_

      sent.append(lemma.lower())
      sent.append(morpheme)

      postags_morphtok.append((t.pos_, t.tag_))
      postags_morphtok.append(("third", "third"))

    elif not t.is_punct:
      contr = token_contains_contraction(t.lower_)
      if contr=="":
        sent.append(t.lower_)
        postags_morphtok.append((t.pos_, t.tag_))
      else:
        sent.append(t.lower_[: -len(contr)])
        sent.append(contr)
        postags_morphtok.append((t.pos_, t.tag_))
        postags_morphtok.append(("contr", "contr"))


  sent_norm_tok= [t.lower_ for t in doc if not t.is_punct]
  postags_normtok = [(t.pos_, t.tag_) for t in doc if not t.is_punct]
  morlvl_fulltags_normtok = []

  if tok_ann:
    try:
      m_len = len(mors_split)
      for i in range(len(doc)):
        if doc[i].is_punct:
          continue
        if i >= m_len:
          break
        morlvl_fulltags_normtok.append(mors_split[i])
    except Exception as e:
      print("Error aligning morlvl_fulltags_normtok:", e)
      print("i:", i)
      print("doc:", [t.text for t in doc])
      print("mors_split:", mors_split)
      pass

  return sent, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok,


def feature_flags(tokens) -> dict:
    flags = {name: 0 for name in FeatureNames}
    for tok in tokens:
        if flags["progressive"] == 0 and is_progressive(tok):
            flags["progressive"] = 1
        if flags["prep_on"] == 0 and is_on(tok):
            flags["prep_on"] = 1
        if flags["prep_in"] == 0 and hasattr(tok, "lower_") and tok.lower_ == "in":
            flags["prep_in"] = 1
        if flags["reg_plural"] == 0 and is_reg_plural(tok):
            flags["reg_plural"] = 1
        if flags["irreg_past"] == 0 and is_irregular_past(tok):
            flags["irreg_past"] = 1
        if flags["poss"] == 0 and is_poss(tok):
            flags["poss"] = 1
        if flags["uncontr_copula"] == 0 and is_uncontr_copula(tok):
            flags["uncontr_copula"] = 1
        if flags["articles"] == 0 and is_article(tok):
            flags["articles"] = 1
        if flags["reg_past"] == 0 and is_regular_past(tok):
            flags["reg_past"] = 1
        if flags["reg_3rd"] == 0 and is_regular_3rd_present(tok):
            flags["reg_3rd"] = 1
        if flags["irreg_3rd"] == 0 and is_irregular_3rd_present(tok):
            flags["irreg_3rd"] = 1
        if flags["uncontr_aux"] == 0 and is_uncontr_aux(tok):
            flags["uncontr_aux"] = 1
        if flags["contr_copula"] == 0 and is_contr_copula(tok):
            flags["contr_copula"] = 1
        if flags["contr_aux"] == 0 and is_contr_aux(tok):
            flags["contr_aux"] = 1
    return flags


def process_row(row, nlp, text_col: str, tok_ann_col: str):
    utt = row.get(text_col, "")
    if not isinstance(utt, str):
        utt = "" if pd.isna(utt) else str(utt)

    tok_ann_raw = row.get(tok_ann_col)
    tok_ann = None
    if isinstance(tok_ann_raw, str):
        try:
            tok_ann = json.loads(tok_ann_raw)
        except json.JSONDecodeError:
            tok_ann = None
    elif isinstance(tok_ann_raw, Sequence):
        tok_ann = tok_ann_raw

   

    doc = nlp(utt.strip())  # always parse for features

    if tokens_aligned(tok_ann):
        # print("here")
        sent_morphtok, sent_normtok, postags_morphtok, postags_normtok, mor_fulltags_normtok= spacy_tokenise(doc, tok_ann)
    else:
        sent_morphtok, sent_normtok, postags_morphtok, postags_normtok, mor_fulltags_normtok = spacy_tokenise(doc)

    # morph_tokens = morph_tokens_from_annotations(tok_ann) if use_ann else morph_tokens_from_spacy(doc)
    # flags = feature_flags(token_views)
    out = {
        "sent_morphtok": " ".join(sent_morphtok),
        "spacy_normtok": " ".join(sent_normtok),
        "postags_morphtok": postags_morphtok,
        "postags_normtok": postags_normtok,
        "mor_fulltags_normtok": mor_fulltags_normtok,
    }
    # out.update(flags)
    return out

def is_nan_like(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def normalize_morph_text(value: object) -> Optional[str]:
    if is_nan_like(value):
        return None
    return str(value).strip().lower()


def load_morphscore_v2_data(lang_code: str = "eng_latn") -> pd.DataFrame:
    try:
        url = MORPHSCORE_V2_URLS[lang_code]
    except KeyError as exc:
        supported = ", ".join(sorted(MORPHSCORE_V2_URLS))
        raise ValueError(
            f"Local MorphScore v2 evaluation only supports: {supported}"
        ) from exc

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine="python")

    required = {
        "wordform",
        "unique",
        "lemma",
        "stem",
        "preceding_part",
        "following_part",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "MorphScore v2 data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return data


def get_word_v2(row: Dict[str, object]) -> str:
    return normalize_morph_text(row.get("wordform")) or ""


def build_gold_morphemes_v2(row: Dict[str, object]) -> List[str]:
    prefix = normalize_morph_text(row.get("preceding_part"))
    stem = normalize_morph_text(row.get("stem"))
    suffix = normalize_morph_text(row.get("following_part"))

    if stem is None:
        return []

    morphemes: List[str] = []
    if prefix is not None:
        morphemes.append(prefix)
    morphemes.append(stem)
    if suffix is not None:
        morphemes.append(suffix)
    return morphemes


def get_predicted_boundaries(tokens: Sequence[str]) -> List[int]:
    boundaries: List[int] = []
    idx = 0
    for tok in tokens:
        idx += len(tok)
        boundaries.append(idx)
    return boundaries


def morph_eval_v2(
    morphemes: Sequence[str],
    tokens: Sequence[str],
    *,
    exclude_single_tok: bool,
    exclude_single_morpheme: bool,
    single_tok_point: float,
    correct_point: float,
    partial_point: float,
) -> Tuple[float, float]:
    if len(tokens) == 1:
        if exclude_single_tok:
            return (math.nan, math.nan)
        return (single_tok_point, single_tok_point)

    pred_boundaries = get_predicted_boundaries(tokens)

    if len(morphemes) == 2:
        gold_boundary = len(morphemes[0])
        if gold_boundary in pred_boundaries:
            return (correct_point, 1.0 / len(pred_boundaries))
        return (0.0, 0.0)

    if len(morphemes) == 3:
        boundary_1 = len(morphemes[0])
        boundary_2 = len(morphemes[0]) + len(morphemes[1])
        matched = sum(int(boundary in pred_boundaries) for boundary in (boundary_1, boundary_2))

        if matched == 2:
            return (correct_point, 2.0 / len(pred_boundaries))
        if matched == 1:
            return (partial_point, 1.0 / len(pred_boundaries))
        return (0.0, 0.0)

    if len(morphemes) == 1:
        if exclude_single_morpheme:
            return (math.nan, math.nan)
        return (
            (single_tok_point, single_tok_point)
            if list(morphemes) == list(tokens)
            else (0.0, 0.0)
        )

    return (math.nan, math.nan)


def filter_v2_rows(
    rows: Sequence[Dict[str, object]],
    *,
    unique_only: bool,
    stem_eq_lemma: bool,
    exclude_numbers: bool,
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []

    for raw_row in rows:
        row = dict(raw_row)
        word = get_word_v2(row)
        if not word:
            continue

        if unique_only and str(row.get("unique", "")).strip().lower() != "unique":
            continue
        if (
            stem_eq_lemma
            and normalize_morph_text(row.get("stem")) != normalize_morph_text(row.get("lemma"))
        ):
            continue
        if exclude_numbers and re.search(r"\d", word):
            continue

        filtered.append(row)

    return filtered


def mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def std_or_zero(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def f1_or_zero(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def get_float(row: Dict[str, object], key: str, default: float) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def normalize_token_list(raw_tokens) -> List[str]:
    if raw_tokens is None:
        return []
    if isinstance(raw_tokens, str):
        raw_tokens = raw_tokens.split()
    return [
        tok_norm
        for tok in raw_tokens
        for tok_norm in [normalize_morph_text(tok)]
        if tok_norm
    ]


def infer_english_boundary_family(
    gold_suffix: Optional[str],
    pred_suffix: Optional[str] = None,
    *,
    word: str = "",
) -> Optional[str]:
    suffixes = {
        normalize_morph_text(gold_suffix) or "",
        normalize_morph_text(pred_suffix) or "",
    }
    suffixes.discard("")
    if any(suffix == "ing" for suffix in suffixes):
        return "progressive"
    if any(suffix in {"ed", "d", "t"} for suffix in suffixes):
        return "reg_past"
    if any(suffix in {"s", "es", "'s"} for suffix in suffixes):
        return "s_suffix"
    if any(suffix in CONTRACTIONS for suffix in suffixes):
        return "contraction"
    if word.endswith("'s"):
        return "contraction"
    return None


def english_family_label(family: Optional[str], kind: str, fallback: str) -> str:
    if family is None:
        return fallback
    return f"{family}_{kind}"


def classify_english_mismatch_label(
    gold: Sequence[str],
    pred: Sequence[str],
    *,
    result_label: str,
    word: str = "",
) -> str:
    gold_tokens = [normalize_morph_text(part) or "" for part in gold]
    pred_tokens = [normalize_morph_text(part) or "" for part in pred]
    if result_label == "correct":
        return "correct"
    family = infer_english_boundary_family(
        gold_tokens[-1] if gold_tokens else "",
        pred_tokens[-1] if pred_tokens else "",
        word=word,
    )
    if result_label == "skipped_single_token":
        return english_family_label(family, "single_token", "skipped_single_token")
    if "".join(gold_tokens) != "".join(pred_tokens):
        return "surface_mismatch"
    if family is not None:
        if (
            len(gold_tokens) >= 2
            and len(pred_tokens) >= 2
            and gold_tokens[0] == pred_tokens[0]
            and gold_tokens[-1] != pred_tokens[-1]
        ):
            return f"{family}_choice"
        return f"{family}_boundary"
    return "boundary_mismatch"


def classify_v2_detail(row: Dict[str, object], morphemes: Sequence[str], tokens: Sequence[str]) -> str:
    if list(tokens) == list(morphemes):
        return ""
    result_label = "skipped_single_token" if len(tokens) == 1 and len(morphemes) > 1 else "wrong"
    return classify_english_mismatch_label(
        morphemes,
        tokens,
        result_label=result_label,
        word=get_word_v2(row),
    )


def collect_v2_detail_counts(detail_df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    if "morphscore_detail" not in detail_df.columns:
        return counts
    for label in detail_df["morphscore_detail"].fillna("").astype(str):
        if label in ENGLISH_DETAIL_LABEL_ORDER:
            counts[label] += 1
    return counts


def print_v2_detail_summary(detail_df: pd.DataFrame) -> None:
    counts = collect_v2_detail_counts(detail_df)
    print("V2 detailed disagreement labels:")
    for label in ENGLISH_DETAIL_LABEL_ORDER:
        print(f"  {label}: {counts.get(label, 0)}")


def get_morphscore_v2(
    rows: Sequence[Dict[str, object]],
    tokenizer,
    *,
    include_all_rows: bool = False,
    freq_scale: bool,
    exclude_single_tok: bool,
    exclude_single_morpheme: bool,
    single_tok_point: float,
    correct_point: float,
    partial_point: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    recall_points: List[float] = []
    precision_points: List[float] = []
    recall_points_unweighted: List[float] = []
    precision_points_unweighted: List[float] = []
    weights: List[float] = []
    token_char_ratios: List[float] = []

    detail_rows: List[Dict[str, object]] = []

    correct_full = 0
    partial = 0
    wrong = 0
    skipped = 0

    for row in rows:
        word = get_word_v2(row)
        morphemes = build_gold_morphemes_v2(row)
        if not morphemes:
            skipped += 1
            continue

        if hasattr(tokenizer, "tokenize"):
            raw_tokens = tokenizer.tokenize(word)
        else:
            raw_tokens = tokenizer(word)
        tokens = normalize_token_list(raw_tokens)

        expected = " ".join(morphemes)
        predicted = " ".join(tokens)
        recall_pt, precision_pt = morph_eval_v2(
            morphemes,
            tokens,
            exclude_single_tok=exclude_single_tok,
            exclude_single_morpheme=exclude_single_morpheme,
            single_tok_point=single_tok_point,
            correct_point=correct_point,
            partial_point=partial_point,
        )

        if len(word) > 0:
            token_char_ratios.append(len(tokens) / len(word))

        if math.isnan(recall_pt) or math.isnan(precision_pt):
            skipped += 1
            detail_label = classify_v2_detail(row, morphemes, tokens)
            if include_all_rows or tokens != morphemes:
                detail_rows.append(
                    {
                        "wordform": word,
                        "lemma": normalize_morph_text(row.get("lemma")) or "",
                        "stem": normalize_morph_text(row.get("stem")) or "",
                        "preceding_part": normalize_morph_text(row.get("preceding_part")) or "",
                        "following_part": normalize_morph_text(row.get("following_part")) or "",
                        "expected_morphtok": expected,
                        "predicted_morphtok": predicted,
                        "morphscore_result": "skipped",
                        "morphscore_detail": detail_label,
                    }
                )
            continue

        weight = get_float(row, "word_freq_norm", 1.0) if freq_scale else 1.0
        weights.append(weight)
        recall_points.append(recall_pt * weight)
        precision_points.append(precision_pt * weight)
        recall_points_unweighted.append(recall_pt)
        precision_points_unweighted.append(precision_pt)

        if recall_pt == correct_point:
            correct_full += 1
            label = "correct"
        elif recall_pt == partial_point:
            partial += 1
            label = "partial"
        else:
            wrong += 1
            label = "wrong"

        detail_label = classify_v2_detail(row, morphemes, tokens)
        if include_all_rows or tokens != morphemes:
            detail_rows.append(
                {
                    "wordform": word,
                    "lemma": normalize_morph_text(row.get("lemma")) or "",
                    "stem": normalize_morph_text(row.get("stem")) or "",
                    "preceding_part": normalize_morph_text(row.get("preceding_part")) or "",
                    "following_part": normalize_morph_text(row.get("following_part")) or "",
                    "expected_morphtok": expected,
                    "predicted_morphtok": predicted,
                    "morphscore_result": label,
                    "morphscore_detail": detail_label,
                }
            )

    total_weight = sum(weights)
    morphscore_recall = sum(recall_points) / total_weight if total_weight else 0.0
    morphscore_precision = sum(precision_points) / total_weight if total_weight else 0.0
    morphscore_recall_unweighted = mean_or_zero(recall_points_unweighted)
    morphscore_precision_unweighted = mean_or_zero(precision_points_unweighted)

    summary = {
        "morphscore_recall": morphscore_recall,
        "morphscore_precision": morphscore_precision,
        "morphscore_f1": f1_or_zero(morphscore_precision, morphscore_recall),
        "morphscore_recall_unweighted": morphscore_recall_unweighted,
        "morphscore_precision_unweighted": morphscore_precision_unweighted,
        "morphscore_f1_unweighted": f1_or_zero(
            morphscore_precision_unweighted,
            morphscore_recall_unweighted,
        ),
        "morphscore_recall_std": std_or_zero(recall_points_unweighted),
        "morphscore_precision_std": std_or_zero(precision_points_unweighted),
        "total_items": float(len(rows)),
        "num_samples": float(len(weights)),
        "mean_token_char_ratio": mean_or_zero(token_char_ratios),
        "correct": float(correct_full),
        "partial": float(partial),
        "wrong": float(wrong),
        "skipped": float(skipped),
    }
    detail_df = pd.DataFrame(
        detail_rows,
        columns=[
            "wordform",
            "lemma",
            "stem",
            "preceding_part",
            "following_part",
            "expected_morphtok",
            "predicted_morphtok",
            "morphscore_result",
            "morphscore_detail",
        ],
    )
    return summary, detail_df


def score_morph_tok(
    df: pd.DataFrame,
    lang_code: str = "eng_latn",
    return_df: bool = True,
    *,
    unique_only: bool = True,
    stem_eq_lemma: bool = True,
    exclude_numbers: bool = True,
    freq_scale: bool = True,
    exclude_single_tok: bool = False,
    exclude_single_morpheme: bool = True,
    single_tok_point: float = 1.0,
    correct_point: float = 1.0,
    partial_point: float = 0.5,
):
    """
    Score an existing morphemic tokenisation column using the local MorphScore v2 evaluator.

    Args:
        df: DataFrame with a `sent_morphtok` column containing space-delimited morpheme tokens.
        lang_code: Language code in ISO 639-3 + ISO 15924 (e.g. 'eng_latn').
        return_df: If True, return `(summary, detail_df)`; else return summary metrics.

    Returns:
        Local MorphScore v2 output.
    """
    token_lists = df["sent_morphtok"].fillna("").astype(str).str.split().tolist()

    class _PreTokenizedTokenizer:
        """Minimal adapter that feeds pre-tokenized sequences to the local scorer."""
        def __init__(self, tokens):
            self._tokens = tokens
            self._i = 0

        def tokenize(self, _text):
            if self._i >= len(self._tokens):
                return []
            out = self._tokens[self._i]
            self._i += 1
            return out

    return score_morph_tok_with_tokenizer(
        _PreTokenizedTokenizer(token_lists),
        lang_code=lang_code,
        return_df=return_df,
        unique_only=unique_only,
        stem_eq_lemma=stem_eq_lemma,
        exclude_numbers=exclude_numbers,
        freq_scale=freq_scale,
        exclude_single_tok=exclude_single_tok,
        exclude_single_morpheme=exclude_single_morpheme,
        single_tok_point=single_tok_point,
        correct_point=correct_point,
        partial_point=partial_point,
    )


def score_morph_tok_with_tokenizer(
    tokenizer,
    lang_code: str = "eng_latn",
    return_df: bool = True,
    *,
    unique_only: bool = True,
    stem_eq_lemma: bool = True,
    exclude_numbers: bool = True,
    freq_scale: bool = True,
    exclude_single_tok: bool = False,
    exclude_single_morpheme: bool = True,
    single_tok_point: float = 1.0,
    correct_point: float = 1.0,
    partial_point: float = 0.5,
):
    """Score a tokenizer directly using the local MorphScore v2 evaluator."""
    rows = filter_v2_rows(
        load_morphscore_v2_data(lang_code).to_dict(orient="records"),
        unique_only=unique_only,
        stem_eq_lemma=stem_eq_lemma,
        exclude_numbers=exclude_numbers,
    )
    summary, detail_df = get_morphscore_v2(
        rows,
        tokenizer,
        freq_scale=freq_scale,
        exclude_single_tok=exclude_single_tok,
        exclude_single_morpheme=exclude_single_morpheme,
        single_tok_point=single_tok_point,
        correct_point=correct_point,
        partial_point=partial_point,
    )
    if return_df:
        return summary, detail_df
    return summary


def unpack_morphscore_v2_output(score_output):
    """Normalize MorphScore v2 output into (summary, detail_df)."""
    if isinstance(score_output, tuple):
        detail_df = next((item for item in score_output if isinstance(item, pd.DataFrame)), None)
        summary = next((item for item in score_output if not isinstance(item, pd.DataFrame)), None)
        return summary, detail_df
    if isinstance(score_output, pd.DataFrame):
        return None, score_output
    return score_output, None


def select_morphscore_v2_rows_for_output(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer mismatch rows for CSV output; fall back to non-perfect rows when needed."""
    if {"expected_morphtok", "predicted_morphtok"} <= set(detail_df.columns):
        mismatch_mask = (
            detail_df["expected_morphtok"].fillna("").astype(str)
            != detail_df["predicted_morphtok"].fillna("").astype(str)
        )
        if mismatch_mask.any():
            return detail_df.loc[mismatch_mask].reset_index(drop=True)
    if "morphscore_result" in detail_df.columns:
        return detail_df[detail_df["morphscore_result"] != "correct"].reset_index(drop=True)
    if "result" in detail_df.columns:
        return detail_df[detail_df["result"] != "correct"].reset_index(drop=True)

    mask = pd.Series(False, index=detail_df.index)
    for col in ("recall", "precision"):
        if col in detail_df.columns:
            mask |= pd.to_numeric(detail_df[col], errors="coerce").fillna(0) < 1

    if mask.any():
        return detail_df.loc[mask].reset_index(drop=True)
    return detail_df.reset_index(drop=True)


def format_morphscore_v2_summary(summary) -> str:
    """Format the main v2 metrics for CLI output when available."""
    if isinstance(summary, dict):
        parts = []
        if "morphscore_recall" in summary:
            parts.append(f"recall={summary['morphscore_recall']:.4f}")
        if "morphscore_precision" in summary:
            parts.append(f"precision={summary['morphscore_precision']:.4f}")
        if "morphscore_f1" in summary:
            parts.append(f"f1={summary['morphscore_f1']:.4f}")
        if "num_samples" in summary:
            parts.append(f"num_samples={int(summary['num_samples'])}")
        if parts:
            return "; ".join(parts)
    if summary is None:
        return "summary unavailable"
    return str(summary)

def morph_eval(morphemes, tokens):
    """Return -1 (wrong), 0 (no split), or 1 (correct) for a 2-part segmentation."""
    if len(tokens) == 1:
        return 0
    for t in range(len(tokens) - 1):
        pt1 = "".join(tokens[: t + 1])
        rest = "".join(tokens[t + 1 :])
        if [pt1, rest] == morphemes:
            return 1
    return -1


def get_morphscore(data: pd.DataFrame, tokenizer):
    """Compute MorphScore-style accuracy over two-part splits."""
    points = []
    error_rows = []
    attempted = 0
    correct = 0
    wrong = 0
    skipped_single_token = 0

    for _, row in data.iterrows():
        morphemes = [str(row["pt1"]).lower(), str(row["rest"]).lower()]
        word = str(row["full_word"]).lower()
        tokens = [str(t).lower() for t in tokenizer(word)]

        point = morph_eval(morphemes, tokens)
        if point == 0:
            skipped_single_token += 1
        else:
            attempted += 1
            points.append(0 if point == -1 else 1)
            if point == 1:
                correct += 1
            else:
                wrong += 1

        if point in {-1, 0, 1}:
            error_rows.append(
                {
                    "full_word": word,
                    "gold": morphemes,
                    "predicted": tokens,
                    "result": (
                        "correct"
                        if point == 1
                        else "skipped_single_token" if point == 0 else "wrong"
                    ),
                }
            )

    morph_score = float(np.mean(points)) if points else 0.0
    error_df = pd.DataFrame(error_rows)
    stats = {
        "attempted": attempted,
        "correct": correct,
        "wrong": wrong,
        "skipped_single_token": skipped_single_token,
        "total_assessed": len(data),
    }
    return morph_score, error_df, stats


def spacy_morph_tokenizer(nlp):
    """Adapter: build a tokenizer callable that applies `spacy_tokenise` to raw text."""
    def _tok(text: str):
        doc = nlp("" if text is None else str(text))
        morphemes, _, _, _, _ = spacy_tokenise(doc)
        return [m.lower() for m in morphemes]
    return _tok


def annotate_error_types(errors_df: pd.DataFrame):
    """
    Label English v1 disagreements with named categories.
    Returns (annotated_df, revised_score) where revised_score treats *_choice labels as correct.
    """
    def classify(row):
        gold = row.get("gold", [])
        pred = row.get("predicted", [])
        if not isinstance(gold, (list, tuple)) or not isinstance(pred, (list, tuple)):
            return row.get("result", "surface_mismatch")
        return classify_english_mismatch_label(
            gold,
            pred,
            result_label=str(row.get("result", "wrong")),
            word=str(row.get("full_word", "")),
        )

    annotated = errors_df.copy()
    annotated["error_type"] = annotated.apply(classify, axis=1)
    scored = annotated[annotated["result"] != "skipped_single_token"].reset_index(drop=True)
    if len(scored) == 0:
        revised_score = 0.0
    else:
        revised_score = (
            (scored["result"] == "correct")
            | scored["error_type"].isin(ENGLISH_CHOICE_LABELS)
        ).mean()
    return annotated, revised_score


def collect_v1_detail_counts(errors_df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    if "error_type" not in errors_df.columns:
        return counts
    for label in errors_df["error_type"].fillna("").astype(str):
        if label in ENGLISH_DETAIL_LABEL_ORDER:
            counts[label] += 1
    return counts


def print_v1_detail_summary(errors_df: pd.DataFrame) -> None:
    counts = collect_v1_detail_counts(errors_df)
    print("V1 detailed disagreement labels:")
    for label in ENGLISH_DETAIL_LABEL_ORDER:
        print(f"  {label}: {counts.get(label, 0)}")


def score_english_morph_data_from_df(
    df: pd.DataFrame,
    morph_col: str = "sent_morphtok",
    url: str = "https://github.com/catherinearnett/morphscore/raw/v1/data/english_morph_data.csv",
    tokenizer=None,
):
    """
    Score the morphemic tokenisation in `morph_col` against the English MorphScore v1 data
    (columns: full_word, pt1, rest). If `tokenizer` is provided, it is used directly;
    otherwise a lookup tokenizer is built from the DataFrame's `morph_col`.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    morph_data = pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine="python")
    if {"full_word", "pt1", "rest"} - set(morph_data.columns):
        raise ValueError("MorphScore data must contain columns: full_word, pt1, rest")
    morph_data["full_word"] = morph_data["full_word"].astype(str).str.lower()
    morph_data["pt1"] = morph_data["pt1"].astype(str).str.lower()
    morph_data["rest"] = morph_data["rest"].astype(str).str.lower()

    if tokenizer is None:
        # Build a lookup from the pre-tokenised column to reuse as a tokenizer.
        lex = {}
        text_col = "utt" if "utt" in df.columns else None
        if text_col:
            for _, row in df[[text_col, morph_col]].dropna(subset=[text_col, morph_col]).iterrows():
                word = str(row[text_col]).strip()
                word_lower = word.lower()
                if word and word_lower not in lex:
                    lex[word_lower] = str(row[morph_col]).lower().split()

        def tokenizer(word: str):
            key = str(word).lower()
            if key in lex:
                return lex[key]
            # fallback: simple char-based split to indicate a segmentation attempt
            return list(key)

    return get_morphscore(morph_data, tokenizer)

def main():
    ap = argparse.ArgumentParser(description="Add morphemic tokenisation and feature flags to an aggregate CSV.")
    ap.add_argument("--input", help="Input CSV path")
    ap.add_argument("--output", help="Output CSV path")
    ap.add_argument("--text-col", default="utt", help="Text column to use (default: utt)")
    ap.add_argument("--tok-ann-col", default="tok_annotations", help="tok_annotations column name")
    ap.add_argument("--spacy-model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg)")
    ap.add_argument("--morphscore", action="store_true", help="Only run MorphScore evaluation on sent_morphtok and exit.")
    ap.add_argument(
        "--morphscore-version",
        choices=("v1", "v2"),
        default="v1",
        help="MorphScore version to use with --morphscore (default: v1).",
    )
    ap.add_argument(
        "--output-morphscore",
        help="Optional path to write MorphScore CSV output. Defaults to --output when set, otherwise a version-specific filename.",
    )
    args = ap.parse_args()

    if args.morphscore:
        nlp = load_spacy(args.spacy_model)
        tokenizer = spacy_morph_tokenizer(nlp)
        if args.morphscore_version == "v2":
            summary, detail_df = unpack_morphscore_v2_output(
                score_morph_tok_with_tokenizer(tokenizer, lang_code="eng_latn", return_df=True)
            )
            out_path = args.output_morphscore or args.output or "english_morphscore_v2.csv"
            out_df = detail_df if detail_df is not None else pd.DataFrame()
            out_df = select_morphscore_v2_rows_for_output(out_df)
            out_df.to_csv(out_path, index=False)
            print(f"MorphScore2 recall (weighted): {summary['morphscore_recall']:.6f}")
            print(f"MorphScore2 precision (weighted): {summary['morphscore_precision']:.6f}")
            print(f"MorphScore2 F1 (weighted): {summary['morphscore_f1']:.6f}")
            print(
                f"MorphScore2 recall (unweighted): "
                f"{summary['morphscore_recall_unweighted']:.6f}"
            )
            print(
                f"MorphScore2 precision (unweighted): "
                f"{summary['morphscore_precision_unweighted']:.6f}"
            )
            print(
                f"MorphScore2 F1 (unweighted): "
                f"{summary['morphscore_f1_unweighted']:.6f}"
            )
            print(f"MorphScore2 recall std: {summary['morphscore_recall_std']:.6f}")
            print(f"MorphScore2 precision std: {summary['morphscore_precision_std']:.6f}")
            print(f"Total assessed words: {int(summary['total_items'])}")
            print(f"Scored words: {int(summary['num_samples'])}")
            print(f"Correct: {int(summary['correct'])}")
            print(f"Partial: {int(summary['partial'])}")
            print(f"Wrong: {int(summary['wrong'])}")
            print(f"Skipped: {int(summary['skipped'])}")
            print(f"Mean token/char ratio: {summary['mean_token_char_ratio']:.6f}")
            print(f"CSV mismatches: {len(out_df)}")
            print_v2_detail_summary(out_df)
            print(f"Wrote: {out_path}")
            return

        score, errors_df, stats = score_english_morph_data_from_df(pd.DataFrame(), tokenizer=tokenizer)
        annotated_errors, revised_score = annotate_error_types(errors_df)
        choice_rows = annotated_errors[
            annotated_errors["error_type"].isin(ENGLISH_CHOICE_LABELS)
        ].reset_index(drop=True)
        errors_wrong = annotated_errors[
            (annotated_errors["result"] == "wrong")
            & ~annotated_errors["error_type"].isin(ENGLISH_CHOICE_LABELS)
        ].reset_index(drop=True)
        out_path = args.output_morphscore or args.output or "english_morphscore_wrong.csv"
        errors_wrong.to_csv(out_path, index=False)
        print(f"V1 MorphScore (boundary accuracy): {score:.6f}")
        print(f"V1 choice-adjusted score: {revised_score:.6f}")
        print(f"Total assessed words: {stats['total_assessed']}")
        print(f"Scored words: {stats['attempted']}")
        print(f"Correct: {stats['correct']}")
        print(f"Wrong: {stats['wrong']}")
        print(f"Skipped single-token predictions: {stats['skipped_single_token']}")
        print(f"Choice rows: {len(choice_rows)}")
        print(f"CSV wrong rows: {len(errors_wrong)}")
        print_v1_detail_summary(annotated_errors)
        print(f"Wrote: {out_path}")
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required unless --morphscore is used.")

    df = pd.read_csv(args.input)

    nlp = load_spacy(args.spacy_model)

    extra_rows: List[dict] = []
    for _, row in df.iterrows():
        extra_rows.append(process_row(row, nlp, args.text_col, args.tok_ann_col))

    extra_df = pd.DataFrame(extra_rows)
    out_df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)

    mask = (
        out_df["spacy_normtok"] #used to be "sent_morphtok"
        .fillna("")
        .str.contains(r"\S+\s+\S")          # at least two tokens
        & ~out_df["sent_morphtok"].str.contains(r"\b(?:xxx|yyy|www)\b", case=False)
        & ~out_df["sent_morphtok"].str.contains(r":", na=False)
        & ~out_df["sent_morphtok"].str.contains(r"-", na=False)
    )
 
    out_df = out_df.loc[mask].reset_index(drop=True)

    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
