import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import spacy

try:
    from morphscore.morphscore import MorphScore
except ImportError:  # pragma: no cover
    MorphScore = None
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

def score_morph_tok(df: pd.DataFrame, lang_code: str = "eng_latn", return_df: bool = True):
    """
    Score an existing morphemic tokenisation column using MorphScore v2.

    Args:
        df: DataFrame with a `morph_tok` column containing space-delimited morpheme tokens.
        lang_code: Language code in ISO 639-3 + ISO 15924 (e.g. 'eng_latn').
        return_df: If True, return MorphScore's per-item DataFrame; else return summary metrics.

    Returns:
        MorphScore output (DataFrame when return_df=True, otherwise summary dict/tuple depending on library version).
    """
    if MorphScore is None:
        raise ImportError("MorphScore v2 is not installed. Install with `pip install morphscore2`.")

    token_lists = df["sent_morphtok"].fillna("").astype(str).str.split().tolist()

    class _PreTokenizedTokenizer:
        """Minimal adapter that feeds pre-tokenized sequences to MorphScore."""
        def __init__(self, tokens):
            self._tokens = tokens
            self._i = 0

        def tokenize(self, _text):
            if self._i >= len(self._tokens):
                return []
            out = self._tokens[self._i]
            self._i += 1
            return out

    tokenizer = _PreTokenizedTokenizer(token_lists)
    scorer = MorphScore()

    if hasattr(scorer, "score_tokenizer"):
        return scorer.score_tokenizer(tokenizer=tokenizer, lang_code=lang_code, return_df=return_df)
    if hasattr(scorer, "score"):
        return scorer.score(tokenizer=tokenizer, lang_code=lang_code, return_df=return_df)
    raise AttributeError("Unsupported MorphScore API version: expected `score_tokenizer` or `score`.")

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

    for _, row in data.iterrows():
        morphemes = [str(row["pt1"]).lower(), str(row["rest"]).lower()]
        word = str(row["full_word"]).lower()
        tokens = [str(t).lower() for t in tokenizer(word)]

        point = morph_eval(morphemes, tokens)
        if point != 0:
            points.append(0 if point == -1 else 1)

        if point in {-1, 1}:
            error_rows.append(
                {
                    "full_word": word,
                    "gold": morphemes,
                    "predicted": tokens,
                    "result": "correct" if point == 1 else "wrong",
                }
            )

    morph_score = float(np.mean(points)) if points else 0.0
    error_df = pd.DataFrame(error_rows)
    return morph_score, error_df


def spacy_morph_tokenizer(nlp):
    """Adapter: build a tokenizer callable that applies `spacy_tokenise` to raw text."""
    def _tok(text: str):
        doc = nlp("" if text is None else str(text))
        morphemes, _, _, _, _ = spacy_tokenise(doc)
        return [m.lower() for m in morphemes]
    return _tok


def annotate_error_types(errors_df: pd.DataFrame):
    """
    Label errors as 'choice' when lemma matches but predicted morpheme is one of {ed, ing, s};
    otherwise keep 'wrong'. Correct rows remain 'correct'.
    Returns (annotated_df, revised_score) where revised_score treats 'choice' as correct.
    """
    def classify(row):
        if row.get("result") != "wrong":
            return row.get("result", "correct")
        gold = row.get("gold", [])
        pred = row.get("predicted", [])
        if not isinstance(gold, (list, tuple)) or not isinstance(pred, (list, tuple)):
            return "wrong"
        if len(gold) < 2 or len(pred) < 2:
            return "wrong"
        if gold[0] == pred[0] and gold[1] != pred[1] and pred[1] in {"ed", "ing", "s"}:
            return "choice"
        return "wrong"

    annotated = errors_df.copy()
    annotated["error_type"] = annotated.apply(classify, axis=1)
    total = len(annotated)
    revised_score = (annotated["error_type"] != "wrong").sum() / total if total else 0.0
    return annotated, revised_score


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
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--text-col", default="utt", help="Text column to use (default: utt)")
    ap.add_argument("--tok-ann-col", default="tok_annotations", help="tok_annotations column name")
    ap.add_argument("--spacy-model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg)")
    ap.add_argument("--morphscore", action="store_true", help="Only run MorphScore evaluation on sent_morphtok and exit.")
    ap.add_argument(
        "--output-morphscore",
        help="Optional path to write MorphScore errors (wrong rows only). Defaults to --output when --morphscore is used.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.morphscore:
        nlp = load_spacy(args.spacy_model)
        tokenizer = spacy_morph_tokenizer(nlp)
        score, errors_df = score_english_morph_data_from_df(df, tokenizer=tokenizer)
        annotated_errors, revised_score = annotate_error_types(errors_df)
        errors_wrong = annotated_errors[annotated_errors["error_type"] == "wrong"].reset_index(drop=True)
        out_path = args.output_morphscore or args.output
        errors_wrong.to_csv(out_path, index=False)
        print(
            f"MorphScore: {score:.4f} "
            f"(choice-adjusted: {revised_score:.4f}; wrong errors saved to {out_path}, total_wrong={len(errors_wrong)})"
        )
        return

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
