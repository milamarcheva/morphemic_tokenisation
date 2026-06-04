import pandas as pd
import re
from collections import defaultdict

# -------------------------
# CONFIGURATION
# -------------------------

# Example: column that contains morpheme-level strings
MOR_COL = "mor"   # change if needed

# -------------------------
# REGEX DEFINITIONS
# -------------------------
# We ONLY keep be-forms explicitly where requested.
# All patterns are regexes over MOR tags.

CATEGORY_REGEXES = {
    "morpheme_ing": [
        r"\b(?:verb|aux)\|[^-]*-Ger\b"
    ],

    "prep_in_on": [
        r"\badp\|in\b",
        r"\badp\|on\b",
    ],

    "plural_s": [
        r"\b(noun|propn)\|[^-]*-Plur\b"
    ],

    "possessive_s": [
        r"\bpart\|s\b"
    ],

    "irregular_past": [
        r"\b(?:verb|aux)\|[^-]*-Past[^-]*-irr\b"
    ],

    "regular_past": [
        r"\bverb\|[^-]*-Past\b"
    ],

    "articles": [
        r"\bdet\|.*-Art\b"
    ],

    "third_person_s": [
        r"\b(?:verb|aux)\|[^-]*-Pres-S3\b"
    ],

    "third_person_irregular": [
        r"\b(?:verb|aux)\|(?:do|have)-Fin-Ind-Pres-S3\b"
    ],

    # -------------------------
    # BE ONLY FROM HERE ON
    # -------------------------

    "contractible_aux": [
        r"\baux\|be-Fin-Ind-Pres-S[123]\b"
    ],

    "contractible_cop": [
        r"\baux\|be-Fin-Ind-Pres-S[123]\b"
    ],

    "uncontractible_aux": [
        r"\baux\|be-Fin-Ind-Past-S[123]-irr\b"
    ],

    "uncontractible_cop": [
        r"\baux\|be-Fin-Ind-Past-S[123]-irr\b"
    ],
}

# -------------------------
# CORE FUNCTION
# -------------------------

def extract_morphological_types(df: pd.DataFrame, column: str):
    """
    Extract sets of MOR tags per category using regexes.
    Only BE is kept explicitly where required.
    """
    results = defaultdict(set)

    for val in df[column].dropna():
        for category, patterns in CATEGORY_REGEXES.items():
            for pat in patterns:
                for match in re.findall(pat, val):
                    # Extract the full tag that matched
                    full_match = re.search(pat, val)
                    if full_match:
                        results[category].add(full_match.group(0))

    return results

# -------------------------
# EXAMPLE USAGE
# -------------------------

if __name__ == "__main__":
    # Example dummy dataframe
    # data = {
    #     "sent_morphtok": [
    #         "verb|sleep-Ger-S aux|be-Fin-Ind-Pres-S3",
    #         "noun|bubble-Plur",
    #         "aux|be-Fin-Ind-Past-S3-irr",
    #         "det|the-Def-Art",
    #         "verb|get-Part-Past-S-irr",
    #         "verb|close-Part-Past-S",
    #         "aux|do-Fin-Ind-Pres-S3",
    #     ]
    # }

    # df = pd.DataFrame(data)
    
    df = pd.read_csv("data/dfs/engna_aggregate_df.csv")
    results = extract_morphological_types(df, MOR_COL)

    # -------------------------
    # PRINT RESULTS
    # -------------------------
    for category, values in results.items():
        print(f"\n{category}")
        for v in sorted(values):
            print(f"  {v}")