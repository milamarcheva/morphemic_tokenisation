import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from add_morphemic_features import (  # noqa: E402
    is_article,
    is_contr_aux,
    is_contr_copula,
    is_in,
    is_irregular_3rd_present,
    is_irregular_past,
    is_on,
    is_poss,
    is_progressive,
    is_reg_plural,
    is_regular_3rd_present,
    is_regular_past,
    is_uncontr_aux,
    is_uncontr_copula,
    spacy_morph_tokenizer,
    tokens_aligned,
    spacy_tokenise
)


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        pytest.skip("spaCy model en_core_web_lg not installed")


# def test_progressive(nlp):
#     assert is_progressive(nlp("We are dining.")[2])


# def test_prepositions(nlp):
#     assert is_in(nlp("The truck is in the box.")[3])
#     assert is_on(nlp("The truck is on the table.")[3])


# def test_regular_plural(nlp):
#     assert is_reg_plural(nlp("The trucks are on the table.")[1])
#     assert not is_reg_plural(nlp("The women are on the table.")[1])
#     assert not is_reg_plural(nlp("The scissors are on the table.")[1])


# def test_irregular_past(nlp):
#     assert is_irregular_past(nlp("We went skiing.")[1])
#     assert not is_irregular_past(nlp("We played volleyball.")[1])


# def test_possessive(nlp):
#     assert is_poss(nlp("Jim's jacket is here.")[1])
#     assert not is_poss(nlp("He's here.")[1])


# def test_uncontractible_copula(nlp):
#     assert is_uncontr_copula(nlp("Jim is here.")[1])
#     assert not is_uncontr_copula(nlp("He's here.")[1])
#     assert is_uncontr_copula(nlp("They are here.")[1])
#     assert not is_uncontr_copula(nlp("He is swimming.")[1])


# def test_articles(nlp):
#     assert is_article(nlp("The dog is here.")[0])
#     assert is_article(nlp("I have a dog.")[2])
#     assert is_article(nlp("you want to see the book")[4])


# def test_regular_past(nlp):
#     assert not is_regular_past(nlp("We went skiing.")[1])
#     assert is_regular_past(nlp("We played volleyball.")[1])


# def test_regular_third_person(nlp):
#     assert not is_regular_3rd_present(nlp("She goes skiing.")[1])
#     assert is_regular_3rd_present(nlp("She likes skiing.")[1])


# def test_irregular_third_person(nlp):
#     assert is_irregular_3rd_present(nlp("She goes skiing.")[1])
#     assert not is_irregular_3rd_present(nlp("She likes skiing.")[1])


# def test_uncontractible_aux(nlp):
#     assert not is_uncontr_aux(nlp("Jim is here.")[1])
#     assert not is_uncontr_aux(nlp("He's here.")[1])
#     assert not is_uncontr_aux(nlp("They are here.")[1])
#     assert is_uncontr_aux(nlp("He is swimming.")[1])
#     assert is_uncontr_aux(nlp("Are they learning?")[0])


# def test_contractible_copula(nlp):
#     assert not is_contr_copula(nlp("Jim is here.")[1])
#     assert is_contr_copula(nlp("He's here.")[1])
#     assert not is_contr_copula(nlp("They are here.")[1])
#     assert is_contr_copula(nlp("I'm ready.")[1])
#     assert not is_contr_copula(nlp("He is swimming.")[1])


# def test_contractible_aux(nlp):
#     assert not is_contr_aux(nlp("Jim is here.")[1])
#     assert not is_contr_aux(nlp("He's here.")[1])
#     assert not is_contr_aux(nlp("They are here.")[1])
#     assert not is_contr_aux(nlp("He is swimming.")[1])
#     assert is_contr_aux(nlp("I'm swimming.")[1])
#     assert is_contr_aux(nlp("He's swimming.")[1])
#     assert is_contr_aux(nlp("They're swimming.")[1])


def test_spacy_morph_tokenizer_wont(nlp):
    tok, _, _, _, _= spacy_tokenise(nlp("I won't make it."))
    print(tok)
    assert tok == ["i", "wo", "n't", "make", "it"]


def test_spacy_morph_tokenizer_cant(nlp):
    tok, _, _, _, _ = spacy_tokenise(nlp("I can't do it."))
    assert tok == ["i", "ca", "n't", "do", "it"]


def test_spacy_morph_tokenizer_hers(nlp, pytest=True):
    tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("Mummy used her's for mopping up that sick"))
    # print(tok)
    # print(sent_norm_tok)
    # print(postags_morphtok)
    # print(postags_normtok)
    assert tok == ["mummy", "use", "ed", "her", "'s", "for", "mop", "ing", "up", "that", "sick"]

# def test_spacy_morph_tokenizer_capital(nlp, pytest=True):
#     tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("his Uncle gave Aladdin a ring ."))
#     print(tok)
#     print(sent_norm_tok)
#     print(postags_morphtok)
#     print(postags_normtok)
#     assert tok == ["his", "uncle", "gave", "aladdin", "a", "ring", "." ]

def test_spacy_morph_tokenizer_capital(nlp, pytest=True):
    tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("you read Feed the Animals , Eve ..."))
    print(tok)
    print(sent_norm_tok)
    print(postags_morphtok)
    print(postags_normtok)
    assert tok == ["you", "read", "feed", "the", "animal", "s", "eve" ]

def test_spacy_morph_tokenizer_contracted(nlp, pytest=True):
    tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("think Nana'll come over for supper ?"))
    print(tok)
    print(sent_norm_tok)
    print(postags_morphtok)
    print(postags_normtok)
    assert tok == ["think", "nana", "'ll", "come", "over", "for", "supper" ]
    
    tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("anything'd be an improvement ."))
    print(tok)
    assert tok == ["anything", "'d", "be", "an", "improvement"]

    tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("she's a cute dog"))
    print(tok)
    assert tok == ["she", "'s", "a", "cute", "dog" ]

def test_spacy_morph_tokenizer_plu(nlp, pytest=True):
    tok, sent_norm_tok, postags_morphtok, postags_normtok, morlvl_fulltags_normtok= spacy_tokenise(nlp("and some Adams !,"))
    print(tok)
    print(sent_norm_tok)
    print(postags_morphtok)
    print(postags_normtok)
    assert tok == ["and", "some", "adam", "s"]
    




def test_tokens_aligned():
    aligned = [
        ["you'll", "pron|you-Prs-Nom-S2~aux|will-Fin-S", "1|3|NSUBJ"],
        ["get", "verb|get-Inf-S", "2|3|AUX"],
        ["dizzy", "adj|dizzy-S1", "3|5|ROOT"],
        ["again", "adv|again", "4|3|XCOMP"],
        ["?", "?", "5|3|ADVMOD"],
        ["", "None", "6|3|PUNCT"],
    ]
    aligned2 = [["where", "adv|where", "1|5|ADVMOD"], ["does", "aux|do-Fin-Ind-Pres-S3", "2|5|AUX"], ["the", "det|the-Def-Art", "3|4|DET"], ["ladder", "noun|ladder", "4|5|NSUBJ"], ["go", "verb|go-Inf-S", "5|5|ROOT"], ["?", "?", "6|5|PUNCT"]]
    misaligned = [
        ["you'll", "verb|get-Inf-S", "2|3|AUX"],
        ["get", "adj|dizzy-S1", "3|5|ROOT"],
        ["dizzy", "adv|again", "4|3|XCOMP"],
        ["again", "?", "5|3|ADVMOD"],
        ["?", "6|3|PUNCT"],
    ]
    assert tokens_aligned(aligned2) is True
    assert tokens_aligned(misaligned) is False


# using mor information:
#what happens when Mummy's sleeping ?
#"[["what", "pron|what-Int-S1", "1|2|NSUBJ"], ["happens", "verb|happen-Fin-Ind-Pres-S3", "2|6|ROOT"], ["when", "adv|when", "3|6|ADVMOD"], ["Mummy's", "propn|Mummy~part|s", "4|6|NMOD-POSS"], ["sleeping", "noun|sleeping-Ger", "5|4|CASE"], ["?", "?", "6|2|ADVCL"], ["", null, "7|2|PUNCT"]]"

# it won't bite you .
# "[["it", "pron|it-Prs-Nom-S3", "1|4|NSUBJ"], ["won't", "aux|will-Fin-S~part|not", "2|4|AUX"], ["bite", "verb|bite-Inf-S", "3|4|ADVMOD"], ["you", "pron|you-Prs-Acc-S2", "4|5|ROOT"], [".", ".", "5|4|OBJ"], ["", null, "6|4|PUNCT"]]"
# it wo n't bite
# # mommy doesn't have her hand in it .
# #"[["mommy", "noun|mommy", "1|4|NSUBJ"], ["doesn't", "aux|do-Fin-Ind-Pres-S3~part|not", "2|4|AUX"], ["have", "verb|have-Inf-S", "3|4|ADVMOD"], ["her", "pron|her-Prs-Gen-S3", "4|8|ROOT"], ["hand", "noun|hand-Acc", "5|6|NMOD-POSS"], ["in", "adp|in", "6|4|OBJ"], ["it", "pron|it-Prs-Acc-S3", "7|8|CASE"], [".", ".", "8|4|OBL"], ["", null, "9|4|PUNCT"]]"

# here's the towel
# "[["here's", "adv|here~aux|be-Fin-Ind-Pres-S3", "1|4|ROOT"], ["the", "det|the-Def-Art", "2|1|COP"], ["towel", "noun|towel", "3|4|DET"], [".", ".", "4|1|NSUBJ"], ["", null, "5|1|PUNCT"]]",
# here 's the towel

# brush Alice's hair !
# "[["brush", "verb|brush-Fin-Imp-S", "1|4|ROOT"], ["Alice's", "propn|Alice~part|s", "2|4|NMOD-POSS"], ["hair", "noun|hair-Acc", "3|2|CASE"], ["!", "!", "4|1|OBJ"], ["", null, "5|1|PUNCT"]]",
# brush alice 's hair

# oh , I'll open the door , Okay . 
# "[["oh", "intj|oh", "1|5|DISCOURSE"], [",", "cm|cm", "2|1|PUNCT"], ["I'll", "pron|I-Prs-Nom-S1~aux|will-Fin-S", "3|5|NSUBJ"], ["open", "verb|open-Inf-S", "4|5|AUX"], ["the", "det|the-Def-Art", "5|9|ROOT"], ["door", "noun|door-Acc", "6|7|DET"], [",", "cm|cm", "7|5|OBJ"], ["Okay", "intj|okay", "8|9|PUNCT"], [".", ".", "9|5|DISCOURSE"], ["", null, "10|5|PUNCT"]]"
# oh i 'll open the door okay

# where'd it go ?
# "[["where'd", "adv|where~aux|would-Fin-S", "1|4|ADVMOD"], ["it", "pron|it-Prs-Nom-S3", "2|4|AUX"], ["go", "verb|go-Inf-S", "3|4|NSUBJ"], ["?", "?", "4|4|ROOT"], ["", null, "5|4|PUNCT"]]"
# where 'd it go

# I thought she'd left .
# "[["I", "pron|I-Prs-Nom-S1", "1|2|NSUBJ"], ["thought", "verb|think-Fin-Ind-Past-S1-irr", "2|5|ROOT"], ["she'd", "pron|she-Prs-Nom-S3~aux|would-Fin-Ind-Past-S3", "3|5|NSUBJ"], ["left", "verb|leave-Part-Past-S-irr", "4|5|AUX"], [".", ".", "5|2|CCOMP"], ["", null, "6|2|PUNCT"]]"
# i thought she 'd left

# i ,like ,dogs
# i like dog s
