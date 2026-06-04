import pandas as pd
import argparse
import spacy
import string
from ast import literal_eval
import re
import nltk
from nltk import Tree

import random

random.seed(10)

nlp = spacy.load("en_core_web_lg")

ed_exceptions = {'aran_reed', 'bed', 'birdfeed', 'birdseed', 'bled', 'bleed',
                 'bunkbed', 'captain_mildred', 'ed','feed', 'fireman_fred', 'flatbed',
                 'fred', 'get_into_bed', 'go_to_bed', 'hundred', 'indeed', 'jed',
                 'knockkneed', 'led', 'mildred', 'mister_reed', 'ned', 'need',
                 'old_macreed', 'playbed', 'red', 'reed', 'reseed',
                 'robber_red', 'shed', 'sled', 'sofabed', 'ted', 'thoroughbred',
                 'wilfred'
                 }

ing_exceptions = {'aboing', 'anything', 'bing', 'blingie^bling^bling', 'boing',  #building, clothing, stocking? have currently not included them as exceptions
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
plural_only = {'binoculars', 'headphones', 'sunglasses', 'glasses',
                 'scissors', 'tweezers', 'jeans', 'pyjamas', 'tights',
                 'knickers', 'shorts', 'trousers', 'pants', 'belongings',
                 'outskirts', 'clothes', 'premises', 'congratulations',
                 'savings', 'earnings', 'stairs', 'goods', 'surroundings',
                 'thanks', 'yours'} #yours added because for some reason it is tagged as NNS by spacy repeatedly

irregular_3rd_person_verbs = {'be', 'have', 'go', 'do'}

corpora_with_thematic_annotation = {'ctb_brown-adam4up+animacy+theta', 'ctb_brown-adam3to4+animacy+theta',
                                    'ctb_brown-eve+animacy+theta', 'ctb_valian+animacy+theta', }
plural_pts = {"NNS", "NNPS"}
verb_3rd_pts = {"VBZ", "AUX", "VBD", }


def remove_unary_subtrees_with_T_terminals(tree):
    if isinstance(tree, Tree):
        # Process children first
        new_children = []
        for child in tree:
            # Recursively process each child
            processed_child = remove_unary_subtrees_with_T_terminals(child)
            if processed_child is not None:
                new_children.append(processed_child)

        # Check if the current tree is a unary subtree with a terminal starting with *T
        if len(new_children) == 1 and isinstance(new_children[0], Tree) and any(t.startswith('*') for t in new_children[0].leaves()):
            return None  # Remove this unary subtree

        return Tree(tree.label(), new_children)
    else:
        return tree

t_example = Tree.fromstring("(ROOT (SQ (VP (AUX do) (NP-<ANIM>-<EXPER-V1> (PRP you)) (VP (VB want) (S-<SUBJMATT-V1> (NP-<ANIM>-<AGENT-V2> (NONE *PRO*)) (VP (TO to))))) (. ?)))")
t_ex = Tree.fromstring("(ROOT (S (NP (PRP you)) (VP (AUX do) (NOT n't) (VP (VB need) (S (NP (-NONE- )) (VP (TO to) (VP (VB make) (NP (PRP it)) (ADVP (RB so) (RB long)))))))) (NP (NNP adam)))")
# print(remove_unary_subtrees_with_T_terminals(t_ex))

def clear_trace(parse):
    #(NP(-NONE-ABAR-WH- *T*-1))
    # Remove the -1 etc notation from the tag the trace relates to
    # -NONE- *PRO*
    pattern = r'-\d+'
    # pattern2 = r'\*[^*]*?(-\d+)?\*?'
    #
    cleared_parse = re.sub(pattern, '', parse)
    # cleared_parse = re.sub(pattern2, '', cleared_parse)
    # cleared_parse = " "
    try:
        nltk_tree = Tree.fromstring(cleared_parse)
        #remove the unary tree including the trace, it has terminal starting *T
        cleared_parse = remove_unary_subtrees_with_T_terminals(nltk_tree).pformat()

    except Exception as e:
        cleared_parse = " "
        # There are only 51 exceptions so won't bother for now to deal with them
    return cleared_parse


# def clear_animacy_and_theta(parse):    # Define a pattern to match and transform the desired parts
#     # Define regex patterns for matching tags
#     tag_pattern = r'(\(\w+)-[^()]+'
#
#     # Function to handle replacements
#     def repl_func(match):
#         tag = match.group(1)  # Extract the tag part
#         if tag == '(PRP$':
#             return match.group(0)  # Return PRP$ as is
#         else:
#             return tag  # Return simplified tag without modifiers
#
#     # Apply replacements using re.sub with a function
#     simplified_text = re.sub(tag_pattern, repl_func, parse)
#
#     return simplified_text

def clear_animacy_and_theta(text): #Need to manually check for = and remove bad examples
    # Define the pattern to match tags
    pattern = re.compile(r'\((\w+)(-[^ ]+)?\s')

    def replacer(match):
        tag = match.group(1)
        return f'({tag} '

    # Replace the patterns in the text
    cleaned_text = pattern.sub(replacer, text)

    return cleaned_text

ex = "(ROOT (S (VP (VB-<V1> look) (PP (IN at) (NP-<INANIM>-<PATIENT-V1> (PRP it))) (ADVP (RB carefully))) (. .)))"
ex2 = "(ROOT (SBARQ (WHNP-1-<INANIM>-<PATIENT-V1> (WP what)) (SQ (VP (AUX do) (NP-<ANIM>-<AGENT-V1> (PRP we)) (VP (VB-<V1> make) (NP (-NONE-ABAR-WH- *T*-1)) (PP (IN in) (NP-<INANIM>-<LOC-V1> (PRP$ our) (NN factory)))))) (. ?)))"
ex3 = "(ROOT (INTJ (UH here)) (, ,) (FRAG (NP-<ANIM>-<EXPER-V1> (PRP you)) (VP (VB-<V1> want) (S-<SUBJMATT-V1> (NP=<ANIM>-<AGENT-V2> (-NONE- *PRO*)) (VP (TO to) (VP (VB-<V2> put) (NP-<INANIM>-<PATIENT-V2> (DT this) (NN one)) (PRT (RP back)) (ADVP (RB there)))))) (ADVP (RB too)) (. ?)))"
# print(clean_tags(ex3))

# def clear_animacy_and_theta(parse):
#     #from the temrinal remove the patter -<ANIM>-<POSSESS-V1>; it might appear more than once
#
#
#     # Define the regex pattern to match '-<...>'
#     pattern = r'-<[^>]*>'
#     pattern2 = r'=<[^>]*'
#     pattern3 = r'<[^>]*'
#     pattern4 = r'-.*?>'
#     pattern5 = r'>'
#     pattern6 = r'\bEXPER(-V[23])?\b'
#
#     # Substitute the matched patterns with an empty string
#     cleared_parse = re.sub(pattern, '', parse)
#     cleared_parse = re.sub(pattern2, '', cleared_parse)
#     cleared_parse = re.sub(pattern3, '', cleared_parse)
#     cleared_parse = re.sub(pattern4, '', cleared_parse)
#     cleared_parse = re.sub(pattern5, '', cleared_parse)
#     cleared_parse = re.sub(pattern6, '', cleared_parse)
#
#     return cleared_parse


def get_lemma(doc, token):
    for t in doc:
        if t.lower_ == token.lower():
            return t.lemma_


def transform_tree(tree_str, conservative = False):
    if tree_str.strip() == "":
        return " "
    nltk_tree = Tree.fromstring(tree_str)
    tree_yield = ' '.join(nltk_tree.leaves())
    doc = nlp(tree_yield)

    def traverse(subtree, doc):
        if isinstance(subtree, Tree):
            # Check if the subtree is a pre-terminal node (one level above the leaf)
            if len(subtree) == 1 and isinstance(subtree[0], str):
                word = subtree[0]

                w_l = get_lemma(doc, word)

                if w_l == word and conservative:
                    pass
                elif word.endswith('ing') and word not in ing_exceptions:
                    base_verb = w_l
                    if base_verb.endswith('ing') and base_verb not in ing_exceptions:
                        base_verb = base_verb[:-3]
                    return Tree(subtree.label(), [Tree('VB', [base_verb]), Tree('ASP', ['ing'])])
                elif word.endswith('ed') and word not in ed_exceptions:
                    base_verb = w_l
                    if base_verb.endswith('ed') and base_verb not in ed_exceptions:
                        base_verb = base_verb[:-2]
                    return Tree(subtree.label(), [Tree('VB', [base_verb]), Tree('T', ['ed'])])
                elif word.endswith('s'):
                    if subtree.label() in {"NNPS", "NNS"} and word not in plural_only:
                        base_n = w_l
                        return Tree(subtree.label(), [Tree(subtree.label()[:-1], [base_n]), Tree('DIV', ['s'])])
                    elif subtree.label() in {"VBZ", "AUX", "VBD"} and w_l not in irregular_3rd_person_verbs:
                        base_verb = w_l
                        return Tree(subtree.label(), [Tree('VB', [base_verb]), Tree('PRS', ['s'])])
            return Tree(subtree.label(), [traverse(child, doc) for child in subtree])
        return subtree

    transformed_tree = traverse(nltk_tree, doc)
    return ' '.join(str(transformed_tree).split())

def transform_tree_conservative(tree_str, conservative = True):
    res = transform_tree(tree_str, True)
    return res


t1 = "(ROOT (S (NP (PRP i)) (VP (AUX was) (VP (VBG crossing) (NP (DT the) (NN street)))) ))"
t1_morph_tok  = "(ROOT (S (NP (PRP i)) (VP (AUX was) (VP (VBG (VB cross) (ASP ing)) (NP (DT the) (NN street))))))"

t2 = "(ROOT (S (NP (PRP you)) (VP (AUX 've) (VP (VBN used) (PRT (RP up)) (ADJP (JJ all) (PP (IN of) (NP (DT the) (NN tape)))))) ))"
t2_morph_tok = "(ROOT (S (NP (PRP you)) (VP (AUX 've) (VP (VBN (VB use) (T ed)) (PRT (RP up)) (ADJP (JJ all) (PP (IN of) (NP (DT the) (NN tape))))))))"

t3 = "(ROOT (FRAG (NP (NNS gifts)) ))"
t3_morph_tok = "(ROOT (FRAG (NP (NNS (NN gift) (DIV s)))))"

t4 = "(ROOT (S (NP (NNP goldilocks)) (VP (VBZ runs) (PP (ADVP (RB away)) (IN from) (NP (DT the) (CD three) (NNS bears)))) ))"
t4_morph_tok = "(ROOT (S (NP (NNP goldilocks)) (VP (VBZ (VB run) (PRS s)) (PP (ADVP (RB away)) (IN from) (NP (DT the) (CD three) (NNS (NN bear) (DIV s)))))))"

t5 = "(ROOT (FRAG (VP (AUX does) (NP (PRP she)) (ADVP (RB ever))) (. .)))"

t6 = "(ROOT (INTJ (UH well)) (, ,) (S (VP (AUX do) (NOT n't) (VP (COP be) (ADJP (JJ scared)))) (. .)))"
t6_morph_tok = "(ROOT (INTJ (UH well)) (, ,) (S (VP (AUX do) (NOT n't) (VP (COP be) (ADJP (JJ (VB scar) (T ed))))) (. .)))"
assert(transform_tree(t1) == t1_morph_tok)
assert(transform_tree(t2) == t2_morph_tok)
assert(transform_tree(t3) == t3_morph_tok)
assert(transform_tree(t4) == t4_morph_tok)
assert(transform_tree(t5) == t5)
assert(transform_tree(t6, True) == t6)
assert(transform_tree_conservative(t6) == t6)
assert(transform_tree(t6) == t6_morph_tok)

def spacy_tokenise(sentence):
  doc = nlp(sentence)
  sent = []

  for t in doc:
    #progressive ing
    if t.lower_[-3:] == 'ing' and t.lower_ not in ing_exceptions: #token.morph.get('Aspect') == ['Prog'] and
      morpheme = t.lower_[-3:]
      sent.append(t.lemma_)
      sent.append(morpheme)
    #regular plurals (-s)
    elif t.tag_ == 'NNS' and t.lower_[-1] == 's' and t.lower_ not in plural_only:
      morpheme = t.lower_[-1:]
      sent.append(t.lemma_)
      sent.append(morpheme)
    #regular past tense
    elif t.lower_[-2:] == 'ed' and  t.lower_ not in ed_exceptions: #token.morph.get('Tense') == ['Past'] and  and token.morph.get('VerbForm')!=['Part']
      morpheme = t.lower_[-2:]
      sent.append(t.lemma_)
      sent.append(morpheme)
    #3rd person present regular (-s)
    elif t.morph.get('Tense') == ['Pres'] and t.morph.get('Person') == ['3'] and t.morph.get('Number') == ['Sing'] and t.lemma_ not in irregular_3rd_person_verbs and t.lower_[-1] == 's':
      morpheme = t.lower_[-1:]
      sent.append(t.lemma_)
      sent.append(morpheme)
    elif not t.is_punct:
      sent.append(t.lower_)

  sent_incl_punct = [t.text for t in doc]

  return sent, sent_incl_punct

def find_constituents_with_ending(tree: str, ending:str) -> list:
    """
    Finds and returns all constituents in a phrase structure tree where a word ends in with the ending.

    Args:
    tree (str): A string representing the phrase structure tree in bracket notation.

    Returns:
    list: A list of constituents (subtrees) where a word ends in 'ing'.
    """
    # Use a regular expression to find all constituents
    constituents = re.findall(r'\(([^()]+)\)', tree)

    ending_constituents = []
    pattern = r'\b\w+' + ending + r'\b'
    for constituent in constituents:
        # Check if the constituent contains a word that ends in 'ing'
        if re.search(pattern, constituent):
            ending_constituents.append(constituent)
    return ending_constituents

def find_ing_ed_constituents(tree: str) -> list:
    """
    Finds and returns all constituents in a phrase structure tree where a word ends in 'ing'.

    Args:
    tree (str): A string representing the phrase structure tree in bracket notation.

    Returns:
    list: A list of constituents (subtrees) where a word ends in 'ing'.
    """
    # Use a regular expression to find all constituents
    constituents = re.findall(r'\(([^()]+)\)', tree)

    ing_constituents = []
    ed_constituents = []
    for constituent in constituents:
        # Check if the constituent contains a word that ends in 'ing'
        if re.search(r'\b\w+ing\b', constituent):
            ing_constituents.append(constituent)
        elif re.search(r'\b\w+ed\b', constituent):
            ed_constituents.append(constituent)

    return ing_constituents, ed_constituents


def find_preterminals_ending(parses, ending):
    preterminals = dict()

    for p in parses:
        endings = find_constituents_with_ending(p, ending)

        for e in endings:
            if e.split()[1].lower() not in ed_exceptions and e.split()[0] not in preterminals.keys():
                preterminals[e.split()[0]] = p
    return preterminals

def find_preterminals(parses):
    ing_preterminals = dict()
    ed_preterminals = dict()

    for p in parses:
        ings, eds = find_ing_ed_constituents(p)

        for ing in ings:
            if ing.split()[1].lower() not in ing_exceptions and ing.split()[0] not in ing_preterminals.keys():
                ing_preterminals[ing.split()[0]] = p

        for ed in eds:
            if ed.split()[1].lower() not in ed_exceptions and ed.split()[0] not in ed_preterminals.keys():
                ed_preterminals[ed.split()[0]] = p

    return ing_preterminals, ed_preterminals

def save_file(lines, name):
    f = open(("/Users/milamarcheva/PycharmProjects/tacl2023/resources/"+name), "w")
    f.writelines([l+"\n" for l in lines])
    f.close()


# def train_val_test_parses(parses):
#     train_end = int(0.8*len(parses))
#     val_end = int(0.9*len(parses))
#
#     save_file(parses[:train_end], "ctb-train.txt")
#     save_file(parses[train_end:val_end], "ctb-valid.txt")
#     save_file(parses[val_end:], "ctb-test.txt")

def train_val_test_parses(df, col = "morph_tok_parses_lower_nopunct", extension = ""):
    brown_adam_parses = df[df.corpus.str.startswith("ctb_brown-adam")][col].tolist()
    no_adam_parses = df[~df.corpus.str.startswith("ctb_brown-adam")][col].tolist()

    train_end = int(0.85 * len(no_adam_parses))

    save_file(no_adam_parses[:train_end], f"ctb-train{extension}.txt")
    save_file(no_adam_parses[train_end:], f"ctb-valid{extension}.txt")
    save_file(brown_adam_parses, f"ctb-test{extension}.txt")



def collapse_to_one_line(input_string):
    # Remove all newline characters
    collapsed_string = re.sub(r'\s+', ' ', input_string)
    return collapsed_string.strip()


def remove_punctuation_constituents(parse_string):
    # Regular expression to match punctuation constituents
    punctuation_pattern = r'\((,|\.|:|``|\'\'|\'|\?|!|;)\s[^\)]+\)'

    # Remove all punctuation constituents
    cleaned_parse = re.sub(punctuation_pattern, '', str(parse_string))

    # Remove any extra spaces that might result from the removal
    cleaned_parse = re.sub(r'\s+', ' ', cleaned_parse).strip()

    return cleaned_parse


def lowercase_terminals(parse_string):
    # Regular expression to find terminal elements
    terminal_pattern = r'(\([A-Z$]+ )([^\(\) ]+)\)'

    # Function to convert terminal to lowercase
    def to_lowercase(match):
        return f'{match.group(1)}{match.group(2).lower()})'

    # Lowercase all terminal elements
    lowercase_parse = re.sub(terminal_pattern, to_lowercase, parse_string)

    return lowercase_parse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='resources/df_ctb.csv', required=True)

    args = vars(parser.parse_args())
    print(args)

    path = args['path']

    df = pd.read_csv(path)

    #COUNT THE NUMBERS OF UTTERANCES FROM EACH SUBCORPUS OF CHILDES TB
    # for subcorpus in df.corpus.unique():
    #     print(f"subcorpus name {subcorpus} and occurences of it {len(df[df.corpus == subcorpus].corpus.tolist())}")

    #WRITE OUT THE BROWN ADAM PARSES TO LOOK AT THEM AND MANUALLY VERIFY
    # brown_adam_parses = df[df.corpus.str.startswith("ctb_brown-adam")].gra.tolist()
    # print(len(brown_adam_parses))
    # save_file(brown_adam_parses, r"brown-parses-original.txt")

    #REMOVE THE FAILED PARSES -- EMPTY
    # df = df.dropna(subset=['morph_tok_parses_lower_nopunct'])
    # df.to_csv("/Users/milamarcheva/PycharmProjects/tacl2023/resources/df_ctb.csv", index=False)

    # ONLY CTB AND SELECTING APPROPROATE COLUMNS ONLY
    # df.drop(columns=["tok", "mor", "pos"], inplace=True)
    # df.to_csv("/Users/milamarcheva/PycharmProjects/tacl2023/resources/df_ctb.csv", index=False)

    # Preprocessing the parses to remove trace and animacy and theta annotations
    # df["gra_cleaned"] = df["gra"].apply(clear_animacy_and_theta)
    # df["gra_cleaned"] = df["gra_cleaned"].apply(clear_trace)
    # df.to_csv(path, index=False)

    # print(df[df.gra_cleaned == " "])


    #INITIAL PREPROCESSING
    # df[['spacy_morph_tokenised', 'spacy_normal_tok_incl_punct']] = pd.DataFrame(
    #     [spacy_tokenise(s) for s in df['wordsjoined']], index=df.index)
    # df.to_csv(path, index=False)
    #
    # print('df updated')
    #
    # sents = [" ".join(s)+'\n' for s in list(df.spacy_morph_tokenised)]
    # # cleaned_sentences = [
    # #     " ".join([t for t in literal_eval(s) if t not in string.punctuation])+'\n'
    # #     for s in sents
    # # ]
    # f = open("/Users/milamarcheva/PycharmProjects/tacl2023/resources/preterminals.txt", "w")
    # f.writelines(sents)
    # f.close()

    #EXPLORING ALL PARSES WITH -ED and -ING and TODO -S
    # parses = list(df[df.corpus!="Manchester"][~df.corpus.isin(corpora_with_thematic_annotation)].gra)
    # spacy_tok_sents = [literal_eval(l) for l in
    #                    list(df[df.corpus!="Manchester"][~df.corpus.isin(corpora_with_thematic_annotation)].spacy_morph_tokenised)]
    #
    # print("The non-manchester, non-thematic-annotated sentences are: ", len(parses),
    #       ". The number of tokens in total, excluding punct, is: ", sum([len(l) for l in spacy_tok_sents]))
    # ing_l, ed_l = find_preterminals(parses)
    #
    # print("ING")
    # for el in ing_l.items():
    #     print(el)
    # print("\nED")
    # for el in ed_l.items():
    #     print(el)
    # print("\nS")
    # preterminals_s = find_preterminals_ending(parses, "s")
    # for pt in preterminals_s.items():
    #     print(pt)


    #ALL CHILDES Treebank PATHS
    # corpora = list(df[df.corpus!="Manchester"].corpus)
    # print(set(corpora))


    # df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(str)

    # FINAL PREPROCESSING
    df["morph_tok_parses_lower_nopunct"] = df["gra"].apply(clear_animacy_and_theta)
    df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(remove_punctuation_constituents)
    df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(lowercase_terminals)
    df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(clear_trace)

    # df["conservative_morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(transform_tree_conservative)
    # df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(transform_tree)

    # df["conservative_morph_tok_parses_lower_nopunct"] = df["conservative_morph_tok_parses_lower_nopunct"].apply(collapse_to_one_line)
    # df["morph_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(collapse_to_one_line)
    df["normal_tok_parses_lower_nopunct"] = df["morph_tok_parses_lower_nopunct"].apply(collapse_to_one_line)

    # df = df.dropna(subset=['conservative_morph_tok_parses_lower_nopunct'])
    # df = df.dropna(subset=['morph_tok_parses_lower_nopunct'])
    #
    # df.to_csv("/Users/milamarcheva/PycharmProjects/tacl2023/resources/df_ctb.csv", index=False)
    #
    #
    # # #WRITING OUT THE CHTB parses in three files: train (80), val(10), test (10)
    # train_val_test_parses(df,"conservative_morph_tok_parses_lower_nopunct", "_cons" )
    # train_val_test_parses(df)
    print(df.head())
    print(df.columns)
    train_val_test_parses(df, col="normal_tok_parses_lower_nopunct", extension="_normal")

    # df["gra"] = df["gra"].apply(clear_animacy_and_theta)
    # save_file(df.gra.tolist(), "all_ctb_parses_noanimacy.txt")
