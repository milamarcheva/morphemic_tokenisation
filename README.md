# `morphemic_tokenisation`

This repository contains English morphemic tokenisation code, treebank-preprocessing utilities, and ready-to-use morphemically tokenised data resources for CHILDES Treebank and PTB-style materials.

## English scripts

- [`brown_functional_morphemes_partIII_dissertation_datasplit.ipynb`](brown_functional_morphemes_partIII_dissertation_datasplit.ipynb) is the original notebook for exploratory English morphemic tokenisation, Brown functional morpheme analyses, and dataset-split experiments. It is useful for tracing the development of the English tokeniser, but it is notebook-based rather than the main reusable pipeline.
- [`morph_tok_english_based_on_partIII.py`](morph_tok_english_based_on_partIII.py) is the plain-text script version of the English morphemic tokeniser and Brown-order feature extractor. It can read plain text or CHAT files and can optionally export Brown-feature counts alongside morphemically tokenised output.
- [`ptb_style_tree_english_morphemic_tokenisation.py`](ptb_style_tree_english_morphemic_tokenisation.py) is the PTB/CHILDES-TB parse preprocessing script. It strips animacy/theta annotations and traces, removes punctuation constituents, lowercases terminals, and writes train/valid/test tree splits; this is the tree-oriented preprocessing step associated with the ready-made CHILDES-TB resources in this repository.
- [`add_morphemic_features.py`](add_morphemic_features.py) is the current dataframe-oriented English workflow. It augments aggregate CSVs with `sent_morphtok`, `spacy_normtok`, POS/morph columns, and optional MorphScore evaluation.

## Data

- [`data/childes_treebank`](data/childes_treebank) contains ready-to-use CHILDES Treebank train/valid/test splits in two versions: `ctb-data_normal` for normalised PTB-style tree strings and `ctb-data_morph` for morphemically tokenised tree strings. The underlying CHILDES Treebank derives from the Pearl_Sprouse derived corpus on TalkBank; this repository contains redistributed prepared splits, while the original underlying resource should be cited as Pearl & Sprouse (2013).
- [`data/ptb`](data/ptb) contains PTB train/valid/test sentence files in both standard and morphemically tokenised forms: `ptb_train.txt`, `ptb_valid.txt`, `ptb_test.txt`, `ptb_train_morphtok.txt`, `ptb_valid_morphtok.txt`, and `ptb_test_morphtok.txt`. The underlying Penn Treebank should be cited as Marcus, Santorini, and Marcinkiewicz (1993).

## Citations

- When using the morphemic tokenisation itself, cite the related paper: <https://escholarship.org/content/qt1wh925mk/qt1wh925mk.pdf>
```bibtex
@inproceedings{Marcheva2025Functional,
  author    = {Marcheva, Mila and Biberauer, Theresa and Sun, Weiwei},
  title     = {Functional category induction with theory-neutral cognitive biases},
  booktitle = {Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume    = {47},
  year      = {2025},
  url       = {https://escholarship.org/uc/item/1wh925mk}
}
```
- When using the bracketed-tree morphemic tokenisation or the ready-made morphemically tokenised CHILDES-TB files in this repository, cite Marcheva, Biberauer, and Sun (2025): <https://aclanthology.org/2025.cmcl-1.7/>
```bibtex
@inproceedings{marcheva-etal-2025-profiling,
    title = "Profiling neural grammar induction on morphemically tokenised child-directed speech",
    author = "Marcheva, Mila  and
      Biberauer, Theresa  and
      Sun, Weiwei",
    editor = "Kuribayashi, Tatsuki  and
      Rambelli, Giulia  and
      Takmaz, Ece  and
      Wicke, Philipp  and
      Li, Jixing  and
      Oh, Byung-Doh",
    booktitle = "Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.cmcl-1.7/",
    doi = "10.18653/v1/2025.cmcl-1.7",
    pages = "47--54",
    ISBN = "979-8-89176-227-5"
   ```
- When using the underlying CHILDES Treebank resource, cite Pearl & Sprouse (2013) via the TalkBank Pearl_Sprouse derived-corpus page: <https://talkbank.org/childes/access/Derived/Pearl_Sprouse.html>
```bibtex
@article{pearl_sprouse_2013, 
author = {Lisa Pearl and Jon Sprouse},
year = 2013, 
title = {Syntactic islands and learning biases: Combining experimental syntax and computational modeling to investigate the language acquisition problem.}, 
journal = {Language Acquisition}, 
url = {https://ling.auf.net/lingbuzz/001493},
}
```
- When using the underlying Penn Treebank resource, cite Marcus, Santorini, and Marcinkiewicz (1993): <https://aclanthology.org/J93-2004/>
```bibtex
@article{PennTreebank,
author = {Marcus, Mitchell P. and Marcinkiewicz, Mary Ann and Santorini, Beatrice},
title = {Building a Large Annotated Corpus of English: The Penn Treebank},
year = {1993},
issue_date = {June 1993},
publisher = {MIT Press},
address = {Cambridge, MA, USA},
volume = {19},
number = {2},
issn = {0891-2017},
journal = {Comput. Linguist.},
month = jun,
pages = {313–330},
numpages = {18}
}
```
