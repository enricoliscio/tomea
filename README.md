# Tomea

This repository contains the code needed to execute the Tomea method as described in the paper _Is Moral Expression Context Dependent? An Explainable Method for Comparing Moral Rhetoric across Contexts_.

The MFTC dataset can be downloaded from [this link](https://osf.io/k5n7y/) through the Twitter API. The trained models are provided separately.

Tomea should be executed as follows:
- **Lexicon generation** will generate the moral lexicons in `data/lexicons`. Run as follows:
	- `python3 lexicon_generation/shap_lexicon_gen.py` to obtain word importances through SHAP;
	- `python3 lexicon_generation/add_lemmas.py` to lemmatize the words in the lexicons;
	- `python3 lexicon_generation/shap_lexicon_gen.py` to group words with the same lemma.
- **Lexicon comparison** will generate _m_-distances and _c_-distances in `data/comparisons`. Run `python3 lexicon_comparison/lexicons_comparison.py` to do so.

All results are separately provided, together with the code used for data analysis and comparison with out-of-context performances and crowd evaluation.
