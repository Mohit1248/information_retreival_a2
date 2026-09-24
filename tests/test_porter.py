"""
tests/test_porter.py -- the dependency-free Porter stemmer and the
tokenizer built on it (submission/porter.py, submission/lm_utils.py).

Expected stems are Porter's published examples (Porter 1980, steps 1a-1c
and later steps) and standard outputs of the original algorithm.
"""
import pytest

from submission import lm_utils, porter

PORTER_EXAMPLES = {
    # step 1a
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    # step 1b
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled",
    "motoring": "motor", "sing": "sing", "conflated": "conflat", "troubled": "troubl",
    "sized": "size", "hopping": "hop", "tanned": "tan", "falling": "fall",
    "hissing": "hiss", "fizzed": "fizz", "failing": "fail", "filing": "file",
    # step 1c
    "happy": "happi", "sky": "sky",
    # later steps / full pipeline
    "relational": "relat", "conditional": "condit", "rational": "ration", "valenci": "valenc",
    "digitizer": "digit", "generalizations": "gener", "oscillators": "oscil",
    "running": "run", "vaccination": "vaccin", "vaccinations": "vaccin",
    "infection": "infect", "infections": "infect",
}


@pytest.mark.parametrize("word,expected", sorted(PORTER_EXAMPLES.items()))
def test_porter_published_examples(word, expected):
    assert porter.stem(word) == expected


def test_short_words_and_digit_tokens_are_left_alone():
    # length <= 2 is untouched (Porter 1980); digits never count as vowels
    for token in ("is", "as", "a", "19", "cov2"):
        assert porter.stem(token) == token


def test_measure_matches_porters_definition():
    # Porter 1980: m=0 TR, EE, TREE, Y, BY; m=1 TROUBLE, OATS, TREES, IVY; m=2 TROUBLES, PRIVATE, OATEN
    for word, m in [("tr", 0), ("ee", 0), ("tree", 0), ("y", 0), ("by", 0),
                    ("trouble", 1), ("oats", 1), ("trees", 1), ("ivy", 1),
                    ("troubles", 2), ("private", 2), ("oaten", 2)]:
        assert porter._measure(word) == m, word


def test_tokenize_lowercases_removes_stopwords_and_stems():
    assert lm_utils.tokenize("The Infections of the coronavirus, and its vaccines!") == [
        "infect", "coronaviru", "vaccin",
    ]
    # documents and queries share one tokenisation, so word forms meet
    assert lm_utils.tokenize("vaccination") == lm_utils.tokenize("Vaccinations")


def test_query_of_only_stopwords_is_not_empty():
    assert lm_utils.tokenize("to be or not to be") == []
    assert lm_utils.tokenize_query("to be or not to be") != []
    # a normal query is unchanged by the fallback
    assert lm_utils.tokenize_query("what is the origin of COVID-19") == ["origin", "covid", "19"]


def test_words_used_in_the_hand_checked_lm_tests_are_stem_invariant():
    # tests/test_relevance_models.py refers to these terms directly, so the
    # tokenizer must leave them unchanged.
    for w in ("mango", "banana", "kiwi", "durian"):
        assert lm_utils.tokenize(w) == [w]
