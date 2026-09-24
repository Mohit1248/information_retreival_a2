"""
submission/porter.py -- a dependency-free implementation of the Porter
(1980) suffix-stripping stemmer, written from the algorithm description in
M.F. Porter, "An algorithm for suffix stripping", Program 14(3), 1980.

Assignment 1 used nltk's PorterStemmer; nltk is not guaranteed to exist on
the grading machine (no network, "basic packages" only), so this replaces
it. It is validated against nltk's ORIGINAL_ALGORITHM mode on the full
corpus vocabulary during development (nltk is NOT imported here), and
against Porter's published examples in tests/test_porter.py.

stem(word) expects a lowercase token (letters/digits) and returns its stem.
"""
from typing import List, Tuple

_VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """m in Porter's [C](VC)^m[V] form: the number of vowel->consonant
    transitions in the stem."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if prev_vowel and not vowel:
            m += 1
        prev_vowel = vowel
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """consonant-vowel-consonant, where the last consonant is not w, x or y."""
    n = len(word)
    return (
        n >= 3
        and _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _replace_if(word: str, rules: List[Tuple[str, str]], min_measure: int) -> str:
    """Apply the first (longest-suffix) matching rule whose stem has
    measure > min_measure. If a suffix matches but the measure condition
    fails, no other rule in the group is tried (as in Porter's paper)."""
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if _measure(stem) > min_measure:
                return stem + replacement
            return word
    return word


# Rules are listed longest suffix first so that the first match is the
# longest match.
_STEP2 = [
    ("ational", "ate"), ("ization", "ize"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("biliti", "ble"), ("tional", "tion"), ("entli", "ent"),
    ("ousli", "ous"), ("ation", "ate"), ("alism", "al"), ("aliti", "al"),
    ("iviti", "ive"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
    ("abli", "able"), ("alli", "al"), ("ator", "ate"), ("eli", "e"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
_STEP4 = [
    ("ement", ""), ("ance", ""), ("ence", ""), ("able", ""), ("ible", ""),
    ("ment", ""), ("ant", ""), ("ent", ""), ("ion", ""), ("ism", ""),
    ("ate", ""), ("iti", ""), ("ous", ""), ("ive", ""), ("ize", ""),
    ("al", ""), ("er", ""), ("ic", ""), ("ou", ""),
]


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _has_vowel(stem):
                return word
            word = stem
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if _ends_double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if _measure(word) == 1 and _ends_cvc(word):
                return word + "e"
            return word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step4(word: str) -> str:
    for suffix, replacement in _STEP4:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if suffix == "ion" and not stem.endswith(("s", "t")):
                return word
            return stem + replacement if _measure(stem) > 1 else word
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]
    return word


def stem(word: str) -> str:
    """Return the Porter stem of a lowercase word. Words of length <= 2 are
    returned unchanged."""
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _replace_if(word, _STEP2, 0)
    word = _replace_if(word, _STEP3, 0)
    word = _step4(word)
    return _step5(word)
