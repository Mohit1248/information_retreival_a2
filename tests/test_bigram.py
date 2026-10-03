"""
tests/test_bigram.py -- hand-checkable examples for the optional
bigram/proximity extension in submission/feedback.py (BIGRAM_WEIGHT,
query_log_likelihood_interp and friends). Scoped to score_candidates()'s
final scoring step only -- see the BIGRAM_WEIGHT comment in feedback.py
for why relevance_model_feedback()'s RM1/RM2/RM3 estimation is untouched.

Same tiny corpus as tests/test_relevance_models.py, reused so the two
files' hand-worked numbers stay cross-checkable:
  d1 = "mango banana mango"  bigrams: (mango,banana), (banana,mango)
  d2 = "banana kiwi"         bigrams: (banana,kiwi)
  d3 = "durian durian"       bigrams: (durian,durian)
Collection: 4 distinct bigrams, each occurring once -> P(bigram|C) = 1/4
for each of those four, 0 for any other pair. Collection unigram counts
(from test_relevance_models.py): mango 2, banana 2, kiwi 1, durian 2,
total 7.
"""
import math

import pytest

from submission import feedback
from submission.lm_utils import CollectionStats

CORPUS = [("d1", "mango banana mango"), ("d2", "banana kiwi"), ("d3", "durian durian")]


@pytest.fixture
def stats(monkeypatch):
    s = CollectionStats.from_corpus(CORPUS)
    counts, total = feedback._build_bigram_collection_stats(CORPUS)
    monkeypatch.setattr(feedback, "_BIGRAM_COLLECTION_COUNTS", counts)
    monkeypatch.setattr(feedback, "_BIGRAM_COLLECTION_LENGTH", total)
    monkeypatch.setattr(feedback, "_BIGRAM_DOC_CACHE", {})
    return s


def test_bigram_collection_stats_by_hand():
    counts, total = feedback._build_bigram_collection_stats(CORPUS)
    assert total == 4
    assert counts == {("mango", "banana"): 1, ("banana", "mango"): 1, ("banana", "kiwi"): 1, ("durian", "durian"): 1}


def test_bigram_collection_prob(stats):
    assert feedback.bigram_collection_prob(("mango", "banana")) == pytest.approx(0.25)
    assert feedback.bigram_collection_prob(("durian", "durian")) == pytest.approx(0.25)
    assert feedback.bigram_collection_prob(("kiwi", "mango")) == 0.0  # never seen


def test_doc_bigram_counts_by_hand(stats):
    assert feedback.doc_bigram_counts("d1", stats) == {("mango", "banana"): 1, ("banana", "mango"): 1}
    assert feedback.doc_bigram_counts("d2", stats) == {("banana", "kiwi"): 1}
    assert feedback.doc_bigram_counts("d3", stats) == {("durian", "durian"): 1}


def test_doc_bigram_prob_by_hand(stats):
    # mu=0 -> plain MLE. d1 has 2 bigram slots, 1 occurrence each -> 1/2 each.
    assert feedback.doc_bigram_prob(("mango", "banana"), "d1", stats, mu=0.0) == pytest.approx(0.5)
    # d2 has exactly 1 bigram slot, which IS (banana, kiwi) -> 1/1.
    assert feedback.doc_bigram_prob(("banana", "kiwi"), "d2", stats, mu=0.0) == pytest.approx(1.0)
    # (mango, banana) never occurs in d2 at all -> 0/1.
    assert feedback.doc_bigram_prob(("mango", "banana"), "d2", stats, mu=0.0) == pytest.approx(0.0)
    # mu=4 (collection total bigrams, a convenient round number):
    # (count + 4 * P(bigram|C)) / (doc_bigram_length + 4)
    #   d1, (mango,banana): (1 + 4*0.25) / (2 + 4) = 2/6 = 1/3
    assert feedback.doc_bigram_prob(("mango", "banana"), "d1", stats, mu=4.0) == pytest.approx(1 / 3)
    #   d2, (mango,banana) -- never occurs in d2: (0 + 4*0.25) / (1 + 4) = 1/5
    assert feedback.doc_bigram_prob(("mango", "banana"), "d2", stats, mu=4.0) == pytest.approx(1 / 5)


def test_query_log_likelihood_interp_by_hand(stats):
    # query = [mango, banana], doc = d1, mu=7 (unigram smoothing, matches
    # test_relevance_models.py's worked example: P(mango|d1, mu=7) = 0.4),
    # bigram_mu=4, bigram_weight=0.5.
    #   first term (mango): plain unigram -> log(0.4)
    #   second term (banana), with left context "mango":
    #     p_uni = P(banana|d1, mu=7) = (1 + 7*2/7) / (3+7) = 3/10 = 0.3
    #     p_bi  = P((mango,banana)|d1, mu=4) = (1 + 4*0.25) / (2+4) = 1/3
    #     p = 0.5*(1/3) + 0.5*0.3 = 1/6 + 3/20 = 19/60
    expected = math.log(0.4) + math.log(19 / 60)
    got = feedback.query_log_likelihood_interp(["mango", "banana"], "d1", stats, mu=7.0, bigram_mu=4.0, bigram_weight=0.5)
    assert got == pytest.approx(expected)


def test_bigram_weight_zero_matches_plain_unigram(stats):
    # bigram_weight=0 must reduce EXACTLY to query_log_likelihood (the
    # required baseline), not merely approximate it -- this is what makes
    # BIGRAM_WEIGHT=0.0 a safe, zero-risk default.
    for doc_id in ("d1", "d2", "d3"):
        plain = feedback.query_log_likelihood(["mango", "banana"], doc_id, stats, mu=7.0)
        interp = feedback.query_log_likelihood_interp(
            ["mango", "banana"], doc_id, stats, mu=7.0, bigram_mu=4.0, bigram_weight=0.0
        )
        assert interp == pytest.approx(plain)


def test_bigram_weight_one_doc_with_the_bigram_scores_highest(stats):
    # bigram_weight=1.0 -> second term is pure bigram probability. Only d1
    # actually contains the bigram (mango, banana); d2/d3 don't contain
    # "mango" at all, so their bigram term falls back to the mu-smoothed
    # collection prior only -- d1 must score highest.
    scores = {
        d: feedback.query_log_likelihood_interp(["mango", "banana"], d, stats, mu=7.0, bigram_mu=4.0, bigram_weight=1.0)
        for d in ("d1", "d2", "d3")
    }
    assert scores["d1"] == max(scores.values())


def test_single_term_query_is_unaffected_by_bigram_weight(stats):
    # A one-word query has no adjacent pair to form a bigram from --
    # query_log_likelihood_interp must degrade to the plain unigram term
    # regardless of bigram_weight.
    for bw in (0.0, 0.5, 1.0):
        got = feedback.query_log_likelihood_interp(["mango"], "d1", stats, mu=7.0, bigram_mu=4.0, bigram_weight=bw)
        assert got == pytest.approx(math.log(0.4))


def test_prepare_populates_bigram_stats(tmp_path, monkeypatch):
    lines = [f'{{"doc_id": "{d}", "text": "{t}"}}' for d, t in CORPUS]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(feedback, "_STATS", None)
    feedback.prepare(str(path))
    assert feedback._BIGRAM_COLLECTION_LENGTH == 4
    assert feedback.bigram_collection_prob(("mango", "banana")) == pytest.approx(0.25)


def test_bigram_weight_is_a_valid_interpolation_weight():
    # Whatever BIGRAM_WEIGHT is currently shipped with (see feedback.py --
    # measured on the dev set, not assumed), it must stay a valid
    # interpolation weight: bigram_weight=0.0 is separately tested above to
    # reduce EXACTLY to the required unigram baseline, so this extension is
    # always a safe, bounded addition on top of it, never a replacement.
    assert 0.0 <= feedback.BIGRAM_WEIGHT <= 1.0
