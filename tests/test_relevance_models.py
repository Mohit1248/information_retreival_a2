"""
tests/test_relevance_models.py -- hand-checkable examples for the unigram
query-likelihood score and the RM1 / RM2 / RM3 estimators in
submission/feedback.py. Every expected number below was worked out on
paper from the tiny corpus in `stats` (see the comments), not produced by
running the code.
"""
import math

import pytest

from submission import feedback
from submission.lm_utils import CollectionStats

# d1 = apple banana apple   (length 3)
# d2 = banana cherry        (length 2)
# d3 = durian durian        (length 2)
# Collection: apple 2, banana 2, cherry 1, durian 2 -> 7 tokens.
CORPUS = [("d1", "apple banana apple"), ("d2", "banana cherry"), ("d3", "durian durian")]


@pytest.fixture
def stats():
    return CollectionStats.from_corpus(CORPUS)


def test_dirichlet_query_likelihood_by_hand(stats):
    # P(apple|C) = 2/7. mu = 7:
    #   d1: (2 + 7*2/7) / (3 + 7) = 4/10 = 0.4
    #   d2: (0 + 2)     / (2 + 7) = 2/9
    assert feedback.query_log_likelihood(["apple"], "d1", stats, 7.0) == pytest.approx(math.log(0.4))
    assert feedback.query_log_likelihood(["apple"], "d2", stats, 7.0) == pytest.approx(math.log(2 / 9))
    # two query terms multiply (log-probabilities add)
    expected = math.log(0.4) + math.log((1 + 7 * 2 / 7) / (3 + 7))  # banana in d1: 3/10
    assert feedback.query_log_likelihood(["apple", "banana"], "d1", stats, 7.0) == pytest.approx(expected)


def test_rm1_by_hand(stats):
    # mu = 0 -> P(w|D) is the plain ML estimate.
    #   d1: apple 2/3, banana 1/3      d2: banana 1/2, cherry 1/2
    # With P(d1|Q)=0.75, P(d2|Q)=0.25:
    #   apple  = .75*2/3               = 0.5
    #   banana = .75*1/3 + .25*1/2     = 0.375
    #   cherry = .25*1/2               = 0.125
    rm = feedback.estimate_rm1(["d1", "d2"], {"d1": 0.75, "d2": 0.25}, stats, mu=0.0)
    assert rm == pytest.approx({"apple": 0.5, "banana": 0.375, "cherry": 0.125})
    assert sum(rm.values()) == pytest.approx(1.0)


def test_rm1_min_support_drops_single_document_terms(stats):
    # Only banana occurs in both feedback documents.
    rm = feedback.estimate_rm1(["d1", "d2"], {"d1": 0.5, "d2": 0.5}, stats, mu=0.0, min_support=2)
    assert list(rm) == ["banana"]
    assert rm["banana"] == pytest.approx(1.0)


def test_rm2_by_hand(stats):
    # Query = [apple], uniform P(D) = 1/2, mu = 0. For each w,
    #   P(w, q) = sum_D P(q|D) P(w|D) P(D)      (the P(w) factors cancel)
    # P(apple|d1) = 2/3 and P(apple|d2) = 0, so only d1 contributes:
    #   apple  = 2/3 * 2/3 * 1/2 = 2/9
    #   banana = 2/3 * 1/3 * 1/2 = 1/9
    #   cherry = 0   (cherry is only in d2, where the query term has probability 0)
    # Normalised: apple 2/3, banana 1/3.
    rm = feedback.estimate_rm2(["apple"], ["d1", "d2"], {"d1": 0.5, "d2": 0.5}, stats, mu=0.0)
    assert rm == pytest.approx({"apple": 2 / 3, "banana": 1 / 3})


def test_rm2_uses_all_query_terms(stats):
    # Query = [apple, banana], mu = 0, uniform prior. Both factors must be
    # nonzero: only "banana" survives from d2's side (apple is 0 in d2),
    # so per w:  P(w,Q) = P(w) * prod_i sum_D P(q_i|D) P(D|w).
    #   P(w=apple)  = .5*2/3           = 1/3 ; P(D|apple)  = {d1: 1, d2: 0}
    #       factor(apple q)  = 2/3 ; factor(banana q) = 1/3   -> 1/3 * 2/3 * 1/3 = 2/27
    #   P(w=banana) = .5*1/3 + .5*1/2  = 5/12; P(d1|banana) = (1/6)/(5/12) = 2/5, P(d2|banana) = 3/5
    #       factor(apple q)  = 2/3*2/5 + 0 = 4/15
    #       factor(banana q) = 1/3*2/5 + 1/2*3/5 = 2/15 + 3/10 = 13/30
    #       -> 5/12 * 4/15 * 13/30 = 260/5400 = 13/270
    #   cherry: apple factor = 0 -> dropped
    rm = feedback.estimate_rm2(["apple", "banana"], ["d1", "d2"], {"d1": 0.5, "d2": 0.5}, stats, mu=0.0)
    a, b = 2 / 27, 13 / 270
    assert rm == pytest.approx({"apple": a / (a + b), "banana": b / (a + b)})


def test_rm3_interpolation_by_hand():
    rm = {"banana": 0.6, "cherry": 0.4}
    model = feedback.interpolate_rm3(["apple"], rm, lam=0.5, n_terms=10)
    assert model == pytest.approx({"apple": 0.5, "banana": 0.3, "cherry": 0.2})

    # query terms that are also expansion terms add up; lam = 1 ignores feedback
    model = feedback.interpolate_rm3(["banana", "apple"], rm, lam=0.4, n_terms=10)
    assert model == pytest.approx({"apple": 0.2, "banana": 0.2 + 0.36, "cherry": 0.24})
    assert feedback.interpolate_rm3(["apple"], rm, lam=1.0, n_terms=10) == pytest.approx(
        {"apple": 1.0, "banana": 0.0, "cherry": 0.0}
    )


def test_rm3_truncation_keeps_best_terms_and_renormalises():
    rm = {"a1": 0.5, "b1": 0.3, "c1": 0.2}
    model = feedback.interpolate_rm3(["q1"], rm, lam=0.0, n_terms=2)
    assert model == pytest.approx({"q1": 0.0, "a1": 0.625, "b1": 0.375})


def test_query_posterior_weights(stats):
    # scale = |Q| gives P(Q|D) normalised over the set. For Q = [apple], mu = 7:
    # d1: 0.4, d2: 2/9 -> weights 0.4/(0.4+2/9), (2/9)/(0.4+2/9)
    w = feedback.query_posterior_weights(["apple"], ["d1", "d2"], stats, mu=7.0, scale=1.0)
    z = 0.4 + 2 / 9
    assert w == pytest.approx({"d1": 0.4 / z, "d2": (2 / 9) / z})
    # a smaller scale flattens the weights toward uniform
    flat = feedback.query_posterior_weights(["apple"], ["d1", "d2"], stats, mu=7.0, scale=0.01)
    assert abs(flat["d1"] - 0.5) < abs(w["d1"] - 0.5)


def test_rank_by_model_orders_by_cross_entropy(stats):
    # model = all mass on "apple", mu = 7 -> ranks by P(apple|D): d1 (0.4) > d2 (2/9) = d3 (0+2)/(2+7)
    ranked = feedback.rank_by_model({"apple": 1.0}, ["d3", "d2", "d1"], stats, 7.0, k=3)
    assert ranked[0][0] == "d1"
    assert ranked[0][1] == pytest.approx(math.log(0.4))
    assert {d for d, _ in ranked} == {"d1", "d2", "d3"}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    lines = [f'{{"doc_id": "{d}", "text": "{t}"}}' for d, t in CORPUS]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(feedback, "_STATS", None)
    feedback.prepare(str(path))
    return feedback


def test_feedback_only_returns_candidate_docs_and_handles_odd_seeds(prepared):
    pool = ["d1", "d2"]
    # seed is not a subset of the pool, has an unknown id and a duplicate
    out = prepared.relevance_model_feedback("apple", ["d3", "d3", "zzz", "d1"], pool, 10)
    assert {d for d, _ in out} <= set(pool)
    assert len({d for d, _ in out}) == len(out)
    # empty / all-unknown seed falls back to plain query likelihood
    for seed in ([], ["zzz"]):
        fb = prepared.relevance_model_feedback("apple", seed, pool, 10)
        assert [d for d, _ in fb] == [d for d, _ in prepared.score_candidates("apple", pool, 10)]


@pytest.mark.parametrize("variant", ["rm1", "rm2", "rm3"])
def test_all_variants_run_without_prior_score_candidates(prepared, monkeypatch, variant):
    monkeypatch.setattr(prepared, "RM_VARIANT", variant)
    out = prepared.relevance_model_feedback("apple banana", ["d1", "d2"], ["d1", "d2", "d3"], 2)
    assert len(out) == 2
    assert {d for d, _ in out} <= {"d1", "d2", "d3"}
    if variant == "rm3":
        # RM3 keeps the query anchored: "apple" only occurs in d1
        assert out[0][0] == "d1"
    else:
        # with min_support=2 and two seed docs, RM1/RM2 reduce to the one
        # shared term ("banana"), which d2 (1/2) covers better than d1 (1/3)
        assert out[0][0] == "d2"
