"""
tests/test_proximity.py -- hand-checkable examples for the optional
proximity extension in submission/feedback.py (PROXIMITY_WEIGHT,
proximity_bonus(), doc_tokens()). Scoped to score_candidates()'s final
scoring step only -- see the PROXIMITY_WEIGHT comment in feedback.py for
why relevance_model_feedback()'s RM1/RM2/RM3 estimation is untouched.

d1 = "mango banana durian mango"  (tokens, 0-indexed: mango=0, banana=1,
     durian=2, mango=3)
"""
import pytest

from submission import feedback
from submission.lm_utils import CollectionStats

CORPUS = [("d1", "mango banana durian mango"), ("d2", "mango kiwi")]


@pytest.fixture
def stats(monkeypatch):
    # doc_tokens()'s cache is keyed by doc_id only, not tied to a specific
    # CollectionStats instance (mirrors prepare()'s own reset of it) -- this
    # file's "d1" text differs from other test files' "d1" convention, so
    # without this the cache would leak stale tokens across test modules.
    monkeypatch.setattr(feedback, "_DOC_TOKENS_CACHE", {})
    return CollectionStats.from_corpus(CORPUS)


def test_doc_tokens_by_hand(stats):
    assert feedback.doc_tokens("d1", stats) == ["mango", "banana", "durian", "mango"]


def test_proximity_bonus_by_hand(stats):
    # query=[mango, durian]: durian at position 2; mango at 0 and 3.
    #   pair (2,0): distance 2 -> 1/2
    #   pair (2,3): distance 1 -> 1/1
    #   total = 0.5 + 1.0 = 1.5
    assert feedback.proximity_bonus(["mango", "durian"], "d1", stats, window=8) == pytest.approx(1.5)


def test_proximity_bonus_respects_the_window(stats):
    assert feedback.proximity_bonus(["mango", "durian"], "d1", stats, window=1) == pytest.approx(1.0)
    assert feedback.proximity_bonus(["mango", "durian"], "d1", stats, window=0) == pytest.approx(0.0)


def test_proximity_bonus_zero_when_a_term_is_missing(stats):
    assert feedback.proximity_bonus(["mango", "kiwi"], "d1", stats, window=8) == pytest.approx(0.0)


def test_proximity_bonus_zero_for_unknown_doc(stats):
    assert feedback.proximity_bonus(["mango", "durian"], "zzz", stats, window=8) == 0.0


def test_proximity_bonus_counts_a_term_against_itself_as_zero_pairs(stats):
    assert feedback.proximity_bonus(["mango", "mango"], "d1", stats, window=8) == pytest.approx(0.0)


def test_proximity_weight_is_a_valid_non_negative_multiplier():
    assert feedback.PROXIMITY_WEIGHT >= 0.0
    assert feedback.PROXIMITY_WINDOW >= 1


def test_proximity_weight_zero_leaves_ql_rerank_unaffected(tmp_path, monkeypatch):
    lines = [f'{{"doc_id": "{d}", "text": "{t}"}}' for d, t in CORPUS]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(feedback, "_STATS", None)
    feedback.prepare(str(path))
    monkeypatch.setattr(feedback, "BIGRAM_WEIGHT", 0.0)  # isolate from the bigram extension
    monkeypatch.setattr(feedback, "PROXIMITY_WEIGHT", 0.0)
    off = feedback.score_candidates("mango durian", ["d1", "d2"], 2)
    assert off == feedback._ql_rerank("mango durian", ["d1", "d2"], 2, feedback._STATS)
    monkeypatch.setattr(feedback, "PROXIMITY_WEIGHT", 5.0)
    on = feedback.score_candidates("mango durian", ["d1", "d2"], 2)
    assert on != off
