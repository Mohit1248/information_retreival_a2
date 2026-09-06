"""
submission/lm_utils.py

Small, shared, non-graded utilities for building unigram language-model
statistics. You are welcome to use this as-is -- tokenising text and
counting terms is plumbing, not the ranking/feedback logic the assignment
grades -- or replace it with your own (e.g. if you want stemming or a
different tokenisation scheme; nothing here is required).

Design note (why per-document term counts are computed lazily): you are
never asked to rank the whole corpus (see docs/SUBMISSION_INTERFACE.md --
score_candidates()/relevance_model_feedback() only ever score the
provided candidate pool, at most ~100 documents). Eagerly tokenising and
counting terms for every document in a several-hundred-thousand-document
corpus up front, when at most ~100 of them will ever actually be scored
for a given query, would reintroduce exactly the kind of indexing-at-scale
engineering Assignment 1 already covered and this assignment is
deliberately not re-testing. CollectionStats below still reads the whole
corpus once in prepare() -- collection-wide background statistics
(P(w|C), Section 3.1) legitimately need that -- but per-document term
counts are computed only for documents you actually ask about, the first
time you ask, and cached from then on.
"""
import re
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-token tokeniser. Deliberately simple (no
    stemming, no stopword removal) -- add either yourself if you want them;
    just apply the same tokenize() consistently to documents and queries."""
    return _TOKEN_RE.findall(text.lower())


class CollectionStats:
    """Collection-wide unigram statistics, plus on-demand per-document
    term counts for whichever documents you actually look up (typically:
    whatever candidate_doc_ids the harness gave you for the current
    query -- see submission/feedback.py).

    Built once in prepare() via from_corpus(). collection_term_counts /
    collection_length are computed eagerly (one pass over the whole
    corpus -- this is the legitimate, unavoidable cost of estimating a
    background language model). Per-document term counts are computed
    lazily via doc_term_counts(doc_id) and cached, so a corpus with
    500,000 documents costs no more per-query than however many candidate
    documents you're actually asked to score.
    """

    def __init__(self) -> None:
        self.doc_texts: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.collection_term_counts: Counter = Counter()
        self.collection_length: int = 0
        self.doc_ids: List[str] = []
        self._term_counts_cache: Dict[str, Counter] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        """Called once per document during from_corpus() -- tokenises
        every document exactly once, to build the collection-wide
        background model. Per-document Counters are NOT retained here
        (see doc_term_counts() below); only aggregate collection counts
        and the raw text (needed later if this doc_id turns out to be a
        candidate) are kept."""
        tokens = tokenize(text)
        counts = Counter(tokens)
        self.doc_texts[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        self.collection_term_counts.update(counts)
        self.collection_length += len(tokens)
        self.doc_ids.append(doc_id)

    def doc_term_counts(self, doc_id: str) -> Counter:
        """Term counts for one document, computed on first request and
        cached. Raises KeyError with a clear message for an unknown
        doc_id (e.g. a typo, or a doc_id from the wrong corpus)."""
        if doc_id not in self.doc_texts:
            raise KeyError(f"doc_id {doc_id!r} was not in the corpus passed to prepare()")
        if doc_id not in self._term_counts_cache:
            self._term_counts_cache[doc_id] = Counter(tokenize(self.doc_texts[doc_id]))
        return self._term_counts_cache[doc_id]

    def collection_prob(self, term: str) -> float:
        """P(term | C), the background/collection language model used by
        both smoothing methods in Section 3.1. Returns 0.0 for an
        out-of-vocabulary term -- callers doing log-probability scoring
        must handle that (e.g. by flooring with a small epsilon), since
        log(0) is undefined."""
        if self.collection_length == 0:
            return 0.0
        return self.collection_term_counts.get(term, 0) / self.collection_length

    @classmethod
    def from_corpus(cls, corpus: List[tuple]) -> "CollectionStats":
        stats = cls()
        for doc_id, text in corpus:
            stats.add_document(doc_id, text)
        return stats


def dirichlet_smoothed_log_prob(term_count: int, doc_length: int, p_collection: float, mu: float) -> float:
    """log P(term | D) under Dirichlet-prior smoothing (assignment Section
    3.1): (c(term, D) + mu * P(term | C)) / (|D| + mu). Returns a very
    negative (not literally -inf) number if the numerator is 0, so a
    query-likelihood score built by summing this over query terms stays a
    finite, comparable float instead of collapsing to -inf on any single
    OOV term."""
    import math
    numerator = term_count + mu * p_collection
    denominator = doc_length + mu
    if denominator <= 0 or numerator <= 0:
        return -50.0  # floor, not -inf -- see docstring
    return math.log(numerator / denominator)


def jelinek_mercer_smoothed_log_prob(term_count: int, doc_length: int, p_collection: float, lam: float) -> float:
    """log P(term | D) under Jelinek-Mercer smoothing (assignment Section
    3.1): (1 - lam) * c(term, D)/|D| + lam * P(term | C). Same -50.0 floor
    convention as dirichlet_smoothed_log_prob() above, for the same reason."""
    import math
    doc_ml = term_count / doc_length if doc_length > 0 else 0.0
    prob = (1 - lam) * doc_ml + lam * p_collection
    if prob <= 0:
        return -50.0
    return math.log(prob)
