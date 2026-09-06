"""
submission/feedback.py -- THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5,
"Submission Interface & Conformance Checking") -- do not rename them,
change their signatures, or move them out of this file.

    prepare(corpus_path: str) -> None
        Called once, before anything else. Load the corpus and build
        whatever collection-wide statistics your unigram LM and
        relevance model need (see submission/lm_utils.py::CollectionStats
        for why this is a cheap, one-time pass and does not require
        building an index). No build/load process split this time --
        grading is in-process, so it is fine to keep everything in
        module-level state here and read it in the two functions below.

    score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]
        Rerank the PROVIDED candidate pool -- typically the top-100 from
        an undisclosed reference retriever (assignment Section 6) -- using
        your own unigram query-likelihood LM (Section 3.1). You are never
        asked to rank the whole corpus; candidate_doc_ids is the entire
        universe of documents you need to consider for this call. Return
        up to k (doc_id, score) pairs, sorted by score descending, drawn
        ONLY from candidate_doc_ids. This is graded directly as Track A.

    relevance_model_feedback(query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]
        RM1/RM2/RM3-based reranking (Section 3.2). Two DIFFERENT lists
        come in, doing two different jobs -- do not confuse them:
          - pseudo_relevant_doc_ids: the (possibly noise-perturbed) seed
            set used to ESTIMATE the relevance model. Supplied BY THE
            HARNESS -- sometimes exactly your own score_candidates()
            top-k', sometimes a version of that with a fraction swapped
            for off-topic documents (query drift stress-testing;
            assignment Section 7, Track C). You have no legitimate way to
            tell which, and must not try to detect it -- see
            docs/SUBMISSION_INTERFACE.md, "call-order independence".
          - candidate_doc_ids: the pool to RERANK using your estimated
            model -- typically the SAME candidate pool passed to
            score_candidates() for this query, unperturbed. Your
            returned doc_ids must come from this list, not from
            pseudo_relevant_doc_ids and not from outside either list.
        Return up to k (doc_id, score) pairs, sorted by score descending.

This file ships with a trivial, fully-working baseline: score_candidates()
does real Dirichlet-smoothed query-likelihood reranking of whatever
candidate pool it's given (so it is not literally a no-op), and
relevance_model_feedback() ignores pseudo_relevant_doc_ids entirely and
just reranks candidate_doc_ids the same way score_candidates() would. It
exercises the full interface correctly end-to-end from your first commit,
including the harness's noise-injection stress test (it will score
identically at every noise level, since it never looks at its seed input
-- which is itself a legitimate, if unambitious, point on the Track C
"retention" axis: you cannot drift if you never expand). Replace the
feedback logic; keep the same function shapes.
"""
import os
from typing import Dict, List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.lm_utils import CollectionStats, dirichlet_smoothed_log_prob, tokenize

# TODO(you): tune this. mu is the Dirichlet smoothing parameter (Section
# 3.1) -- larger values smooth more aggressively toward the collection
# model. There is no single "correct" value; it is a real design choice.
DIRICHLET_MU = 1500.0

# ---------------------------------------------------------------------------
# Module-level state. prepare() populates this; both retrieval functions
# read it. No build/load process split this assignment (see module
# docstring) -- unlike Assignment 1, it is fine for this to just live in
# memory for the lifetime of the harness process.
# ---------------------------------------------------------------------------
_STATS: Optional[CollectionStats] = None


def prepare(corpus_path: str) -> None:
    """Load the corpus and build collection-wide statistics. Called once,
    before any score_candidates()/relevance_model_feedback() calls."""
    global _STATS
    corpus = load_corpus(corpus_path)
    _STATS = CollectionStats.from_corpus(corpus)


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs from `candidate_doc_ids`,
    best first, under a Dirichlet-smoothed unigram query-likelihood model
    (Section 3.1)."""
    if _STATS is None:
        raise RuntimeError(
            "score_candidates() called before prepare(); the harness "
            "always calls prepare(corpus_path) before any retrieval "
            "calls. If you're testing manually, do the same."
        )
    return _ql_rerank(query, candidate_doc_ids, k, _STATS)


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs from `candidate_doc_ids`,
    best first, using a relevance model estimated from
    `pseudo_relevant_doc_ids` (Section 3.2). See the module docstring --
    these are two different lists doing two different jobs.
    """
    if _STATS is None:
        raise RuntimeError(
            "relevance_model_feedback() called before prepare(); see "
            "score_candidates()'s error for the same reason."
        )

    # TODO(you): replace this with real RM1 -> RM3 feedback, e.g.:
    #
    #   relevance_model = estimate_relevance_model(query, pseudo_relevant_doc_ids, _STATS)
    #   interpolated = interpolate_with_query_model(relevance_model, query, lam=0.5)
    #   return rank_by_relevance_model(interpolated, candidate_doc_ids, _STATS, k)
    #
    # The trivial baseline below ignores pseudo_relevant_doc_ids entirely
    # and just reranks candidate_doc_ids with plain query-likelihood.
    return _ql_rerank(query, candidate_doc_ids, k, _STATS)


# ---------------------------------------------------------------------------
# Trivial reference baseline internals -- DO NOT submit this as your final
# entry. A real Dirichlet-smoothed QL reranker (not a stub), so it is a
# legitimate Track A entry on its own; only the feedback layer is a no-op.
# ---------------------------------------------------------------------------
def _ql_rerank(query: str, doc_ids: List[str], k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    scores: Dict[str, float] = {}
    for doc_id in doc_ids:
        doc_length = stats.doc_lengths.get(doc_id, 0)
        term_counts = stats.doc_term_counts(doc_id)
        log_prob = 0.0
        for term in query_terms:
            p_collection = stats.collection_prob(term)
            log_prob += dirichlet_smoothed_log_prob(
                term_counts.get(term, 0), doc_length, p_collection, DIRICHLET_MU
            )
        scores[doc_id] = log_prob

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]
