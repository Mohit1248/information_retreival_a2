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

Implementation summary
    score_candidates(): Dirichlet-smoothed unigram query likelihood over
    the candidate pool.

    relevance_model_feedback(): estimates P(w|R) from the seed documents
    (RM1, or RM2), optionally interpolates it with the original query
    model (RM3), and reranks the candidate pool by negative cross-entropy
    between that term model and each candidate's smoothed LM. Seed
    documents are weighted by their likelihood under the ORIGINAL query,
    and expansion terms must be supported by more than one seed document;
    both limit how far a few off-topic seed documents can pull the model.
    The seed argument is used exactly as passed -- nothing is cached from
    score_candidates().
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from submission.corpus_utils import load_corpus
from submission.lm_utils import CollectionStats, dirichlet_smoothed_log_prob, tokenize_query

# ---------------------------------------------------------------------------
# Tunable parameters. Every one of these is a real design choice; the
# values below are the ones picked from the sweeps in scripts/sweep.py.
# ---------------------------------------------------------------------------
# Dirichlet smoothing for the reranking score (Section 3.1). Larger mu
# smooths more aggressively toward the collection model.
DIRICHLET_MU = 500.0
# Dirichlet mu used inside the relevance-model estimate, P(w|D).
RM_DOC_MU = 1500.0

# Which relevance model relevance_model_feedback() uses: "rm1", "rm2", "rm3".
RM_VARIANT = "rm3"
# RM3 interpolation: P_RM3(w) = FB_LAMBDA * P(w|Q) + (1 - FB_LAMBDA) * P(w|R).
# FB_LAMBDA = 1 ignores feedback, 0 trusts the relevance model completely.
FB_LAMBDA = 0.35
# Expansion terms kept from P(w|R) (top-N by probability, renormalised).
FB_TERMS = 15
# At most this many seed documents (best under the ORIGINAL query's
# likelihood) contribute to the relevance model.
FB_DOCS = 10
# P(D|Q) is proportional to exp(POSTERIOR_SCALE * loglik(Q|D) / |Q|); a
# smaller scale flattens the document weights, a larger one concentrates
# them on the best-matching seed documents.
POSTERIOR_SCALE = 10.0
# A candidate expansion term must occur in at least this many seed docs
# (capped by the number of seed docs actually used).
MIN_TERM_SUPPORT = 2

_STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been before being "
    "below between both but by can could did do does doing down during each few for from further "
    "had has have having he her here hers herself him himself his how i if in into is it its "
    "itself just me more most my myself no nor not now of off on once only or other our ours "
    "ourselves out over own same she should so some such than that the their theirs them "
    "themselves then there these they this those through to too under until up very was we were "
    "what when where which while who whom why will with would you your yours yourself "
    "yourselves also may might must shall us via within without et al".split()
)

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

    The seed is used exactly as given: it is never replaced by anything
    this module computed earlier, and nothing here assumes it came from
    score_candidates(). Seed documents are weighted only by how well they
    match the ORIGINAL query (P(D|Q)), which is how RM1 is defined.
    """
    if _STATS is None:
        raise RuntimeError(
            "relevance_model_feedback() called before prepare(); see "
            "score_candidates()'s error for the same reason."
        )
    stats = _STATS
    query_terms = [t for t in tokenize_query(query) if stats.collection_prob(t) > 0.0]
    if not query_terms or not candidate_doc_ids:
        return _ql_rerank(query, candidate_doc_ids, k, stats)

    fb_docs = _select_feedback_docs(query_terms, pseudo_relevant_doc_ids, stats)
    if not fb_docs:
        return _ql_rerank(query, candidate_doc_ids, k, stats)
    weights = query_posterior_weights(query_terms, fb_docs, stats, DIRICHLET_MU, POSTERIOR_SCALE)

    variant = RM_VARIANT.lower()
    if variant == "rm2":
        rm = estimate_rm2(query_terms, fb_docs, weights, stats, RM_DOC_MU, MIN_TERM_SUPPORT)
    else:
        rm = estimate_rm1(fb_docs, weights, stats, RM_DOC_MU, MIN_TERM_SUPPORT)

    if variant == "rm3":
        model = interpolate_rm3(query_terms, rm, FB_LAMBDA, FB_TERMS)
    else:
        model = truncate_model(rm, FB_TERMS)
    if not model:
        return _ql_rerank(query, candidate_doc_ids, k, stats)
    return rank_by_model(model, candidate_doc_ids, stats, DIRICHLET_MU, k)


# ---------------------------------------------------------------------------
# Language-model building blocks. All of these are pure functions of their
# arguments (plus CollectionStats), so each can be checked by hand on a
# tiny corpus -- see tests/test_relevance_models.py.
# ---------------------------------------------------------------------------
def _ql_rerank(query: str, doc_ids: List[str], k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    query_terms = tokenize_query(query)
    if not query_terms:
        return []
    doc_ids = list(dict.fromkeys(doc_ids))
    scores = {d: query_log_likelihood(query_terms, d, stats, DIRICHLET_MU) for d in doc_ids}
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def query_log_likelihood(query_terms: Sequence[str], doc_id: str, stats: CollectionStats, mu: float) -> float:
    """log P(Q|D) = sum over query terms of the Dirichlet-smoothed log P(w|D)."""
    doc_length = stats.doc_lengths.get(doc_id, 0)
    term_counts = stats.doc_term_counts(doc_id) if doc_id in stats.doc_texts else {}
    total = 0.0
    for term in query_terms:
        total += dirichlet_smoothed_log_prob(
            term_counts.get(term, 0), doc_length, stats.collection_prob(term), mu
        )
    return total


def doc_term_prob(term: str, doc_id: str, stats: CollectionStats, mu: float) -> float:
    """Dirichlet-smoothed P(term|D) = (c(term,D) + mu*P(term|C)) / (|D| + mu)."""
    doc_length = stats.doc_lengths.get(doc_id, 0)
    count = stats.doc_term_counts(doc_id).get(term, 0)
    return (count + mu * stats.collection_prob(term)) / (doc_length + mu)


def _select_feedback_docs(query_terms: Sequence[str], seed_ids: Sequence[str], stats: CollectionStats) -> List[str]:
    """Deduplicate the seed, drop ids that are not in the corpus, and keep
    the FB_DOCS seed documents that best match the original query."""
    seen = [d for d in dict.fromkeys(seed_ids) if d in stats.doc_texts]
    if len(seen) <= FB_DOCS:
        return seen
    scored = sorted(
        seen, key=lambda d: query_log_likelihood(query_terms, d, stats, DIRICHLET_MU), reverse=True
    )
    return scored[:FB_DOCS]


def query_posterior_weights(
    query_terms: Sequence[str], fb_docs: Sequence[str], stats: CollectionStats, mu: float, scale: float = 1.0
) -> Dict[str, float]:
    """P(D|Q) over the feedback documents, uniform document prior:
    proportional to exp(scale * log P(Q|D) / |Q|). With scale = |Q| this is
    exactly P(Q|D) normalised over the set (the RM1 definition); smaller
    scales flatten it. Returns weights summing to 1."""
    n = max(len(query_terms), 1)
    logs = {d: scale * query_log_likelihood(query_terms, d, stats, mu) / n for d in fb_docs}
    top = max(logs.values())
    exps = {d: math.exp(v - top) for d, v in logs.items()}
    z = sum(exps.values())
    return {d: v / z for d, v in exps.items()}


def _candidate_vocabulary(fb_docs: Sequence[str], stats: CollectionStats, min_support: int) -> List[str]:
    """Terms of the feedback docs that may enter the relevance model:
    not a stopword, not a single character, and present in at least
    min_support feedback documents (a term backed by one document only is
    the easiest way for a single off-topic document to steer the model)."""
    support: Counter = Counter()
    for d in fb_docs:
        support.update(stats.doc_term_counts(d).keys())
    need = min(min_support, len(fb_docs))
    return [
        t for t, c in support.items()
        if c >= need and len(t) > 1 and t not in _STOPWORDS
    ]


def estimate_rm1(
    fb_docs: Sequence[str], doc_weights: Dict[str, float], stats: CollectionStats, mu: float, min_support: int = 1
) -> Dict[str, float]:
    """RM1 (Lavrenko & Croft): P(w|R) = sum_D P(w|D) P(D|Q), normalised over
    the candidate vocabulary. `doc_weights` is P(D|Q)."""
    vocab = _candidate_vocabulary(fb_docs, stats, min_support)
    scores = {
        w: sum(doc_weights[d] * doc_term_prob(w, d, stats, mu) for d in fb_docs) for w in vocab
    }
    return _normalise(scores)


def estimate_rm2(
    query_terms: Sequence[str],
    fb_docs: Sequence[str],
    doc_weights: Dict[str, float],
    stats: CollectionStats,
    mu: float,
    min_support: int = 1,
) -> Dict[str, float]:
    """RM2 (Lavrenko & Croft): query terms are sampled independently
    *given the word w*, rather than given the document:
        P(w, q_1..q_k) = P(w) * prod_i sum_D P(q_i|D) P(D|w),
        P(D|w) = P(w|D) P(D) / P(w),   P(w) = sum_D P(w|D) P(D).
    `doc_weights` is the document prior P(D) (pass uniform weights for the
    textbook form). Result is P(w|R) = P(w, Q) normalised over the
    candidate vocabulary. Computed in log space."""
    vocab = _candidate_vocabulary(fb_docs, stats, min_support)
    p_q_given_d = {d: [doc_term_prob(q, d, stats, mu) for q in query_terms] for d in fb_docs}
    log_scores: Dict[str, float] = {}
    for w in vocab:
        p_w_given_d = {d: doc_term_prob(w, d, stats, mu) for d in fb_docs}
        p_w = sum(doc_weights[d] * p_w_given_d[d] for d in fb_docs)
        total = math.log(p_w)
        for i in range(len(query_terms)):
            inner = sum(p_q_given_d[d][i] * p_w_given_d[d] * doc_weights[d] / p_w for d in fb_docs)
            if inner <= 0.0:  # only reachable with unsmoothed (mu = 0) estimates
                total = float("-inf")
                break
            total += math.log(inner)
        log_scores[w] = total
    finite = [v for v in log_scores.values() if v != float("-inf")]
    if not finite:
        return {}
    top = max(finite)
    return _normalise({w: math.exp(v - top) for w, v in log_scores.items() if v != float("-inf")})


def truncate_model(model: Dict[str, float], n_terms: int) -> Dict[str, float]:
    """Keep the n_terms highest-probability terms and renormalise."""
    top = sorted(model.items(), key=lambda kv: (-kv[1], kv[0]))[:n_terms]
    return _normalise(dict(top))


def interpolate_rm3(
    query_terms: Sequence[str], relevance_model: Dict[str, float], lam: float, n_terms: int
) -> Dict[str, float]:
    """RM3: P_RM3(w) = lam * P(w|Q) + (1 - lam) * P(w|R), where P(w|Q) is
    the maximum-likelihood query model and P(w|R) is the relevance model
    truncated to its n_terms best terms."""
    rm = truncate_model(relevance_model, n_terms)
    query_counts = Counter(query_terms)
    q_len = sum(query_counts.values())
    model: Dict[str, float] = {}
    for w, c in query_counts.items():
        model[w] = lam * c / q_len
    for w, p in rm.items():
        model[w] = model.get(w, 0.0) + (1.0 - lam) * p
    return model


def rank_by_model(
    model: Dict[str, float], doc_ids: Sequence[str], stats: CollectionStats, mu: float, k: int
) -> List[Tuple[str, float]]:
    """Rank documents by negative cross-entropy between the term model and
    each document's smoothed LM: score(D) = sum_w model(w) log P(w|D). This
    is rank-equivalent to KL(model || P(.|D)); higher is better."""
    scores: Dict[str, float] = {}
    for d in dict.fromkeys(doc_ids):
        if d not in stats.doc_texts:
            scores[d] = float("-inf")
            continue
        scores[d] = sum(p * math.log(doc_term_prob(w, d, stats, mu)) for w, p in model.items())
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def _normalise(scores: Dict[str, float]) -> Dict[str, float]:
    z = sum(scores.values())
    if z <= 0:
        return {}
    return {w: v / z for w, v in scores.items()}
