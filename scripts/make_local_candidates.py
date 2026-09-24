#!/usr/bin/env python
"""
LOCAL TUNING AID ONLY -- not part of the submission.

Course staff release data/full/candidates_dev.jsonl separately (it comes
from an undisclosed first-pass retriever we cannot regenerate). Until it
is available, this script builds a stand-in top-K pool per dev query with
a plain BM25 first-pass ranker over data/full/corpus.jsonl, written to
data/full/candidates_local.jsonl in the same format, so the local sweep in
scripts/sweep.py has realistic (imperfect-recall) pools to rerank.

The staff generator is different, so treat any tuning done against this
file as provisional and re-check against the real pool once released.

Usage:
    python scripts/make_local_candidates.py [--k 100]
"""
import argparse
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.candidates_io import write_candidates  # noqa: E402
from harness.trec_io import read_queries  # noqa: E402
from submission.corpus_utils import load_corpus  # noqa: E402
from submission.lm_utils import tokenize  # noqa: E402

K1, B = 1.2, 0.75


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "full")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(root, "corpus.jsonl"))
    parser.add_argument("--queries", default=os.path.join(root, "queries_dev.tsv"))
    parser.add_argument("--out", default=os.path.join(root, "candidates_local.jsonl"))
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    queries = read_queries(args.queries)
    query_terms = {qid: sorted(set(tokenize(text))) for qid, text in queries}
    wanted = {t for terms in query_terms.values() for t in terms}

    doc_ids, lengths = [], []
    postings = defaultdict(list)  # term -> [(doc_index, tf)], only for query terms
    for idx, (doc_id, text) in enumerate(load_corpus(args.corpus)):
        tokens = tokenize(text)
        doc_ids.append(doc_id)
        lengths.append(len(tokens))
        tf = defaultdict(int)
        for t in tokens:
            if t in wanted:
                tf[t] += 1
        for t, c in tf.items():
            postings[t].append((idx, c))

    n_docs = len(doc_ids)
    avg_len = sum(lengths) / n_docs
    candidates = {}
    for qid, terms in query_terms.items():
        scores = defaultdict(float)
        for t in terms:
            plist = postings.get(t, [])
            df = len(plist)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, c in plist:
                norm = c + K1 * (1 - B + B * lengths[idx] / avg_len)
                scores[idx] += idf * c * (K1 + 1) / norm
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: args.k]
        candidates[qid] = [(doc_ids[i], s) for i, s in top]

    write_candidates(args.out, candidates)
    print(f"Wrote {len(candidates)} candidate pools (top-{args.k}) to {args.out}")


if __name__ == "__main__":
    main()
