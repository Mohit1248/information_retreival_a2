#!/usr/bin/env python
"""
Build submission/adversarial_set.json for the OPTIONAL bonus track (dev
tooling, not part of grading).

In the round-robin, each other team's relevance_model_feedback() gets, for
each query in this file, OUR doc-id list as its pseudo-relevant seed
(reranking the same official candidate pool it would normally see), and we
score by how far their nDCG@10 drops versus their own clean run.

Strategy: pick seed docs that are
  (a) JUDGED non-relevant to the query (qrels == 0), so they are reliably
      wrong,
  (b) lexically close to the query (high query likelihood), so a defender
      that weights seed docs by query match still trusts them, and
  (c) chosen greedily, one at a time, for maximum damage.
We cannot run other teams' code, so damage is measured against several
deliberately different defender configurations of our own RM machinery --
from a naive uniform-weight RM3 up to our robust default -- and only the
queries with the largest average drop across all of them are kept (an
attack that only fools one configuration is not kept).

    python scripts/build_adversarial.py --n-queries 8 --out runs/adversarial_set.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import sweep  # noqa: E402  (scripts/sweep.py: load_task)
from harness.metrics import ndcg_at_k  # noqa: E402
from submission import feedback  # noqa: E402
from submission.lm_utils import tokenize_query  # noqa: E402

SEED_LEN = 10
# Defender styles: (name, overrides of feedback.py constants). "robust" is
# our own submission as shipped (empty override).
DEFENDERS = [
    ("robust (ours)", {}),
    ("naive rm3", {"POSTERIOR_SCALE": 0.0, "MIN_TERM_SUPPORT": 1, "FB_LAMBDA": 0.5, "FB_TERMS": 30}),
    ("aggressive rm3", {"POSTERIOR_SCALE": 1.0, "MIN_TERM_SUPPORT": 1, "FB_LAMBDA": 0.2, "FB_TERMS": 50}),
    ("pure rm1", {"RM_VARIANT": "rm1", "POSTERIOR_SCALE": 0.0, "MIN_TERM_SUPPORT": 1, "FB_TERMS": 50}),
]


def _with(overrides):
    saved = {k: getattr(feedback, k) for k in overrides}
    for k, v in overrides.items():
        setattr(feedback, k, v)
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(feedback, k, v)


class QueryAttack:
    def __init__(self, qid, text, pool, qrels_q):
        self.qid, self.text, self.pool, self.qrels_q = qid, text, pool, qrels_q
        clean_seed = [d for d, _ in feedback.score_candidates(text, pool, 10)]
        self.clean = {}
        for name, ov in DEFENDERS:
            saved = _with(ov)
            run = feedback.relevance_model_feedback(text, clean_seed, pool, 10)
            _restore(saved)
            self.clean[name] = ndcg_at_k([d for d, _ in run], qrels_q, 10)

    def drops(self, seed):
        out = {}
        for name, ov in DEFENDERS:
            saved = _with(ov)
            run = feedback.relevance_model_feedback(self.text, seed, self.pool, 10)
            _restore(saved)
            out[name] = self.clean[name] - ndcg_at_k([d for d, _ in run], self.qrels_q, 10)
        return out

    def mean_drop(self, seed):
        d = self.drops(seed)
        return sum(d.values()) / len(d)

    def candidates(self, n):
        """Judged non-relevant docs (qrels == 0) with the highest query
        likelihood: wrong, but they look right to a query-weighted model."""
        stats = feedback._STATS
        q_terms = tokenize_query(self.text)
        bad = [d for d, r in self.qrels_q.items() if r == 0 and d in stats.doc_texts]
        bad.sort(key=lambda d: feedback.query_log_likelihood(q_terms, d, stats, feedback.DIRICHLET_MU), reverse=True)
        return bad[:n]

    def greedy_seed(self, cand_n):
        cands = self.candidates(cand_n)
        seed = []
        for _ in range(SEED_LEN):
            best, best_drop = None, -9.0
            for d in cands:
                if d in seed:
                    continue
                drop = self.mean_drop(seed + [d])
                if drop > best_drop:
                    best, best_drop = d, drop
            if best is None:
                break
            seed.append(best)
        return seed


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "full")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=root)
    ap.add_argument("--candidates", default="candidates_dev.jsonl")
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--screen-top", type=int, default=16, help="queries that get the expensive greedy search")
    ap.add_argument("--cand-n", type=int, default=60, help="candidate docs considered per greedy step")
    ap.add_argument("--out", default="runs/adversarial_set.json")
    args = ap.parse_args()

    queries, qrels, pools = sweep.load_task(args.data_dir, args.candidates)
    feedback.prepare(os.path.join(args.data_dir, "corpus.jsonl"))
    t0 = time.perf_counter()

    attacks = {qid: QueryAttack(qid, text, pools[qid], qrels[qid]) for qid, text in queries if qid in qrels}
    # cheap screen: the plain top-10 "wrong but on-topic-sounding" docs
    screen = {}
    for qid, atk in attacks.items():
        screen[qid] = atk.mean_drop(atk.candidates(SEED_LEN))
    order = sorted(screen, key=screen.get, reverse=True)[: args.screen_top]
    print("screen (mean drop, top queries):", {q: round(screen[q], 3) for q in order}, flush=True)

    results = {}
    for qid in order:
        seed = attacks[qid].greedy_seed(args.cand_n)
        per = attacks[qid].drops(seed)
        results[qid] = (seed, per)
        print(f"q{qid}: mean drop {sum(per.values()) / len(per):.3f}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in per.items()) +
              f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

    keep = sorted(results, key=lambda q: min(results[q][1].values()) + sum(results[q][1].values()) / len(results[q][1]),
                  reverse=True)[: args.n_queries]
    out = {qid: results[qid][0] for qid in sorted(keep, key=int)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    avg = [sum(results[q][1].values()) / len(DEFENDERS) for q in keep]
    print(f"kept {len(out)} queries {sorted(out, key=int)}, mean drop across defenders: {sum(avg) / len(avg):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
