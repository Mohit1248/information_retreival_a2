#!/usr/bin/env python
"""
Local parameter sweep for submission/feedback.py (not part of grading).

Loads the corpus once, then for every parameter combination in --grid
runs the same pipeline as harness/run_harness.py -- clean seed = the
top-PRF_DEPTH of score_candidates(), then relevance_model_feedback() at
several noise levels using harness/noise_injection.py -- and prints/writes
one row per combination: Track A nDCG@10, clean feedback nDCG@10, nDCG@10
at each noise mode x level (averaged over --noise-seeds independent draws),
and retention ratios (mean noisy nDCG / clean nDCG). Noise modes:
  pool   -- the PUBLIC practice recipe (swap in other docs from the same pool)
  corpus -- swap in uniformly random corpus documents (clearly off-topic)
  other  -- swap in docs that another query's first pass ranked highly
            (plausible-but-wrong topical drift; the hardest of the three)
ret_pool uses levels 0.25/0.5 like the harness; the others use 0.25/0.5/0.75.

Examples:
    python scripts/sweep.py --grid '{"DIRICHLET_MU": [300, 1000, 2500]}'
    python scripts/sweep.py --grid '{"FB_LAMBDA": [0.0, 0.25, 0.5, 0.75, 1.0]}' \\
        --out runs/lambda_sweep.csv
Every key in --grid must be a module-level constant of submission.feedback.
"""
import argparse
import csv
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness import noise_injection  # noqa: E402
from harness.candidates_io import read_candidates  # noqa: E402
from harness.metrics import evaluate_run  # noqa: E402
from harness.trec_io import read_qrels, read_queries  # noqa: E402
from submission import feedback  # noqa: E402

PRF_DEPTH = 10
PRACTICE_LEVELS = (0.25, 0.5)  # what harness/run_harness.py reports
DEFAULT_LEVELS = (0.25, 0.5, 0.75)  # 0.75 is a stress level beyond the practice recipe


def load_task(data_dir, candidates_name):
    queries = read_queries(os.path.join(data_dir, "queries_dev.tsv"))
    qrels = read_qrels(os.path.join(data_dir, "qrels_dev.txt"))
    cands = read_candidates(os.path.join(data_dir, candidates_name))
    pools = {qid: [d for d, _ in cands[qid]] for qid, _ in queries if qid in cands}
    return [(qid, text) for qid, text in queries if qid in pools], qrels, pools


NOISE_MODES = ("pool", "corpus", "other")


def _swap_in_donors(seed, donors, fraction, rng):
    """Replace round(fraction * len(seed)) random seed positions with docs
    drawn from `donors` (never one already in the seed)."""
    n_swap = round(len(seed) * fraction)
    have = set(seed)
    donors = [d for d in donors if d not in have]
    out = list(seed)
    if n_swap == 0 or not donors:
        return out
    positions = rng.sample(range(len(seed)), n_swap)
    for pos, doc in zip(positions, rng.sample(donors, min(n_swap, len(donors)))):
        out[pos] = doc
    return out


def make_noisy_seed(mode, seed, pool, level, rng, corpus_ids, other_pools):
    if mode == "pool":  # the PUBLIC practice recipe (harness/noise_injection.py)
        return noise_injection.perturb_pseudo_relevant_set(seed, pool, level, rng)
    if mode == "corpus":  # off-topic: uniformly random documents from the whole corpus
        return _swap_in_donors(seed, rng.sample(corpus_ids, 200), level, rng)
    if mode == "other":  # plausible-but-wrong: top docs another query's first pass retrieved
        return _swap_in_donors(seed, other_pools, level, rng)
    raise ValueError(mode)


def evaluate_config(queries, qrels, pools, levels=DEFAULT_LEVELS, noise_seeds=3, modes=NOISE_MODES):
    """One full evaluation with whatever constants feedback.py currently has."""
    corpus_ids = feedback._STATS.doc_ids
    ql_run, clean_run = {}, {}
    noisy_runs = {(m, lvl, s): {} for m in modes for lvl in levels for s in range(noise_seeds)}
    for qid, text in queries:
        pool = pools[qid]
        own = set(pool)
        other_pools = sorted({d for q2, p2 in pools.items() if q2 != qid for d in p2[:30]} - own)
        ql_run[qid] = feedback.score_candidates(text, pool, 10)
        seed = [d for d, _ in feedback.score_candidates(text, pool, PRF_DEPTH)]
        clean_run[qid] = feedback.relevance_model_feedback(text, seed, pool, 10)
        for m in modes:
            for lvl in levels:
                for s in range(noise_seeds):
                    rng = random.Random(noise_injection.stable_seed(qid, 1000 * s + int(lvl * 100)))
                    noisy = make_noisy_seed(m, seed, pool, lvl, rng, corpus_ids, other_pools)
                    noisy_runs[(m, lvl, s)][qid] = feedback.relevance_model_feedback(text, noisy, pool, 10)

    ql = evaluate_run(ql_run, qrels)["aggregate"]
    clean = evaluate_run(clean_run, qrels)["aggregate"]["ndcg@10"]
    out = {"ql_ndcg": ql["ndcg@10"], "ql_map": ql["map@10"], "clean_ndcg": clean}
    means = {}
    for m in modes:
        for lvl in levels:
            vals = [evaluate_run(noisy_runs[(m, lvl, s)], qrels)["aggregate"]["ndcg@10"] for s in range(noise_seeds)]
            means[(m, lvl)] = sum(vals) / len(vals)
            out[f"{m}_{int(lvl * 100)}"] = means[(m, lvl)]

    def ret(keys):
        return min(1.0, sum(means[k] for k in keys) / len(keys) / clean) if keys and clean > 0 else float("nan")

    out["ret_pool"] = ret([("pool", l) for l in PRACTICE_LEVELS if ("pool", l) in means])
    out["ret_corpus"] = ret([("corpus", l) for l in levels if ("corpus", l) in means])
    out["ret_other"] = ret([("other", l) for l in levels if ("other", l) in means])
    out["ret_all"] = ret(list(means))
    return out


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "full")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=root)
    parser.add_argument("--candidates", default="candidates_local.jsonl")
    parser.add_argument("--grid", default="{}", help="JSON: {constant_name: [values...]}")
    parser.add_argument("--noise-seeds", type=int, default=3)
    parser.add_argument("--out", default=None, help="CSV path")
    args = parser.parse_args()

    grid = json.loads(args.grid)
    for name in grid:
        if not hasattr(feedback, name):
            raise SystemExit(f"feedback.py has no constant named {name!r}")
    queries, qrels, pools = load_task(args.data_dir, args.candidates)
    t0 = time.perf_counter()
    feedback.prepare(os.path.join(args.data_dir, "corpus.jsonl"))
    print(f"prepare(): {time.perf_counter() - t0:.1f}s, {len(queries)} queries", flush=True)

    names = list(grid)
    defaults = {n: getattr(feedback, n) for n in names}
    rows = []
    for combo in itertools.product(*(grid[n] for n in names)) if names else [()]:
        for n, v in zip(names, combo):
            setattr(feedback, n, v)
        t1 = time.perf_counter()
        res = evaluate_config(queries, qrels, pools, noise_seeds=args.noise_seeds)
        row = {**dict(zip(names, combo)), **res}
        rows.append(row)
        print(
            f"{dict(zip(names, combo))}  QL={res['ql_ndcg']:.4f}  clean={res['clean_ndcg']:.4f}  "
            f"ret[pool/corpus/other/all]={res['ret_pool']:.3f}/{res['ret_corpus']:.3f}/"
            f"{res['ret_other']:.3f}/{res['ret_all']:.3f}  "
            f"worst(other_50)={res.get('other_50', float('nan')):.4f}  ({time.perf_counter() - t1:.1f}s)",
            flush=True,
        )
    for n, v in defaults.items():
        setattr(feedback, n, v)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
