#!/usr/bin/env python
"""
Plot the report figure (assignment Section 6, item 1) from a CSV written by
scripts/sweep.py:  nDCG@10 vs lambda at 0% noise and at the highest public
practice noise level (50%), plus the harsher stress conditions.

    python scripts/plot_report.py runs/report_lambda_sweep.csv runs/lambda_plot.png
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/report_lambda_sweep.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "runs/lambda_plot.png"
    with open(src, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["FB_LAMBDA"]))
    lam = [float(r["FB_LAMBDA"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(lam, col("clean_ndcg"), "o-", label="0% noise (clean seed)")
    ax.plot(lam, col("pool_50"), "s-", label="50% noise, public practice recipe")
    ax.plot(lam, col("other_50"), "^--", label="50% noise, other-topic docs (stress)")
    ax.plot(lam, col("corpus_50"), "v--", label="50% noise, random corpus docs (stress)")
    ax.axhline(float(rows[0]["ql_ndcg"]), color="gray", ls=":", label="plain query likelihood (no feedback)")
    ax.set_xlabel("RM3 lambda  (weight on original query; 1 = no feedback)")
    ax.set_ylabel("nDCG@10 (dev set)")
    ax.set_title("nDCG@10 vs lambda")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(lam, col("ret_pool"), "s-", label="retention, practice recipe (0.25/0.5)")
    ax.plot(lam, col("ret_other"), "^--", label="retention, other-topic noise")
    ax.plot(lam, col("ret_corpus"), "v--", label="retention, random-corpus noise")
    ax.set_xlabel("RM3 lambda")
    ax.set_ylabel("mean noisy nDCG@10 / clean nDCG@10")
    ax.set_title("Drift retention vs lambda")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
