"""Accuracy of fGC on the CEfinal benchmark as a function of the aggregation.

Reproduces Section 5, Table 6 and Figure 3(b): total accuracy of fGC on the 112
categorical pairs, on the raw categories and after frequency aggregation to K
categories.  K = 3 gives 60.71%.

    python causality/run_cefinal_sweep.py
    python causality/run_cefinal_sweep.py --k 0 2 3 5

Decision rule: causality(a, b) returns (cost A->B, cost B->A), so "A causes B" is
(b - a) > 0, and the benchmark target 1 means A -> B.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fgc  # noqa: E402

fgc.add_gc_module_path()
from GeometricComplexity import causality as fgc_causality  # noqa: E402

DEFAULT_K = (0, 2, 3, 4, 5, 6, 8, 10, 15, 20)   # 0 means the raw categories
PAPER = {0: 53.57, 2: 58.93, 3: 60.71, 4: 57.14, 5: 53.57, 6: 54.46,
         8: 54.46, 10: 55.36, 15: 53.57, 20: 53.57}


def sweep(ks=DEFAULT_K):
    pairs = fgc.load_cefinal()
    print(f"{len(pairs)} categorical pairs")
    out = {}
    for K in ks:
        hits = []
        for a_vals, b_vals, tgt in pairs:
            x = np.asarray(a_vals) if K == 0 else fgc.aggregate_to_k(a_vals, K)
            y = np.asarray(b_vals) if K == 0 else fgc.aggregate_to_k(b_vals, K)
            a, b = fgc_causality(x, y)
            hits.append((b - a > 0) == (tgt == 1))
        acc = float(np.mean(hits) * 100)
        out[K] = acc
        label = "raw" if K == 0 else str(K)
        tag = ""
        if K in PAPER:
            tag = "  ok" if abs(acc - PAPER[K]) < 0.05 else f"  paper {PAPER[K]:.2f}"
        print(f"  K={label:>4s}   {acc:6.2f}%{tag}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = sweep(tuple(args.k) if args.k else DEFAULT_K)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", args.json)
    return 0 if abs(out.get(3, 0) - 60.71) < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
