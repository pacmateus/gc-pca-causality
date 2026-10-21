"""Accuracy of fGC on the Tuebingen CEP benchmark as a function of the level.

Reproduces Section 5, Table 4 (the fGC rows) and Figure 3(a): weighted accuracy
of fGC after linear quantisation to L levels.  L = 3 gives 77.81%.

    python causality/run_cep_sweep.py
    python causality/run_cep_sweep.py --levels 2 3 4

The decision rule: geometric_complexity.causality(x, y) returns the codelengths
(cost X->Y, cost Y->X), so "x causes y" is (b - a) > 0.  Pairs are loaded
cause-first, so a correct decision is exactly (b - a) > 0.
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

DEFAULT_LEVELS = (2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 256)
PAPER = {2: 61.57, 3: 77.81, 4: 66.78, 5: 63.79, 6: 69.07, 8: 66.93,
         12: 66.93, 16: 58.42, 32: 52.84, 64: 52.41, 256: 49.39}


def sweep(levels=DEFAULT_LEVELS):
    pairs = fgc.load_tuebingen()
    w = np.array([p["w"] for p in pairs])
    print(f"{len(pairs)} eligible pairs, total weight {w.sum():.3f}")
    out = {}
    for L in levels:
        hits = []
        for p in pairs:
            a, b = fgc_causality(fgc.qlin(p["x"], L), fgc.qlin(p["y"], L))
            hits.append(b - a > 0)
        acc = float(np.sum(w * np.array(hits)) / w.sum() * 100)
        out[L] = acc
        tag = ""
        if L in PAPER:
            tag = "  ok" if abs(acc - PAPER[L]) < 0.05 else f"  paper {PAPER[L]:.2f}"
        print(f"  L={L:4d}   {acc:6.2f}%{tag}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", type=int, default=None)
    ap.add_argument("--json", default=None, help="write the numbers to this file")
    args = ap.parse_args()
    out = sweep(tuple(args.levels) if args.levels else DEFAULT_LEVELS)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", args.json)
    return 0 if abs(out.get(3, 0) - 77.81) < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
