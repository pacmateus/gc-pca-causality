#!/usr/bin/env python3
"""
The levelled-accuracy figure: how the discretisation level moves the accuracy.

(a) T\"ubingen CEP, continuous pairs: fGC after linear quantisation to L levels.
(b) CEfinal, nominal pairs: fGC after frequency aggregation to K categories.

Both curves are recomputed here from scratch (same harness as the paper:
fGC = Geometric Complexity of the multinomial mechanism, forced decision,
T\"ubingen weighted by the official pairmeta weights).

Run with /opt/homebrew/bin/python3.10.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "causality")))

import fgc                                                  # noqa: E402

fgc.add_gc_module_path()
from GeometricComplexity import causality as fgc_causality  # noqa: E402
from fgc import load_tuebingen, load_cefinal                 # noqa: E402
C_ORANGE, C_GREEN, C_DARK, C_BLUE = "#e67e22", "#27ae60", "#2c3e50", "#3498db"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def qlin(x, levels):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros(len(x), np.uint8)
    return np.clip(((x - lo) / (hi - lo) * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)


def aggregate_to_k(values, k):
    """Frequency aggregation: keep the k-1 most frequent labels, merge the rest."""
    most = [v for v, _ in Counter(values).most_common(k - 1)]
    mapping = {v: i for i, v in enumerate(most)}
    return np.array([mapping.get(v, k - 1) for v in values], np.uint8)


def tuebingen_curve():
    pairs = load_tuebingen()
    ws = np.array([p["w"] for p in pairs])
    out = {}
    for L in (2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 256):
        hit = []
        for p in pairs:
            a, b = fgc_causality(qlin(p["x"], L), qlin(p["y"], L))
            hit.append((b - a > 0))
        out[L] = float(np.sum(ws * np.array(hit)) / ws.sum() * 100)
        print(f"  L={L:4d}  {out[L]:6.2f}%", flush=True)
    return out


def cefinal_curve():
    pairs = load_cefinal()
    print(f"  {len(pairs)} CEfinal pairs", flush=True)
    out = {}
    for K in (0, 2, 3, 4, 5, 6, 8, 10, 15, 20):
        hit = []
        for x, y, tgt in pairs:
            xx = x if K == 0 else aggregate_to_k(x, K)
            yy = y if K == 0 else aggregate_to_k(y, K)
            a, b = fgc_causality(np.asarray(xx), np.asarray(yy))
            # causality() returns (cost A->B, cost B->A); A is the cause iff the
            # first is smaller, and the benchmark target 1 means A->B.
            hit.append((b - a > 0) == (tgt == 1))
        out[K] = float(np.mean(hit) * 100)
        print(f"  K={'raw' if K == 0 else K:>4}  {out[K]:6.2f}%", flush=True)
    return out


def main():
    print("Tubingen L sweep (weighted, %):")
    tub = tuebingen_curve()
    print("CEfinal K sweep (total accuracy, %):")
    cef = cefinal_curve()

    with open(os.path.join(HERE, "levels_sweep_results.json"), "w") as fh:
        json.dump(dict(tuebingen=tub, cefinal=cef), fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))

    ax = axes[0]
    L = list(tub.keys()); v = [tub[k] for k in L]
    ax.semilogx(L, v, "-o", color=C_BLUE, ms=3, lw=1.2)
    ax.axvline(3, color=C_GREEN, ls="--", lw=1.0)
    best = max(tub, key=tub.get)
    ax.annotate(f"$L={best}$: {tub[best]:.1f}\\%", xy=(best, tub[best]),
                xytext=(best * 1.35, tub[best] - 5.5),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_DARK), fontsize=7.5)
    ax.set_xlabel("quantisation levels $L$")
    ax.set_ylabel("weighted accuracy (\\%)")
    ax.set_title("(a) T\u00fcbingen CEP, fGC")
    ax.set_ylim(min(v) - 3, max(v) + 3)

    ax = axes[1]
    K = [k for k in cef if k > 0]; v = [cef[k] for k in K]
    ax.plot([0], [cef[0]], "s", color=C_DARK, ms=4, label="raw categories")
    ax.plot(K, v, "-o", color=C_ORANGE, ms=3, lw=1.2, label="frequency aggregation")
    ax.axvline(3, color=C_GREEN, ls="--", lw=1.0, label="$K=3$")
    ax.set_xlabel("categories $K$ after aggregation")
    ax.set_ylabel("total accuracy (\\%)")
    ax.set_title("(b) CEfinal, fGC")
    ax.legend(frameon=False)
    ax.set_ylim(min(v + [cef[0]]) - 2.5, max(v + [cef[0]]) + 3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "levels_sweep.pdf"))
    plt.close(fig)
    print("wrote levels_sweep.pdf")


if __name__ == "__main__":
    main()
