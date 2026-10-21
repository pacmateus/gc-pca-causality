"""Cumulative explained variance of the dimensions selected by each method.

The paper reports these numbers in Section 4.3 to make two points: the explained
variance gives a scale for the decision, and it cannot replace the criterion,
because on some datasets a single component already explains almost everything.

    python pca/explained_variance.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_pca_tables import DATASETS, load  # noqa: E402

# dimension selected by GC in the paper's tables, and by the other methods
SELECTED = {
    "Wine": {"knee": 2, "GC": 6, "Minka": 12, "Tavory": 11},
    "Sonar": {"knee": 8, "GC": 14, "Minka": 56, "Tavory": 58},
    "Breast Cancer (WD)": {"knee": 2, "GC": 20, "Minka": 29, "Tavory": 28},
    "Divorce": {"knee": 2, "GC": 7, "Minka": 30, "Tavory": 52},
    "Statlog Vehicle": {"knee": 2, "GC": 15, "Minka": 17, "Tavory": 16},
}


def explained(X):
    Z = X - X.mean(axis=0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    return np.cumsum(ev / ev.sum())


def main():
    print(f"{'Dataset':20s} " + "  ".join(f"{m:>16s}" for m in ("knee", "GC", "Minka", "Tavory"))
          + "   m for 90/95/99%")
    for name, spec in DATASETS:
        if name not in SELECTED:
            continue
        X = load(spec)
        if X is None:
            continue
        cum = explained(X)
        sel = SELECTED[name]
        cells = []
        for m in ("knee", "GC", "Minka", "Tavory"):
            k = sel[m]
            cells.append(f"m={k:2d} {cum[k-1]*100:6.2f}%" if k <= len(cum) else " " * 17)
        thresholds = " / ".join(str(int(np.searchsorted(cum, t) + 1)) for t in (0.90, 0.95, 0.99))
        print(f"{name:20s} " + "  ".join(cells) + f"   {thresholds}")


if __name__ == "__main__":
    main()
