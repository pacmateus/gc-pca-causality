"""Data loading and discretisation for the causal experiments of Section 5.

Two benchmarks are used:

* Tuebingen cause-effect pairs (CEP), real-valued.  Loaded cause-first, so the
  ground truth is always X -> Y, with the official pairmeta weights.
* CEfinal, nominal categorical.  Target 1 means A -> B.

Both are reduced to a small alphabet before the functional Geometric Complexity
(fGC) criterion is applied: linear quantisation to L levels for continuous
variables, frequency aggregation to K categories for nominal ones.  L = K = 3 is
the setting used in the paper.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TUEBINGEN = os.path.join(DATA, "tuebingen")
CEFINAL = os.path.join(DATA, "cefinal")

# preprocessing constants of the paper
MAXLEN = 8000          # series longer than this are subsampled
MIN_SAMPLES = 50
LEVELS = 3


def qlin(x, levels=LEVELS):
    """Linear quantisation of a continuous variable to `levels` ordered bins."""
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros(len(x), np.uint8)
    return np.clip(((x - lo) / (hi - lo) * (levels - 1)).round(),
                   0, levels - 1).astype(np.uint8)


def aggregate_to_k(values, k=LEVELS):
    """Frequency aggregation: keep the k-1 most frequent labels, merge the rest.

    Used for nominal categorical data, where the labels carry no order.
    """
    counter = Counter(values)
    most = [v for v, _ in counter.most_common(k - 1)]
    mapping = {v: i for i, v in enumerate(most)}
    return np.array([mapping.get(v, k - 1) for v in values], np.uint8)


def load_tuebingen(directory=TUEBINGEN):
    """Eligible univariate pairs, cause first, with the official weights.

    Returns a list of dicts with keys 'pair', 'x', 'y', 'w'.  The ground truth is
    always x -> y, which is why the pairs are read cause-first from pairmeta.
    """
    meta = os.path.join(directory, "pairmeta.txt")
    out = []
    with open(meta) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 6:
                continue
            pid, c0, c1, e0, e1, w = (int(f[0]), int(f[1]), int(f[2]),
                                      int(f[3]), int(f[4]), float(f[5]))
            if c0 != c1 or e0 != e1:      # univariate pairs only
                continue
            path = os.path.join(directory, "pair%04d.txt" % pid)
            if not os.path.exists(path):
                continue
            d = np.loadtxt(path)
            if d.shape[0] < MIN_SAMPLES:
                continue
            if d.shape[0] > MAXLEN:
                idx = np.linspace(0, d.shape[0] - 1, MAXLEN).astype(int)
                d = d[idx]
            x, y = d[:, c0 - 1], d[:, e0 - 1]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            out.append({"pair": pid, "x": x, "y": y, "w": w})
    return out


def load_cefinal(directory=CEFINAL):
    """Categorical pairs with a decided target.

    Returns a list of (a, b, target) with target 1 for A -> B and -1 for B -> A.
    """
    info = pd.read_csv(os.path.join(directory, "CEfinal_train_publicinfo.csv"))
    targ = pd.read_csv(os.path.join(directory, "CEfinal_train_target.csv"))
    tcol = targ.columns[1]
    info = info[info["A type"].isin(["Binary", "Categorical"])
                & info["B type"].isin(["Binary", "Categorical"])]
    targ = targ[targ[tcol].isin([-1, 1])]
    df = info.merge(targ, on="SampleID")
    out = []
    for _, r in df.iterrows():
        path = os.path.join(directory, str(r["SampleID"]) + ".txt")
        if not os.path.exists(path):
            continue
        d = np.loadtxt(path, dtype=str, ndmin=2)
        if d.shape[1] < 2 or d.shape[0] < MIN_SAMPLES:
            continue
        out.append((d[:, 0], d[:, 1], 1 if r[tcol] == 1 else -1))
    return out


def add_gc_module_path():
    """Make the authors' fGC module importable (it imports its own helpers)."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "gc_module")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
