"""Tests of the aggregation on the CEfinal benchmark (Section 5, Table 6)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "causality"))

import fgc                                          # noqa: E402
fgc.add_gc_module_path()
from GeometricComplexity import causality as fgc_causality  # noqa: E402

RAW = 53.57
K3 = 60.71


def _accuracy(k):
    pairs = fgc.load_cefinal()
    hits = []
    for a_vals, b_vals, tgt in pairs:
        x = np.asarray(a_vals) if k == 0 else fgc.aggregate_to_k(a_vals, k)
        y = np.asarray(b_vals) if k == 0 else fgc.aggregate_to_k(b_vals, k)
        a, b = fgc_causality(x, y)
        hits.append((b - a > 0) == (tgt == 1))
    return float(np.mean(hits) * 100), len(pairs)


def test_aggregation_to_three():
    acc3, n = _accuracy(3)
    assert n == 112, n
    assert abs(acc3 - K3) < 0.05, acc3


def test_raw_categories():
    acc0, _ = _accuracy(0)
    assert abs(acc0 - RAW) < 0.05, acc0


def test_aggregate_to_k_keeps_k_levels():
    vals = np.array(["a"] * 10 + ["b"] * 5 + ["c"] * 3 + ["d"] + ["e"])
    out = fgc.aggregate_to_k(vals, 3)
    assert len(set(out)) <= 3
    assert out[0] == out[1] == out[2]


if __name__ == "__main__":
    for fn in (test_aggregate_to_k_keeps_k_levels, test_raw_categories,
               test_aggregation_to_three):
        fn()
        print("PASS", fn.__name__)
