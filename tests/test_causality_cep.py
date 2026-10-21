"""Tests of fGC on the Tuebingen CEP benchmark (Section 5, Tables 4 and 5)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "causality"))

import fgc                                              # noqa: E402
fgc.add_gc_module_path()
from GeometricComplexity import causality as fgc_causality  # noqa: E402

WEIGHTED_L2 = 61.57
WEIGHTED_L3 = 77.81


def _sweep(levels):
    pairs = fgc.load_tuebingen()
    w = np.array([p["w"] for p in pairs])
    out = {}
    for L in levels:
        hits = []
        for p in pairs:
            a, b = fgc_causality(fgc.qlin(p["x"], L), fgc.qlin(p["y"], L))
            hits.append(b - a > 0)
        out[L] = (float(np.sum(w * np.array(hits)) / w.sum() * 100), hits, w)
    return pairs, out


def test_two_and_three_levels():
    _, out = _sweep((2, 3))
    assert abs(out[2][0] - WEIGHTED_L2) < 0.05, out[2][0]
    assert abs(out[3][0] - WEIGHTED_L3) < 0.05, out[3][0]


def test_paired_test_against_bz2():
    """Table 5: fGC and bz2 disagree on 34 pairs, bz2 wins 18 of them, p = 0.86."""
    import baselines_compression as bc
    _, out = _sweep((3,))
    _, hits, w = out[3]
    bz = [r["right"] for r in bc.evaluate("bz2")["per_pair"]]
    a_only = [i for i in range(len(hits)) if hits[i] and not bz[i]]
    b_only = [i for i in range(len(hits)) if bz[i] and not hits[i]]
    assert (len(a_only), len(b_only)) == (16, 18), (len(a_only), len(b_only))
    n, k = len(a_only) + len(b_only), min(len(a_only), len(b_only))
    p = min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)
    assert abs(p - 0.86) < 0.01, p


def test_decision_sign_convention():
    """causality(x, y) returns (cost x->y, cost y->x): x causes y iff the second is larger."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 400)
    y = np.minimum(x, 1)                       # deterministic, many-to-one
    a, b = fgc_causality(x, y)
    assert b > a, (a, b)


def test_bijection_has_no_asymmetry():
    """On a bijective pair both directions have the same codelength, so the methods tie.

    This is why the synthetic benchmark of Section 5.5 restricts the mechanism to
    be non-injective: on a permutation there is nothing to detect.
    """
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 400)
    perm = {0: 1, 1: 2, 2: 0}
    y = np.array([perm[v] for v in x])
    a, b = fgc_causality(x, y)
    assert abs(a - b) < 1e-9, (a, b)


if __name__ == "__main__":
    for fn in (test_decision_sign_convention, test_bijection_has_no_asymmetry,
               test_two_and_three_levels, test_paired_test_against_bz2):
        fn()
        print("PASS", fn.__name__)
