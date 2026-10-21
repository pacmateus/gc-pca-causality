"""Tests of Table 1 of Section 4: the criterion recovers a known dimension k.

Only the GC column is asserted: it is deterministic and returns k for every draw
of the construction.  The knee and Tavory columns depend on the draw, which is why
the paper's single set of values is not pinned here.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pca"))

from gc_optk import gc_optk                        # noqa: E402
from run_pca_synthetic import KS, sonar, synthetic  # noqa: E402


def test_gc_recovers_k():
    X = sonar()
    for seed in (0, 1, 2):
        for k in KS:
            Z = synthetic(X, k, seed)
            assert Z.shape[1] > k, "covariance must stay full rank"
            assert gc_optk(Z) == k, (seed, k, gc_optk(Z))


def test_construction_has_the_right_covariance_structure():
    """The extra columns are linear functions of the first k, so the rank is k.

    The perturbation of variance 1e-12 leaves a numerical tail at about 2e-6 of the
    first singular value, which is why the rank is taken at 1e-4.
    """
    X = sonar()
    Z = synthetic(X, 5, 0)
    sv = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    assert int(np.sum(sv > sv[0] * 1e-4)) == 5


if __name__ == "__main__":
    for fn in (test_gc_recovers_k, test_construction_has_the_right_covariance_structure):
        fn()
        print("PASS", fn.__name__)
