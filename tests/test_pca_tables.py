"""Tests of the PCA criterion against the numbers in Section 4 of the paper.

Run with pytest, or directly:

    python tests/test_pca_tables.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pca"))

from gc_optk import gc_optk, nml_optk                    # noqa: E402
from run_pca_tables import DATASETS, load                 # noqa: E402

# Table 3 of the paper: dimension selected on each dataset
EXPECTED = {
    "Wine": (6, 12, 11),
    "Sonar": (14, 56, 58),
    "Breast Cancer (WD)": (20, 29, 28),
    "Divorce": (7, 30, 52),
    "Statlog Vehicle": (15, 17, 16),
    "Parkinsons": (18, 20, 20),
    "Music origins": (21, 67, 66),
    "Drug consumption": (11, 11, 10),
}


def _data(name):
    for n, spec in DATASETS:
        if n == name:
            return load(spec)
    raise KeyError(name)


def test_gc_reproduces_table():
    for name, (gc, _, _) in EXPECTED.items():
        X = _data(name)
        assert X is not None, f"missing data for {name}"
        assert gc_optk(X, s=32) == gc, name


def test_tavory_reproduces_table():
    for name, (_, _, tav) in EXPECTED.items():
        X = _data(name)
        assert nml_optk(X) == tav, name


def test_gc_never_exceeds_dimension():
    for name in EXPECTED:
        X = _data(name)
        m = gc_optk(X, s=32)
        assert 1 <= m <= X.shape[1], name


def test_precision_constant_is_honoured():
    """The selected dimension depends on the precision s, monotonically.

    A larger s means a larger volume penalty, hence a smaller selected dimension:
    on Sonar, s = 4, 8, 16, 32, 64 give 30, 26, 20, 14, 10.  s = 32 (32-bit
    floats) is the value fixed in the paper.
    """
    X = _data("Sonar")
    dims = [gc_optk(X, s=s) for s in (4, 8, 16, 32, 64)]
    assert dims == sorted(dims, reverse=True), dims
    assert dims == [30, 26, 20, 14, 10], dims


if __name__ == "__main__":
    for fn in (test_gc_reproduces_table, test_tavory_reproduces_table,
               test_gc_never_exceeds_dimension, test_precision_constant_is_honoured):
        fn()
        print("PASS", fn.__name__)
