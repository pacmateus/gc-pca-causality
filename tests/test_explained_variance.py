"""Tests of the explained-variance numbers quoted in Section 4.3."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pca"))

from explained_variance import explained             # noqa: E402
from run_pca_tables import DATASETS, load             # noqa: E402


def _data(name):
    for n, spec in DATASETS:
        if n == name:
            return load(spec)
    raise KeyError(name)


def test_sonar_at_the_gc_dimension():
    cum = explained(_data("Sonar"))
    assert abs(cum[7] * 100 - 83.83) < 0.05        # knee, m = 8
    assert abs(cum[13] * 100 - 93.03) < 0.05       # GC, m = 14
    assert int(np.searchsorted(cum, 0.95) + 1) == 17


def test_wine_saturates():
    cum = explained(_data("Wine"))
    assert abs(cum[0] * 100 - 99.81) < 0.05        # one component
    assert cum[5] * 100 > 99.999                   # six explain everything


def test_vehicle_saturates():
    cum = explained(_data("Statlog Vehicle"))
    assert abs(cum[1] * 100 - 97.73) < 0.05        # two components


if __name__ == "__main__":
    for fn in (test_sonar_at_the_gc_dimension, test_wine_saturates,
               test_vehicle_saturates):
        fn()
        print("PASS", fn.__name__)
