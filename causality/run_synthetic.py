"""fGC on synthetic data generated under the hypothesis it assumes (Section 5.5).

The mechanism is Y = f(X) + multinomial noise with f surjective and *not*
injective, and |Y| < |X| strictly.  That condition is what gives the functional
hypothesis its content: if the relation is injective, the reverse map is a
function as well and no criterion of functional asymmetry has power (see
tests/test_causality_cep.py::test_bijection_has_no_asymmetry).

The paper reports fGC at 100% in every cell of this grid, at every sample size,
and the comparators (DC, CISC, HCR) below it; the comparators are third-party
implementations and are not bundled here (see docs/EXPERIMENTS.md).

    python causality/run_synthetic.py            # full grid, fifty trials per cell
    python causality/run_synthetic.py --runs 5   # quick check
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fgc                                        # noqa: E402

fgc.add_gc_module_path()
from GeometricComplexity import causality as fgc_causality  # noqa: E402

K_GRID = (3, 5, 8, 10)
N_GRID = (50, 100, 200, 500, 1000, 1500, 2000, 3000)
NOISE = 10          # a sample is corrupted with probability 1/NOISE


def faithful_pair(k, y_len, N, noise=NOISE, dependency="independent"):
    """Y = f(X) + noise, with f surjective and non-injective (|Y| < |X|)."""
    assert y_len < k, "the effect must have strictly fewer labels than the cause"
    x = np.random.randint(0, k, size=N)
    f = np.random.choice(np.arange(y_len), size=k, replace=True)   # collisions
    y = f[x].copy()
    if noise:
        hit = np.random.randint(noise, size=N) == 1
        err = np.random.randint(0, y_len, size=N)
        err = np.where(err == y, (err + 1) % y_len, err)           # a different label
        y = np.where(hit, err, y)
    return x, y


def run(runs=50, seed=20260924):
    np.random.seed(seed)
    total = correct = 0
    non_injective = 0
    for dependency in ("independent", "dependent"):
        for k in K_GRID:
            for N in N_GRID:
                cell = 0
                for _ in range(runs):
                    y_len = np.random.randint(2, k)                 # 2 .. k-1
                    x, y = faithful_pair(k, y_len, N, NOISE, dependency)
                    non_injective += int(len(set(y.tolist())) < len(set(x.tolist())))
                    a, b = fgc_causality(x, y)
                    ok = b - a > 0
                    cell += ok
                    correct += ok
                    total += 1
                print(f"  {dependency:11s} k={k:2d} N={N:5d}   {cell / runs * 100:6.1f}%")
    print(f"overall {correct}/{total} = {correct / total * 100:.1f}%  "
          f"(datasets verified non-injective: {non_injective}/{total})")
    return correct == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    return 0 if run(args.runs) else 1


if __name__ == "__main__":
    sys.exit(main())
