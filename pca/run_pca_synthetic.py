"""Table 1 of Section 4: synthetic data with a known reduced dimension.

Construction (the paper's): start from the Sonar data, keep the first k columns,
and replace the remaining d-k by random linear combinations of them plus a
Gaussian perturbation of variance 1e-12.  The reduced dimension of the result is k
by construction, so a criterion can be scored against a known answer.

The GC column is deterministic: it returns k for every draw.  The knee and Tavory
columns depend on the random draw, which is why they are printed per seed here
rather than pinned to the paper's single draw.

    python pca/run_pca_synthetic.py
    python pca/run_pca_synthetic.py --seeds 0 1 2 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gc_optk import gc_optk, knee_optk, nml_optk               # noqa: E402
from run_pca_tables import minka                               # noqa: E402

DATA = os.path.join(HERE, "data")
KS = (5, 10, 30)
TAU2 = 1e-12


def sonar():
    """Sonar with the class label as a numeric column (d = 61, as in the paper)."""
    raw = pd.read_csv(os.path.join(DATA, "sonar.all-data.csv"), header=None)
    lab = raw.iloc[:, -1].map({"R": 0, "M": 1}).to_numpy(float)
    X = raw.iloc[:, :-1].to_numpy(float)
    Z = X - X.mean(axis=0)
    return np.column_stack([Z, lab - lab.mean()])


def synthetic(X, k, seed, tau2=TAU2):
    N, d = X.shape
    rng = np.random.default_rng(seed)
    Xk = X[:, :k]
    W = rng.standard_normal((k, d - k))
    eps = rng.standard_normal((N, d - k)) * np.sqrt(tau2)
    return np.hstack([Xk, Xk @ W + eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    X = sonar()
    print(f"Sonar, d = {X.shape[1]}, N = {X.shape[0]}")
    print(f"{'seed':>4s} {'k':>3s} {'Knee':>5s} {'Minka':>6s} {'Tavory':>7s} {'GC':>4s}")
    ok = True
    for seed in args.seeds:
        for k in KS:
            Z = synthetic(X, k, seed)
            gc = gc_optk(Z)
            ok = ok and gc == k
            print(f"{seed:4d} {k:3d} {knee_optk(Z):5d} {str(minka(Z)):>6s} "
                  f"{nml_optk(Z):7d} {gc:4d}   {'ok' if gc == k else 'ERRO'}")
    print("GC recovers k in every cell" if ok else "GC failed somewhere")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
