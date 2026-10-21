"""Reproduce Tables 1 to 3 of Section 4 (PCA optimal dimensionality).

Usage
-----
    python pca/run_pca_tables.py            # all datasets present locally
    python pca/run_pca_tables.py --quick    # only the five datasets of the tests

Expected values are the ones printed in the paper.  The three tables in the
paper are:

  Table 1  synthetic data with known k        (see tests/test_pca_synthetic.py)
  Table 2  Sonar: knee 8, Minka 56, Tavory 58, GC 14
  Table 3  nine further datasets

Data: the CSVs in pca/data/ are the features only (label columns dropped) with
constant columns removed, exactly as used in the paper; see docs/DATA.md for
the source of each one.  BlogFeedback is not shipped (63 MB) but the row of
Table 3 can be reproduced after downloading blogData_train.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gc_optk import gc_optk, knee_optk, nml_optk, zeromean  # noqa: E402

DATA = os.path.join(HERE, "data")

# name -> (file, columns to drop from the right, delimiter, header lines to skip,
#          explicit column selection or None)
DATASETS = [
    ("Drug consumption", ("drug_consumption.data", 0, ",", 0, list(range(1, 13)))),
    ("Divorce", ("divorce/divorce.csv", 1, ";", 1, None)),
    ("Wine", ("wine.csv", 0, ",", 0, None)),
    ("Breast Cancer (WD)", ("breast_cancer_wisconsin_diagnostic.csv", 0, ",", 0, None)),
    ("Statlog Vehicle", ("statlog_vehicle.csv", 0, ",", 0, None)),
    ("Statlog Image Seg.", ("statlog_image_segmentation.csv", 0, ",", 0, None)),
    ("Parkinsons", ("parkinsons.csv", 0, ",", 0, None)),
    ("Music origins", ("music_origins.txt", 2, ",", 0, None)),
    ("Sonar", ("sonar.all-data.csv", 1, ",", 0, None)),
]

EXPECTED_GC = {"Wine": 6, "Sonar": 14, "Breast Cancer (WD)": 20, "Divorce": 7,
               "Statlog Vehicle": 15, "Statlog Image Seg.": 15,
               "Parkinsons": 18, "Music origins": 21, "Drug consumption": 11}

# Rows whose value in the paper differs from a fresh recomputation go here, and
# are flagged in the output instead of silently adjusted.
KNOWN_DIFF = set()


def load(spec):
    """Read a CSV of features, skipping rows that are not numeric.

    The parsing is deliberately literal: every entry is converted with float() and
    nothing is rescaled. On ill-conditioned data this matters, see docs/DATA.md
    (Image Segmentation flips between 13 and 15 depending on the reader).
    """
    fname, drop, sep, skip, usecols = spec
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i < skip or not line.strip():
                continue
            parts = line.rstrip("\n").split(sep)
            if usecols is not None:
                try:
                    parts = [parts[i] for i in usecols]
                except IndexError:
                    continue
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue
            rows.append(vals)
    X = np.array(rows, float)
    if drop:
        X = X[:, :-drop]
    # contiguous, so that the SVD path (and therefore the last decimals of the
    # criterion) does not depend on the memory layout of the array
    return np.ascontiguousarray(X[:, X.std(axis=0) > 1e-10])


def minka(X):
    """Minka's Bayesian selection, as implemented by PCA(n_components='mle')."""
    from sklearn.decomposition import PCA
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return int(PCA(n_components="mle").fit(zeromean(X)).components_.shape[0])
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="only Wine, Sonar, Breast WD, Divorce, Statlog Vehicle")
    args = ap.parse_args()
    quick = {"Wine", "Sonar", "Breast Cancer (WD)", "Divorce", "Statlog Vehicle"}

    print(f"{'Dataset':20s} {'d':>4s} {'N':>6s} {'Knee':>5s} {'Minka':>6s} {'Tavory':>7s} {'GC':>4s}  check")
    print("-" * 72)
    bad = 0
    for name, spec in DATASETS:
        if args.quick and name not in quick:
            continue
        X = load(spec)
        if X is None:
            print(f"{name:20s}  (ficheiro ausente: {spec[0]})")
            continue
        d, N = X.shape[1], X.shape[0]
        k = knee_optk(X)
        mi = minka(X)
        tv = nml_optk(X)
        gc = gc_optk(X)
        exp = EXPECTED_GC.get(name)
        if exp is None or exp == gc:
            ok = "ok"
        elif name in KNOWN_DIFF:
            ok = f"nota: paper {exp}"
        else:
            ok = f"GC != {exp}"
            bad += 1
        print(f"{name:20s} {d:4d} {N:6d} {k:5d} {str(mi):>6s} {tv:7d} {gc:4d}  {ok}")
    print("-" * 72)
    print("GC reproduces the paper" if bad == 0 else f"{bad} linha(s) divergem do paper")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
