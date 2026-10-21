#!/usr/bin/env python3
"""
Generate the figures of the paper (replaces the four placeholder boxes).

Run with /opt/homebrew/bin/python3.10 (needs matplotlib + scikit-learn).

Figures produced in this directory:
  pca_scree_comparison.pdf   scree plot + optimal m of each method (Wine, Sonar)
  pca_gc_tradeoff.pdf        GC codelength decomposition around m* (Sonar)
  aggregation_schematic.pdf  linear quantization and frequency aggregation to 3 levels
  cefinal_comparison.pdf     CEfinal total accuracy, fGC against the baselines

Self-check: gc_curve() must reproduce the GC column of the paper's tables
(Wine 6, Sonar 14, Breast Cancer 20, Divorce 7, Vehicle 15); it prints the
comparison so the figure is never silently out of sync with the paper.
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "pca", "data"))
S = 32                                  # precision parameter of the paper

# colourblind-friendly palette (same as the rest of the working notes)
C_RED, C_BLUE, C_ORANGE, C_GREEN, C_DARK = "#e74c3c", "#3498db", "#e67e22", "#27ae60", "#2c3e50"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_csv(path, drop_labels=0, delimiter=",", max_rows=None):
    """Read a numeric CSV, dropping `drop_labels` trailing label columns."""
    raw = np.genfromtxt(path, delimiter=delimiter, dtype=str, max_rows=max_rows)
    keep = raw.shape[1] - drop_labels
    out = []
    for row in raw:
        try:
            out.append([float(v) for v in row[:keep]])
        except ValueError:
            continue
    return np.array(out, float)


def clean(X, drop_constant=True):
    X = X[np.isfinite(X).all(axis=1)]
    return X[:, X.std(axis=0) > 1e-10] if drop_constant else X


# Column counts of the CSVs in PCA_tests_py/Data, matching the paper's tables:
# wine/breast/vehicle carry features only, sonar/divorce carry a label column.
def load_wine():
    return clean(load_csv(os.path.join(DATA, "wine.csv")))


def load_sonar():
    return clean(load_csv(os.path.join(DATA, "sonar.all-data.csv"), drop_labels=1))


def load_breast():
    p = os.path.join(DATA, "breast_cancer_wisconsin_diagnostic.csv")
    return clean(load_csv(p))


def load_divorce():
    p = os.path.join(DATA, "divorce", "divorce.csv")
    return clean(load_csv(p, drop_labels=1, delimiter=";"))


def load_vehicle():
    return clean(load_csv(os.path.join(DATA, "statlog_vehicle.csv")))


# --------------------------------------------------------------------------
# the GC criterion of the paper, term by term (Section 3.3, s = 32)
# --------------------------------------------------------------------------
def gc_curve(X, s=S):
    """Per-m GC terms for a data matrix.  Returns a dict of arrays indexed by m."""
    zm = X - X.mean(axis=0)
    N, d = zm.shape
    pca = PCA().fit(zm)
    lam = pca.explained_variance_               # eigenvalues of the covariance
    rot = zm @ pca.components_.T
    Sg = rot.T @ rot                            # = diag(N * lam) up to ddof

    sumlog = [0.0]
    acc = 0.0
    for a in range(1, d + 1):
        acc += math.log(a) + math.lgamma(a / 2)
        sumlog.append(acc)

    nll, dim, vol, curv, total = [], [], [], [], []
    for m in range(1, d + 1):
        lam_bar = (sum(lam[m:]) / (d - m)) if m < d else 1.0
        Q = np.diag([1.0 / e for e in lam[:m]] + [1.0 / lam_bar] * (d - m))
        r = d * math.log(2 * math.pi) + sum(math.log(e) for e in lam[:m])
        r += (d - m) * math.log(lam_bar) + np.trace(Q @ Sg)
        r *= N / 2.0                                        # -log p(x^N | Q_hat)
        dim_t = m * (m + 1) / 4 * math.log(N / (2 * math.pi))
        curv_t = -m * (m + 2) * (m - 1) / (24 * N)
        vol_t = (-m / 2 * math.log(2) + m * (m + 1) / 4 * math.log(math.pi)
                 - sumlog[m] + m * math.log(s * math.log(2))
                 + (2 * s + 1) * m * (m - 1) / 4 * math.log(2))
        nll.append(r); dim.append(dim_t); vol.append(vol_t); curv.append(curv_t)
        total.append(r + dim_t + curv_t + vol_t)
    return dict(nll=np.array(nll), dim=np.array(dim), vol=np.array(vol),
                curv=np.array(curv), total=np.array(total),
                lam=lam, N=N, d=d, mstar=int(np.argmin(total) + 1))


# --------------------------------------------------------------------------
# figure 1: scree plot with the optimal dimension of each method
# --------------------------------------------------------------------------
PAPER_M = {
    "Wine":  dict(knee=2, minka=12, tavory=11, gc=6),
    "Sonar": dict(knee=8, minka=56, tavory=58, gc=14),
}


def fig_scree():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, (name, loader) in zip(axes, (("Wine", load_wine), ("Sonar", load_sonar))):
        X = loader()
        zm = X - X.mean(axis=0)
        pca = PCA().fit(zm)
        ev = pca.explained_variance_ratio_
        m = np.arange(1, len(ev) + 1)
        ax.plot(m, ev * 100, "-", color=C_DARK, lw=1.2, marker="o", ms=2.2)
        cfg = PAPER_M[name]
        for key, col, ls, lab in (("knee", C_RED, "--", "Knee"),
                                  ("minka", C_BLUE, ":", "Minka"),
                                  ("tavory", C_ORANGE, "-.", "Tavory"),
                                  ("gc", C_GREEN, "-", "GC")):
            ax.axvline(cfg[key], color=col, ls=ls, lw=1.1, alpha=0.9,
                       label=f"{lab}: {cfg[key]}")
        ax.set_xlabel("number of components $m$")
        ax.set_ylabel("explained variance ratio (\\%)")
        ax.set_title(f"{name}  ($d={X.shape[1]}$, $N={X.shape[0]}$)")
        ax.legend(frameon=False, loc="upper right")
        ax.set_xlim(0.5, min(len(ev), 30) + 0.5)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(ev.min() * 100 * 0.5, 1e-3))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pca_scree_comparison.pdf"))
    plt.close(fig)
    print("  wrote pca_scree_comparison.pdf")


# --------------------------------------------------------------------------
# figure 2: GC codelength decomposition around m*
# --------------------------------------------------------------------------
def fig_tradeoff(loader=load_sonar, name="Sonar"):
    X = loader()
    g = gc_curve(X)
    m = np.arange(1, g["d"] + 1)
    ms = g["mstar"]
    ref = ms - 1
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(m, g["total"] - g["total"][ref], "-", color=C_DARK, lw=1.8,
            label="total $L^*(x^N;m)$")
    ax.plot(m, g["nll"] - g["nll"][ref], "-", color=C_BLUE, lw=1.0,
            label="fit term $-\\log p(x^N|\\hat Q)$")
    ax.plot(m, g["dim"] - g["dim"][ref], "--", color=C_ORANGE, lw=1.0,
            label="dimension penalty")
    ax.plot(m, g["vol"] - g["vol"][ref], "-.", color=C_GREEN, lw=1.0,
            label="volume of $M(s)$")
    ax.plot(m, g["curv"] - g["curv"][ref], ":", color=C_RED, lw=1.0,
            label="curvature correction")
    ax.axvline(ms, color=C_DARK, lw=0.8, alpha=0.5)
    ax.annotate(f"$m^*={ms}$", xy=(ms, 0), xytext=(ms + 2.5, 0.45 * ax.get_ylim()[1]),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_DARK), fontsize=8)
    ax.set_xlabel("reduced dimension $m$")
    ax.set_ylabel("codelength relative to $m^*$  (nats)")
    ax.set_title(f"GC codelength decomposition, {name} ($d={g['d']}$, $N={g['N']}$, $s={S}$)")
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(0.5, g["d"] + 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pca_gc_tradeoff.pdf"))
    plt.close(fig)
    print(f"  wrote pca_gc_tradeoff.pdf   (m* = {ms})")
    return ms


# --------------------------------------------------------------------------
# figure 3: aggregation to three levels, continuous and nominal
# --------------------------------------------------------------------------
def fig_schematic():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))

    # (a) continuous -> three equal-width bins
    ax = axes[0]
    x = np.concatenate([rng.normal(-1.0, 0.55, 260), rng.normal(1.1, 0.45, 240)])
    lo, hi = x.min(), x.max()
    edges = np.linspace(lo, hi, 4)
    cols = [C_BLUE, C_ORANGE, C_GREEN]
    _c, _b, patches = ax.hist(x, bins=28, color="#bdc3c7", edgecolor="white", lw=0.3)
    for patch in patches:                       # colour each bar by its own bin
        centre = patch.get_x() + patch.get_width() / 2
        idx = min(2, int((centre - lo) / (hi - lo) * 3))
        patch.set_facecolor(cols[idx])
        patch.set_alpha(0.55)
    for i in range(3):
        ax.text((edges[:-1][i] + edges[1:][i]) / 2, ax.get_ylim()[1] * 0.88,
                f"${i}$", ha="center", fontsize=11, color=C_DARK)
    for e in edges[1:-1]:
        ax.axvline(e, color=C_DARK, lw=0.8, ls="--")
    ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    ax.set_xlabel("$x$")
    ax.set_ylabel("count")
    ax.set_title("(a) continuous: 3 equal-width bins")

    # (b) nominal categories -> top 2 kept, rest merged into "other"
    ax = axes[1]
    counts = np.array([38, 27, 14, 9, 6])
    colours = [C_BLUE, C_ORANGE, "#bdc3c7", "#bdc3c7", "#bdc3c7"]
    ax.bar(range(len(counts)), counts, color=colours, edgecolor=C_DARK, lw=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.8, str(c), ha="center", fontsize=7, color=C_DARK)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th"])
    ax.set_xlabel("categories, by decreasing count")
    ax.set_ylabel("count")
    ax.set_title("(b) nominal: keep the 2 most frequent, merge the rest")
    ax.text(0.5, counts.max() * 1.16, "labels $0$ and $1$", ha="center", fontsize=7.5,
            color=C_DARK)
    ax.annotate("", xy=(4.35, counts.max() * 0.72), xytext=(1.65, counts.max() * 0.72),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_DARK))
    ax.text(3.0, counts.max() * 0.76, "merged: label $2$", ha="center", fontsize=7.5,
            color=C_DARK)
    ax.set_ylim(0, counts.max() * 1.28)
    ax.set_xlim(-0.65, 4.65)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "aggregation_schematic.pdf"))
    plt.close(fig)
    print("  wrote aggregation_schematic.pdf")


# --------------------------------------------------------------------------
# figure 4: CEfinal total accuracy
# --------------------------------------------------------------------------
def fig_cefinal():
    methods = ["fGC\n(no aggregation)", "fGC + agg\n($K=3$)", "CISC", "DC", "HCR"]
    acc = [53.6, 60.7, 57.1, 50.9, 50.0]
    colours = [C_BLUE, C_GREEN, C_DARK, "#7f8c8d", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.bar(range(len(methods)), acc, color=colours, edgecolor="white", lw=0.6,
                  width=0.62)
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}\\%", ha="center",
                fontsize=7.5, color=C_DARK)
    ax.axhline(57.1, color=C_DARK, ls=":", lw=1.0)
    ax.text(4.45, 57.6, "CISC", fontsize=7, color=C_DARK, ha="right")
    ax.axhline(50.0, color="#bdc3c7", ls="--", lw=0.9)
    ax.text(-0.45, 50.35, "chance", fontsize=7, color="#7f8c8d", ha="left")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("total accuracy (\\%)")
    ax.set_ylim(45, 64)
    ax.set_title("CEfinal benchmark, 112 categorical pairs")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "cefinal_comparison.pdf"))
    plt.close(fig)
    print("  wrote cefinal_comparison.pdf")


if __name__ == "__main__":
    print("self-check: GC selection against the paper's Table 3")
    for name, loader, expected in (("Wine", load_wine, 6), ("Sonar", load_sonar, 14),
                                   ("Breast WD", load_breast, 20),
                                   ("Divorce", load_divorce, 7),
                                   ("Vehicle", load_vehicle, 15)):
        try:
            X = loader()
            g = gc_curve(X)
            ok = "OK" if g["mstar"] == expected else f"MISMATCH (paper says {expected})"
            print(f"  {name:10s} d={X.shape[1]:3d} N={X.shape[0]:5d}  "
                  f"GC m* = {g['mstar']:3d}   {ok}")
        except Exception as exc:
            print(f"  {name:10s} FAILED: {exc}")

    print("\nfigures:")
    fig_scree()
    fig_tradeoff()
    fig_schematic()
    fig_cefinal()
