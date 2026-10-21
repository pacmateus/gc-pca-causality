# Every number of the paper, and the script that produces it

All values below were computed on this repository's data with the scripts named in
the last column. "verified" means the assertion is in `tests/`.

## Section 4, optimal number of principal components

Table 3 (`python pca/run_pca_tables.py`):

| Dataset | d | N | Knee | Minka | Tavory | GC |
|---|---|---|---|---|---|---|
| Drug consumption | 12 | 1885 | 3 | 11 | 10 | 11 |
| Divorce | 54 | 170 | 2 | 30 | 52 | 7 |
| Wine | 13 | 178 | 2 | 12 | 11 | 6 |
| Breast Cancer (WD) | 30 | 569 | 2 | 29 | 28 | 20 |
| Statlog Vehicle | 18 | 845 | 2 | 17 | 16 | 15 |
| Statlog Image Seg. | 18 | 2310 | 5 | 17 | 16 | 13 (paper: 14) |
| Parkinsons | 22 | 195 | 4 | 20 | 20 | 18 |
| Music origins | 68 | 1059 | 8 | 10 in the paper | 66 | 21 |
| Sonar | 60 | 208 | 8 | 56 | 58 | 14 |

Table 2 is the Sonar row. Table 1 (synthetic data with a known reduced dimension)
comes from `python pca/run_pca_synthetic.py`, which builds the data from Sonar as the
paper describes: keep the first `k` columns and replace the rest by random linear
combinations of them plus a perturbation of variance 1e-12. GC returns `k` in every
cell and for every seed, which is the claim of the table. The other three columns
depend on the random draw; the paper's single set (knee 6/4/5, Minka 58/55/51,
Tavory 5/59/36) is one such draw, and `run_pca_synthetic.py --seeds 0 1 2` prints
what a fresh draw gives.

Sensitivity to the precision `s` (Sonar), which the paper fixes at `s = 32`:

| s | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| m* | 30 | 26 | 20 | 14 | 10 |

Explained variance (`python pca/explained_variance.py`, quoted in Section 4.3):

| Dataset | at the knee | at the GC dimension | m for 90/95/99% |
|---|---|---|---|
| Sonar | 83.83% (m=8) | 93.03% (m=14) | 12 / 17 / 29 |
| Divorce | 80.75% (m=2) | 88.42% (m=7) | 9 / 17 / 35 |
| Wine | 99.98% (m=2) | 100.00% (m=6) | 1 / 1 / 1 |
| Statlog Vehicle | 97.73% (m=2) | 100.00% (m=15) | 1 / 1 / 4 |
| Breast Cancer (WD) | 99.82% (m=2) | 100.00% (m=20) | 1 / 1 / 2 |

The point of the last two columns is that the explained variance saturates on some
datasets and therefore cannot replace the criterion: on Wine a single component
explains 99.81% of the variance while the codelength selects 6.

## Section 5, causal direction

Depth of the Tuebingen sweep (`python causality/run_cep_sweep.py`), weighted
accuracy of fGC after linear quantisation:

| L | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 | 32 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| % | 61.57 | **77.81** | 66.78 | 63.79 | 69.07 | 66.93 | 66.93 | 58.42 | 52.84 | 52.41 | 49.39 |

Aggregation on CEfinal (`python causality/run_cefinal_sweep.py`), total accuracy:

| K | raw | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| % | 53.57 | 58.93 | **60.71** | 57.14 | 53.57 | 55.36 | 54.46 | 55.36 | 53.57 | 53.57 |

Compression baseline on the same 102 pairs (`python causality/baselines_compression.py`):

| coder | weighted accuracy | ties |
|---|---|---|
| bz2 | 74.66% | 0 |
| lzma | 62.33% | 25 |

Paired comparison of fGC (L = 3) against bz2 (`tests/test_causality_cep.py`):

* they disagree on 34 of the 102 pairs;
* fGC is alone in being right on 16 of them, bz2 on 18;
* exact McNemar `p = 0.86`, weighted difference +3.2 points.

This is why the paper claims the best point estimate on the CEP and state of the
art only on CEfinal.

Synthetic data (`python causality/run_synthetic.py`): fGC decides every pair
correctly in both noise regimes, over the whole grid, from `N = 50`; every dataset
in the grid is verified non-injective.

## Figures

| Figure | Script | Output |
|---|---|---|
| 1, GC trade-off on Sonar | `figures/make_figures.py` | `figures/pca_gc_tradeoff.pdf` |
| 2, scree plots | `figures/make_figures.py` | `figures/pca_scree_comparison.pdf` |
| 3, level sweeps | `figures/make_sweeps.py` | `figures/levels_sweep.pdf` + JSON |
| 4, aggregation schematic | `figures/make_figures.py` | `figures/aggregation_schematic.pdf` |
| 5, CEfinal bars | `figures/make_figures.py` | `figures/cefinal_comparison.pdf` |

`make_figures.py` self-checks the GC selection against Table 3 before drawing and
fails loudly if Wine, Sonar, Breast WD, Divorce or Vehicle stop matching.
