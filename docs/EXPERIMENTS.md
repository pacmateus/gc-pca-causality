# What is bundled, what is not, and how to get the rest

Everything needed to recompute the numbers of the paper is either in this
repository or is one of the third-party baselines listed below. Nothing is hidden
behind an author's machine.

## Bundled

| Item | Files | Size |
|---|---|---|
| Tuebingen CEP, 102 eligible pairs + `pairmeta.txt` | `causality/data/tuebingen/` | 4.1 MB |
| CEfinal, 112 categorical pairs + target and type tables | `causality/data/cefinal/` | ~2 MB |
| UCI datasets of Tables 2 and 3 (features only) | `pca/data/` | ~2.4 MB |
| The fGC module of the authors | `causality/gc_module/` | 17 KB |

`pca/data/` holds each dataset with the label columns removed and constant columns
dropped, which is the form used in the paper; `docs/DATA.md` records which columns
were dropped and why.

## Not bundled

**BlogFeedback** (63 MB, `blogData_train.csv`). Its row of Table 3 is GC = 24:
280 columns, of which 252 are constant and removed, first 1000 samples. Put the CSV
in `pca/data/` and run `pca/run_pca_tables.py` to reproduce it.

**SLOPE** (Marx & Vreeken 2017). The paper quotes 73.90% weighted on the same 102
pairs, from a re-implementation of the two-part MDL score described in the paper
(global fit with linear, quadratic, cubic, exponential and reciprocal classes, plus
the local refinement of their Algorithm 1, precision 3, local scale 5). The
authors' R code is released with their paper (arXiv:1709.08915); note that their
published headline is 82.4%, which is a decided-subset figure, and 77.2% when a
decision is forced on every pair.

**IGCI** (Janzing et al. 2012). The paper quotes 66.06%, the uniform-reference
entropy estimator on the same pairs, which matches the figure published with the
method.

**CISC, DC and HCR** are the categorical comparators of Table 6 and of the
synthetic table. They are not bundled: CISC and DC are the authors' Python
implementations from the CISC repository, and HCR is an R function driven through
`rpy2` (`HCR_function.r`, R at `/opt/homebrew/bin/R` on our machines). Each method
has its own sign convention, and copying one method's decision test to another is
the mistake that inverted the CISC column of an earlier draft:

| method | correct test |
|---|---|
| fGC | `b > a` |
| DC | `b > a` |
| CISC | `a > b` |
| HCR | `a > b` |

`GeometricComplexity.causality(x, y)` returns `(cost X->Y, cost Y->X)`, which is
where the `b > a` convention comes from. Ties must be counted separately: DC and
HCR abstain on a non-trivial share of the grid, and every method ties on a
bijective pair.

## The synthetic generator

The comparator suite used a generator shipped with the authors' code whose mapping
was built with `np.random.choice(..., replace=False)` and is therefore injective
whenever the cause has no more labels than the effect; on such pairs the reverse map
is single-valued, the pair is symmetric, and no criterion of functional asymmetry
can have power. The generator in `causality/run_synthetic.py` builds `f` with
`replace=True` and asserts `|Y| < |X|`, and reports how many datasets were verified
non-injective. The authors' generator was left untouched, since it is the artefact
behind the published subsection.
