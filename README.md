# Geometric Complexity: PCA and causal direction — reproduction code

Code for the paper **"Geometric Complexity, applications to optimal PCA and
Causality"** (A. M. Carvalho, S. Di Giorgio, P. Mateus, B. Mera, A. Souto).

Geometric Complexity (GC) is a codelength for a parametric model family that keeps
the curvature of the Fisher metric, and therefore refines the usual BIC-style
penalty by two terms: the Riemannian volume of the model space and a curvature
correction. This repository contains the code that produces every number and every
figure in the two applications of the paper:

* **Section 4, PCA.** The number of principal components is chosen by minimising
  the GC of the zero-mean Gaussian family with free covariance.
* **Section 5, causality.** The causal direction of a pair is chosen by comparing
  the GC of a multinomial mechanism in the two directions (fGC), with both
  variables aggregated to three levels.

## What is in here

```
pca/
  gc_optk.py               GC criterion for PCA (+ Tavory's NML and the knee)
  run_pca_tables.py        Tables 2 and 3 of Section 4
  run_pca_synthetic.py     Table 1: synthetic data with a known dimension
  explained_variance.py    cumulative explained variance at the selected dimension
  data/                    the small UCI datasets used in the tables

causality/
  fgc.py                   data loading, quantisation and frequency aggregation
  gc_module/               the fGC implementation (authors' module, see Provenance)
  run_cep_sweep.py         Tuebingen CEP: weighted accuracy against L  (Table 4, Fig. 3a)
  run_cefinal_sweep.py     CEfinal: total accuracy against K            (Table 6, Fig. 3b)
  baselines_compression.py sorted-conditional compressors bz2 / lzma    (Table 4)
  run_synthetic.py         fGC on synthetic data with a known mechanism (Table 5)
  data/tuebingen/          102 eligible CEP pairs + pairmeta weights
  data/cefinal/            the 112 categorical pairs (A, B) + target

figures/
  make_figures.py          Figures 1, 2 and 5 (scree, GC trade-off, CEfinal bars)
  make_sweeps.py           Figure 3 (the level sweeps) + levels_sweep_results.json

tests/                     19 checks: every headline number is asserted
docs/RESULTS.md            each number of the paper and the script that produces it
docs/EXPERIMENTS.md        what is bundled, what is third-party, how to obtain it
docs/DATA.md               data provenance and preprocessing
```

## Install and run

Python 3.10 with numpy, scipy, scikit-learn, pandas, matplotlib. (A `kneed`
installation is optional: the knee baseline falls back to a built-in
implementation of the same rule when the package is absent.)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q tests/            # ~20 s, 17 tests
```

Then, each script prints the numbers of the paper next to the value it computes:

```bash
python pca/run_pca_tables.py           # Tables 2 and 3
python pca/run_pca_synthetic.py        # Table 1
python pca/explained_variance.py       # the variance figures quoted in Section 4.3
python causality/run_cep_sweep.py      # Table 4 (fGC rows) and Figure 3(a)
python causality/run_cefinal_sweep.py  # Table 6 and Figure 3(b)
python causality/baselines_compression.py   # bz2 and lzma rows of Table 4
python causality/run_synthetic.py      # Table 5  (--runs 5 for a quick check)
python figures/make_figures.py         # writes four of the five figures as PDF
python figures/make_sweeps.py          # writes Figure 3 and its results JSON
```

Everything runs on a laptop; the full test suite takes about twenty seconds and the
two sweeps about a minute each.

## The two criteria, in one paragraph each

**PCA.** For a reduced covariance of dimension `m` the codelength is the fit
`-log p(x^N | Q_hat)`, plus the dimension penalty `m(m+1)/4 log(N/2pi)`, plus the
logarithm of the volume of the model space, plus the curvature correction
`-m(m+2)(m-1)/(24N)`. The optimum `m*` minimises the sum (`pca/gc_optk.py`,
printed term by term in Figure 1 of the paper). The precision `s` is the smallest
integer with `Sigma <= 2^{2s} I_d`; the paper fixes `s = 32`, the 32-bit floating
point representation. The selected dimension does depend on `s`, monotonically: on
Sonar, `s = 4, 8, 16, 32, 64` give `30, 26, 20, 14, 10`.

**Causality.** For a pair of discrete variables the mechanism is a table of counts,
a point of a simplex, and the GC of the multinomial family gives its codelength in
each direction; the shorter code is taken as causal. Both variables are reduced to
three levels first: linear quantisation for continuous data, frequency aggregation
for nominal data. Three is the smallest alphabet in which the mechanism can be
many-to-one without being constant, and the sweep is flat on both sides of it.

## Conventions that bite

Two conventions are easy to get wrong and both are pinned by tests:

* `GeometricComplexity.causality(x, y)` returns `(cost X->Y, cost Y->X)`, so
  "x causes y" is `b - a > 0`. The symptom of a flip is not a small numerical
  drift but the exact complement of every accuracy (39.29% where 60.71% is right).
* The compression baseline normalises by the delta code of each variable in its
  **original** order. Normalising by the sorted sequence instead gives 70.46% on
  the CEP rather than 74.66%, which looks like a weak baseline rather than a bug.

## Results

Weighted accuracy on the 102 eligible Tuebingen pairs and total accuracy on the 112
CEfinal pairs, all reproduced by the scripts above:

| Method | CEP (weighted) | CEfinal (total) |
|---|---|---|
| fGC, three levels | **77.81%** | **60.71%** |
| fGC, raw categories | — | 53.57% |
| bz2 sorted-conditional | 74.66% | 37.5% |
| lzma sorted-conditional | 62.33% | 50.0% |
| SLOPE | 73.90% (our re-run) | — |
| IGCI | 66.06% | — |
| CISC / HCR / DC | — | 57.14 / 50.00 / 50.90 |

The margin of fGC over bz2 on the CEP is **not** resolved by 102 pairs: the two
methods disagree on 34 pairs and bz2 wins 18 of them, exact McNemar `p = 0.86`.
The paper says so, and the test
`tests/test_causality_cep.py::test_paired_test_against_bz2` asserts it.

On synthetic data whose mechanism satisfies the assumption of the criterion (a
surjective, non-injective `f`, `|Y| < |X|`), fGC is right on every pair of the grid
from fifty samples; on a bijective pair the two codelengths coincide and every
method ties, which is why the benchmark restricts the mechanism.

## Provenance

`causality/gc_module/` is the fGC implementation of the authors (the module behind
the IDEAL 2025 paper *Causality in categorical data using geometric complexity*),
kept verbatim except for the correction to the multinomial volume term of
Theorem 3.6 that this paper introduces. It is included so that the numbers can be
recomputed, not as an independent implementation.

The bz2/lzma baseline in `causality/baselines_compression.py` was written from the
algorithm description and reproduces the published figures exactly: the same
102/102 per-pair decisions for both compressors and the same 25 lzma ties.

## Citing

If you use this code, please cite the paper:

```bibtex
@misc{carvalho2026geometric,
  title  = {Geometric Complexity, applications to optimal PCA and Causality},
  author = {Carvalho, Alexandra M. and Di Giorgio, Serena and Mateus, Paulo
            and Mera, Bruno and Souto, Andr{\'e}},
  year   = {2026},
  note   = {Manuscript}
}
```

and, for the GC framework itself, B. Mera, P. Mateus, A. M. Carvalho, *Model
complexity in statistical manifolds: the role of curvature*, IEEE Trans. Inf.
Theory 68(9):5619-5636, 2022.

## License

MIT, except `causality/gc_module/`, which is distributed by its authors under the
same terms as the paper it accompanies.
