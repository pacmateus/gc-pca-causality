# Data provenance and preprocessing

## PCA datasets (`pca/data/`)

Each CSV holds the features only, with the label column removed and constant
columns dropped, so that `pca/run_pca_tables.py` can read it directly. `d` and `N`
of the tables are the shape of these files.

| Dataset | File | Columns kept | Source |
|---|---|---|---|
| Wine | `wine.csv` | 13 features, label removed | UCI 109, `acuna:09` |
| Sonar | `sonar.all-data.csv` | 60 features, trailing `R`/`M` label dropped | UCI 151, `gorman:88` |
| Breast Cancer WD | `breast_cancer_wisconsin_diagnostic.csv` | 30 features, label removed | UCI 17, `wolberg:95` |
| Divorce | `divorce/divorce.csv` | 54 features + label, semicolon-separated, header row | `yontem:19` |
| Statlog Vehicle | `statlog_vehicle.csv` | 18 features, label removed | UCI 149 |
| Statlog Image Seg. | `statlog_image_segmentation.csv` | 18 features, label removed | UCI 147 |
| Parkinsons | `parkinsons.csv` | 22 features, label removed | UCI 174, `moro:14` |
| Drug consumption | `drug_consumption.data` | columns 2 to 13 (id and the 19 `CL*` labels dropped) | UCI 373, `fehrman:16` |
| Music origins | `music_origins.txt` | 68 features, latitude and longitude dropped | UCI 507, `zhou:14` |

Two notes on this table. The preprocessing is not neutral: dropping the wrong
column still produces a plausible dimension (Wine returns 6 even with 12 of its 13
features), so `pca/run_pca_tables.py` checks every value against the paper's tables
and prints a warning when a row diverges.

The files are written at full precision (`%.17g`), and on ill-conditioned data that
matters: Image Segmentation contains many duplicate rows, so its smallest
eigenvalues are tiny and rounding the entries to ten decimals moves the selected
dimension from 15 to 13 while changing every entry by less than 1e-10. If a CSV of
this repository is ever regenerated, keep the full precision or the last column of
Table 3 will drift.

## Tuebingen CEP (`causality/data/tuebingen/`)

The 102 eligible pairs of Mooij et al. (2016), with the official `pairmeta.txt`.
Format of a metadata line:

```
pair_id  col_x1 col_x2 col_y1 col_y2  weight
```

Loading conventions, which are not optional:

* only **univariate** pairs are used (`c0 == c1` and `e0 == e1`);
* each pair is read **cause first**, `x = d[:, c0-1]`, `y = d[:, e0-1]`, so the
  ground truth is always `X -> Y`;
* at least 50 samples, and series longer than 8000 are subsampled with `linspace`;
* the accuracy is weighted by the last column.

The file column order is **not** the ground truth: 27 of the 102 pairs store the
effect in column 1, which is exactly why `pairmeta.txt` exists. Scoring the pairs in
file order inverts about a quarter of them and produces a plausible-looking number.

## CEfinal (`causality/data/cefinal/`)

`CEfinal_train_publicinfo.csv` (types of A and B), `CEfinal_train_target.csv`
(target 1 for A -> B, -1 for B -> A, 0 when undecided) and one text file per pair
with the two sequences, space-separated.

Preprocessing: keep the pairs whose two types are Binary or Categorical and whose
target is decided (-1 or 1), which leaves the 112 pairs of the paper. The target 1
means A -> B, so the hit test is `(b - a > 0) == (tgt == 1)`.

## Synthetic data

* **PCA, Table 1**: built from Sonar, `pca/run_pca_synthetic.py`. First `k` columns
  kept, the other `d-k` replaced by `X_k W + eps` with `W` standard normal and
  `eps` Gaussian of variance 1e-12.
* **Causality, Table 5**: built in `causality/run_synthetic.py`, with `f` surjective
  and non-injective and `|Y| < |X|` strictly, and multinomial noise with
  `p_0 = 0.9`, independent or drawn per value of the cause. Each dataset is checked
  to be non-injective before it is scored.
