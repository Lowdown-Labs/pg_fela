# pg_fela SQL reference

Every function pg_fela exposes, its arguments, and what each output column means. For the quick
start hero path, see the main [README](../README.md#how-to-use-it); for a runnable, end to end
demo of every function below, see [`examples/automl_showcase.sql`](../examples/automl_showcase.sql).

Everything is built on the one FelaTab forward pass. Feature arrays are row major, flattened, and
raw (the model standardizes internally using the support set statistics).

## Implicit AutoML: one call, then just `SELECT *`

This is the hero path: one call, then `SELECT *` gives you predictions and whether to trust them.
`fela_create_view` builds `<tbl>_ml` (a materialized view by default) that joins the requested
AutoML outputs back onto the base table, so you never hand tie `row_id`s back to your rows:

```sql
SELECT fela_create_view('fruits', 'kind');   -- add cluster/anomaly_score/predicted/confidence/trust/trust_label/ood/band
SELECT id, kind, cluster, anomaly_score, predicted, confidence, trust, trust_label, ood, band
FROM fruits_ml;                               -- keyed by your PK
SELECT * FROM fruits_ml WHERE ood;            -- rows unlike anything the model learned from
SELECT * FROM fruits_ml WHERE trust_label = 'low trust';  -- same rows, without remembering trust cutoffs
SELECT * FROM fruits_ml ORDER BY confidence;  -- triage the least sure predictions first
REFRESH MATERIALIZED VIEW fruits_ml;          -- recompute after the data changes
```

`fela_create_view(tbl text, target text DEFAULT NULL, add text DEFAULT 'cluster,anomaly,predict', k int DEFAULT 3, materialized bool DEFAULT true) -> text`

- `add` selects which of `cluster` / `anomaly` / `predict` (alias `impute`) columns to include.
- The `predict` group surfaces `predicted`, `confidence`, `trust`, `trust_label`, `ood`, and `band`
  together (it joins `fela_predict_trust`, the per row OOD/trust signal, instead of the narrower
  `fela_automl`: the `prediction`/`confidence` values are unchanged, `trust`/`ood` just ride
  along). This is additive: a caller only selecting `predicted` sees the same value as before.
- The base table's single column primary key is detected and carried through (the view's row
  identity); with no single column PK it falls back to the scan ordinal (reported in the return
  message).
- Clustering excludes the target column from its features (the label as feature gotcha).
- If `target` is NULL, `predicted`/`confidence`/`trust`/`trust_label`/`ood`/`band` and
  `anomaly_score` are skipped (all need labels); `cluster` still works.
- `fela_predict_trust` needs at least 2 labeled rows to build its OOD reference (one more than
  `fela_automl` needed for `predicted` alone). With fewer than 2 labeled rows, `predict` degrades
  gracefully instead of failing the view: it falls back to the `fela_automl` based CTE, so the
  view still builds with `predicted`/`confidence` populated and `trust`/`trust_label`/`ood`/`band`
  NULL (a notice explains why). `cluster` and `anomaly_score` are never affected by this fallback.

### Column semantics (`<tbl>_ml`)

So a `SELECT *` is self explaining without reading the source:

| Column | Meaning |
|---|---|
| `predicted` | The point prediction: the class code as is for classification, or the regression value rounded to the target column's own observed decimal precision (never more precise than your input data). |
| `confidence` (0..1) | Classification: the softmax max class probability, how sure the model is among the classes it has seen. Regression: this instead mirrors `trust` below (same 0..1 k-NN distance to support score), not a probability; treat regression `confidence` as "how typical is this row", not "how tight is this estimate" (that's what `band` is for). |
| `trust` (0..1) | The per row k-NN distance to support score, both tasks: how close this row's features are to the labeled rows the model actually learned from. High means interpolating inside familiar data; low means extrapolating. For regression it is numerically identical to `confidence` (see above). |
| `ood` (bool) | `true` when this row is flagged out of distribution by the same k-NN geometry as `trust` (it is the "low trust" state, as a boolean you can filter on directly). |
| `trust_label` | One of `'trusted'` / `'check'` / `'low trust'`, the same vocabulary `fela_explain` uses for `trust` (and `ood`, which always maps to `'low trust'`). Lets you `WHERE trust_label = 'low trust'` instead of remembering numeric cutoffs. |
| `band` | Regression only (NULL for classification, and NULL when there are fewer than 4 labeled rows to calibrate against). The split conformal half width at 80% coverage: read `predicted +/- band` as "the true value usually (about 80% of the time) falls in this range." It is a coverage interval, not the average error, and is rounded to the same precision as `predicted`. |
| `cluster` | k-means cluster id (k from the `k` argument, default 3); excludes the target column from its own features. |
| `anomaly_score` | Per labeled row disagreement score (NULL for the rows you are predicting): classification is `1 - P(true class)`, regression is the standardized residual from the GBM fit. High means this labeled row does not fit the pattern the model learned from the rest of the data. |

## Inline `_over()` window functions

So the surface reads as column expressions instead of table name strings: the window sees all rows
in the frame and returns a per row answer:

```sql
SELECT name, fela_cluster_over(sugar, acidity, weight) OVER () AS cluster FROM fruits;
SELECT name, fela_anomaly_over(sugar, acidity, weight) OVER () AS novelty FROM fruits;
```

| Function | Returns | Notes |
|---|---|---|
| `fela_cluster_over(VARIADIC float8[]) OVER ()` | `int` per row | k-means cluster id (k=3). Exactly matches `fela_cluster`. |
| `fela_anomaly_over(VARIADIC float8[]) OVER ()` | `float8` per row | Unsupervised RMS z-score novelty (distance from the frame centroid). Distinct from the supervised, model based `fela_anomaly(tbl, target)`. |

(Predict/classify is naturally per row with support, so it stays a regular function; an aggregate
doesn't fit there.)

## Table level "AutoML in a SELECT" (SPI)

| Function | Returns | What it does |
|---|---|---|
| `fela_automl(tbl, target)` | `TABLE(row_id bigint, prediction float8, confidence float8, task text)` | Auto detect classify/regress, learn from labeled rows, predict the rows where `target` is `NULL`. `confidence` is the softmax max class probability for classification; for regression it's the per row 0..1 trust score (k-NN distance to the support rows), not a std/error estimate. |
| `fela_impute(tbl, target)` | `TABLE(row_id bigint, imputed float8)` | Fill NULLs in any numeric column (automl, values only). |
| `fela_anomaly(tbl, target)` | `TABLE(row_id bigint, score float8, is_outlier bool)` | Per labeled row disagreement score (classify: `1-P(true class)`; regress: standardized residual). |
| `fela_cluster(tbl, k)` | `TABLE(row_id bigint, cluster int)` | k-means on standardized features. Uses all numeric non PK columns; skips NULL rows. |
| `fela_cluster_ex(tbl, k, exclude)` | `TABLE(row_id bigint, cluster int)` | Like `fela_cluster`, but also excludes `exclude` (a label/target) from the features. |
| `fela_importance(tbl, target)` | `TABLE(feature text, importance float8)` | Permutation importance per feature (in sample, disclosed): how much accuracy (classify) or error (regress) gets worse when that feature's values are shuffled. |
| `fela_explain(tbl, target)` | `text` | Plain language summary: task type and method, the top 1 to 2 features by permutation importance, and (regression only) a worked example row with its `predicted` value, the "usually within +/-X" conformal band, and its trust word. |
| `fela_explain_row(tbl, target, row_id bigint)` | `TABLE(feature text, contribution float8, direction text, value float8)` | Local, per row occlusion attribution: top features that drove this row's prediction, signed toward/away from the predicted class (or up/down for regression). |
| `fela_predict_trust(tbl, target)` | `TABLE(row_id bigint, prediction float8, confidence float8, trust float8, ood bool, task text)` | Same as `fela_automl`, plus a per row OOD/trust score (k-NN distance to support geometry) so a confident prediction on a row unlike the training data gets flagged instead of silently trusted. `trust` (0..1) and `ood` are the canonical fields; for regression `confidence` is set equal to `trust`, for classification `confidence` is the softmax max probability and is independent of `trust`. |
| `fela_conformal_regress(tbl, target, coverage)` | `TABLE(row_id bigint, prediction, lo, hi float8)` | Split conformal prediction intervals (regression, experimental). |
| `fela_conformal_threshold(tbl, target, coverage)` | `float8` | Split conformal min confidence threshold for classification abstention. |
| `fela_detect_task(tbl, target)` | `text` | Report the auto detected task and shape. |

`row_id` is the 1 based scan ordinal (stable within the query snapshot). Rows with a NULL in any
feature column are skipped. Table names are resolved via `regclass` and identifiers are quoted, so
qualified/mixed case names are safe.

## Array primitives

Lower level building blocks: you hand in the support/query feature arrays yourself, useful for
calling from application code or for full control over what counts as "support" vs "query".

| Function | Returns | Notes |
|---|---|---|
| `fela_classify(query_feats float8[], support_feats float8[], support_labels int[], n_feat int, n_class int)` | `float8[]` `[n_query*n_class]` | Softmax class probabilities, row major. |
| `fela_regress(query_feats float8[], support_feats float8[], support_labels float8[], n_feat int)` | `float8[]` `[n_query*2]` | `(mean, std)` per query row, GBM (fit on support). |
| `fela_classify_gated(query_feats, support_feats, support_labels, n_feat, n_class, threshold float8)` | `int` (nullable) | Class, or NULL when max prob `< threshold` (answer only when sure). |
| `fela_similar(query_feats float8[], support_feats float8[], n_feat int, k int)` | `TABLE(support_idx int, distance float8)` | k nearest support rows by standardized Euclidean distance. |
| `fela_argmax(probs float8[])` | `int` | 0 based top class. |
| `fela_confidence(probs float8[])` | `float8` | Max probability. |

## Introspection

| Function | Returns | Notes |
|---|---|---|
| `fela_version()` | `text` | Extension version, e.g. `'pg_fela 1.0.0 (pgrx; in tree FelaTab)'`. |
| `fela_model_info()` | `text` | Currently loaded model source and capabilities. |
| `fela_caps()` | `TABLE(max_features int, max_classes int, dim int, n_layers int, n_heads int, n_landmarks int)` | The embedded model's shape. |
