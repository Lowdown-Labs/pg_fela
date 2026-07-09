[![CI](https://github.com/Lowdown-Labs/pg_fela/actions/workflows/ci.yml/badge.svg)](https://github.com/Lowdown-Labs/pg_fela/actions/workflows/ci.yml)
[![License: PostgreSQL](https://img.shields.io/badge/license-PostgreSQL-blue.svg)](LICENSE)
[![PostgreSQL 14 to 18](https://img.shields.io/badge/PostgreSQL-14%E2%80%9318-336791.svg)](https://www.postgresql.org/)
[![pgrx 0.19.1](https://img.shields.io/badge/pgrx-0.19.1-informational.svg)](https://github.com/pgcentralfoundation/pgrx)

# pg_fela - in database AutoML, a pgrx Postgres extension

[pg_fela.webm](https://github.com/user-attachments/assets/afa4c350-550d-4df5-9160-89ee573c62b5)

A full Rust [pgrx](https://github.com/pgcentralfoundation/pgrx) PostgreSQL extension that runs a
frozen tabular foundation model (FelaTab) inside the database: classify, impute, cluster, score
anomalies, rank feature importance, and explain predictions with a single `SELECT`, no training
step, nothing leaving Postgres.

# How to use it

Zero configuration: the FelaTab model is embedded in the extension binary at build time, so with
no GUC set, no file to place, and no network access at query time, `CREATE EXTENSION` and a
prediction work out of the box. The weights are not in git or Git-LFS; the build fetches them from
a public CDN (or a local `FELATAB_WEIGHTS` override) and embeds them into `pg_fela.so`. Point
`fela.model_path` at another file to load a different model.

```sql
CREATE EXTENSION pg_fela;

-- AutoML in a SELECT: learns from the labeled rows, predicts the ones where target is NULL
SELECT * FROM fela_automl('my_table', 'target_column');

-- Same, plus a per row trust/OOD score so a confident prediction on unfamiliar data gets flagged
SELECT * FROM fela_predict_trust('my_table', 'target_column');

-- Why did row 42 get this prediction? Top contributing features, signed toward/away from it
SELECT * FROM fela_explain_row('my_table', 'target_column', 42);

-- Implicit AutoML: builds my_table_ml, joining prediction/confidence/trust/ood/cluster back onto the base table
SELECT fela_create_view('my_table', 'target_column');
SELECT * FROM my_table_ml WHERE ood;             -- rows unlike anything the model learned from
SELECT * FROM my_table_ml ORDER BY confidence;   -- triage the least sure predictions first
```

The geometric ops (`fela_cluster`, `fela_similar`, and the `_over()` window functions) work on any
`float8[]` vector, so you can cluster, dedup, or flag outliers on embedding columns too, not just
tabular features.

Full SQL reference (every function, its arguments, and what each output column means):
[`docs/SQL_REFERENCE.md`](docs/SQL_REFERENCE.md).

# Limitations and Gotchas

- **No persisted model, every call refits.** There is no train once, serve many step: the in
  context FelaTab forward rereads the full support set on every classification call, and
  `rust_gbm` fits fresh on the support rows on every regression call. This buys "zero training
  step, instant AutoML on your current rows", not warehouse scale repeated serving. It is a
  great fit for hundreds to a few thousand support rows queried occasionally; it is the wrong
  tool for a support set of hundreds of thousands+ rows hit on every dashboard page load, or a
  materialized view refreshed on a tight schedule over a huge base table. A real pipeline that
  trains once and serves cheap lookups is the better tool at that scale.
- **Tabular model.** Features are auto detected and encoded (text to categorical, boolean, date,
  and missing value imputation all happen automatically, with a notice explaining what changed),
  but the underlying model was trained on tabular data. Supervised classify/regress on embedding
  vectors is unvalidated; use the geometric ops (`fela_cluster`, `fela_anomaly`, `fela_similar`,
  the `_over()` functions) on embeddings instead, and `fela_classify`/`fela_automl` on tabular
  columns.
- **Regression runs on `rust_gbm`, not the FM.** All regression (`fela_automl`, `fela_regress`,
  the regression branch of `fela_anomaly`/`fela_importance`, `fela_conformal_regress`) fits a
  gradient boosting model on the support rows at query time. FelaTab itself is classification only.
- **`n_feat` is capped** at `max_features` (100 for the shipped model).
- **`row_id` is the 1 based scan ordinal**, stable within a query snapshot, not a primary key.
  Rows with a NULL in any feature column are skipped.
- **Conformal intervals are experimental** for regression (`fela_conformal_regress`); read `band`
  as a coverage interval, not the average error.

# How to install and deploy

Zero config, one command:

```bash
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d ghcr.io/lowdown-labs/pg_fela:pg18
psql -h localhost -U postgres -c "SELECT * FROM fela_automl('your_table', 'your_target');"
```

The extension is `CREATE EXTENSION`ed automatically on first boot; the model is embedded in
`pg_fela.so`, so no GUC and no query-time network access are needed. (The image build fetches the
weights from `FELATAB_MODEL_URL` and embeds them at compile time; see
[`docs/INSTALL.md`](docs/INSTALL.md).)

A tarball onto an existing PostgreSQL install, and building from source with `cargo pgrx`, are
also supported. Full instructions: [`docs/INSTALL.md`](docs/INSTALL.md).

# What this Repo Builds / Provides

- **A full Rust pgrx extension** (`src/lib.rs`) exposing the SQL surface above: no C, no FFI seam.
- **A self contained FelaTab forward pass and its CPU kernels**, vendored in tree (`src/felatab.rs`,
  `src/ops.rs`, `src/qgemm.rs`, `src/safetensors_io.rs`), verbatim from Lowdown Labs' `fela-core`/
  `fela-models`. No dependency on any external inference server: this extension's one job is
  serving the model on Postgres as fast and as safely as possible.
- **A gradient boosting regressor** (`src/gbm.rs`, `rust_gbm`) for all regression paths.
- **The model embedded at build time.** The weights (`felatab_int8.safetensors`) are NOT committed
  to git or Git-LFS; `build.rs` fetches them at compile time from `FELATAB_MODEL_URL` (or a local
  `FELATAB_WEIGHTS` file), verifies the sha256, and `include_bytes!`s them into `pg_fela.so`. The
  tiny matching config (`tests/fixtures/felatab/felatab_config.json`) stays in-repo and is
  `include_str!`d. The weights come from [`lowdown-labs/fela-tab`](https://huggingface.co/lowdown-labs/fela-tab)
  and are licensed Apache-2.0, not under this repository's PostgreSQL License; their attribution
  notice is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Runnable SQL examples** ([`examples/`](examples)) and a Docker image ([`Dockerfile`](Dockerfile))
  that builds a ready to query `postgres:18` image with pg_fela already installed.
- Kernels run single threaded (no thread pool inside a PG backend); pgrx 0.19 supports PG13 to
  PG18, CI builds and tests against PG14 to PG18.

# Security, Trust, Performance and Testing

**In database, no data egress.** Predictions, clustering, and explanations run inside the Postgres
backend process; nothing is sent to an external service.

**Licensing.** pg_fela is under the PostgreSQL License (see [`LICENSE`](LICENSE)). The extension
statically links `rust_gbm` and `safetensors`, both Apache-2.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push/PR:

- **lint**: `rustfmt` + `clippy -D warnings` on the glue (the vendored kernels are byte faithful
  and lint exempt).
- **security**: `cargo deny check` (advisories, bans, licenses, sources; see
  [`deny.toml`](deny.toml)).
- **test**: `cargo pgrx test`, the 28 test `#[pg_test]` suite including the honesty gate, on
  PG 14 through 18.
- **e2e**: packages onto a throwaway stock PG18 cluster, `CREATE EXTENSION pg_fela`, then runs the
  golden test cases, the full AutoML showcase, and a latency check
  (`test/run_pgrx_test.sh`).

**Golden Tests** Two independent checks, both self contained:

- In database output vs a frozen golden reference: `test/run_pgrx_test.sh` packages the
  extension, spins a throwaway stock PostgreSQL cluster, `CREATE EXTENSION`s it, runs
  `fela_classify` on the golden fixture, and asserts the in database output matches the frozen
  fp32 reference within the int8 vs fp32 parity bar. Measured: `max|pgrx - golden| = 1.36e-2`,
  pass.
- In process parity: a `#[pg_test]` recomputes the reference from the in tree `FelaTabModel` in the
  same backend and asserts the pgrx SPI path matches it within `1e-5`, proving the array
  marshaling is exact.

**Performance.** Measured on a single core, this machine, with the small model (int8, dim 512, 14
layers) and a 6 row by 3 feature support set: 100 single row `fela_classify` calls in ~9.3 ms
total, ~0.093 ms per call. The model cache is per backend: the first call in a connection pays the
load, later calls reuse it.

# Contributing

Issues and pull requests are welcome. To build and test locally:

```bash
cargo install cargo-pgrx --version 0.19.1 --locked
cargo pgrx init --pg18 $(which pg_config)
cargo pgrx test pg18                # the #[pg_test] suite, including the honesty gate
bash test/run_pgrx_test.sh          # package -> throwaway cluster -> CREATE EXTENSION -> e2e checks
```

Licensed under the PostgreSQL License (see [`LICENSE`](LICENSE)). The FelaTab model weights
(`felatab_int8.safetensors`), embedded into the extension binary at build time, come from
[`lowdown-labs/fela-tab`](https://huggingface.co/lowdown-labs/fela-tab) and are licensed under the
Apache License, Version 2.0. Both licenses are permissive and impose no restriction on commercial
use. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the required Apache-2.0 attribution.
