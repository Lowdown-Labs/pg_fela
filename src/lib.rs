use std::cell::RefCell;
use std::ffi::CString;

#[allow(clippy::all, dead_code)]
mod felatab;
#[allow(clippy::all, dead_code)]
mod ops;
#[allow(clippy::all, dead_code)]
mod qgemm;
#[allow(clippy::all, dead_code)]
mod safetensors_io;

use crate::felatab::FelaTabModel;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use pgrx::prelude::*;
use pgrx::spi::{quote_identifier, quote_literal, quote_qualified_identifier, SpiResult};
use serde::Deserialize;

::pgrx::pg_module_magic!();

static EMBEDDED_MODEL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/felatab_int8.safetensors"));
static EMBEDDED_CONFIG: &str = include_str!("../tests/fixtures/felatab/felatab_config.json");

static FELA_MODEL_PATH: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);
static FELA_CONFIG_PATH: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    GucRegistry::define_string_guc(
        c"fela.model_path",
        c"Filesystem path to the FelaTab safetensors weights (optional).",
        c"Superuser-settable; names a file the server backend reads. When unset (the default), the \
          bundled small int8 model embedded in the extension binary is used instead, with nothing \
          to configure. Set this to opt into a bigger tier.",
        &FELA_MODEL_PATH,
        GucContext::Suset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"fela.config_path",
        c"Path to the FelaTab config JSON (optional).",
        c"Only consulted when fela.model_path is set. Empty => derived as \
          <dir(model_path)>/felatab_config.json.",
        &FELA_CONFIG_PATH,
        GucContext::Suset,
        GucFlags::default(),
    );
    #[cfg(not(feature = "pg14"))]
    unsafe {
        pgrx::pg_sys::MarkGUCPrefixReserved(c"fela".as_ptr());
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ModelSource {
    Embedded,
    File {
        model_path: String,
        config_path: String,
    },
}

struct CachedModel {
    model: FelaTabModel,
    source: ModelSource,
}

thread_local! {
    static MODEL_CACHE: RefCell<Option<CachedModel>> = const { RefCell::new(None) };
}

#[derive(Deserialize, Clone, Copy)]
struct Caps {
    max_features: usize,
    max_classes: usize,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    #[serde(default)]
    n_landmarks: usize,
}

fn guc_string(g: &GucSetting<Option<CString>>) -> String {
    g.get()
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn resolve_source() -> ModelSource {
    let model_path = guc_string(&FELA_MODEL_PATH);
    if model_path.is_empty() {
        return ModelSource::Embedded;
    }
    let cfg = guc_string(&FELA_CONFIG_PATH);
    let config_path = if !cfg.is_empty() {
        cfg
    } else {
        match model_path.rfind('/') {
            Some(i) => format!("{}/felatab_config.json", &model_path[..i]),
            None => "felatab_config.json".to_string(),
        }
    };
    ModelSource::File {
        model_path,
        config_path,
    }
}

fn read_caps() -> Caps {
    let source = resolve_source();
    let json = match &source {
        ModelSource::Embedded => EMBEDDED_CONFIG.to_string(),
        ModelSource::File { config_path, .. } => std::fs::read_to_string(config_path)
            .unwrap_or_else(|e| error!("pg_fela: cannot read config \"{config_path}\": {e}")),
    };
    serde_json::from_str::<Caps>(&json).unwrap_or_else(|e| match &source {
        ModelSource::Embedded => error!("pg_fela: parse embedded config: {e}"),
        ModelSource::File { config_path, .. } => {
            error!("pg_fela: parse config \"{config_path}\": {e}")
        }
    })
}

fn with_model<R>(f: impl FnOnce(&FelaTabModel) -> Result<R, String>) -> R {
    let source = resolve_source();
    MODEL_CACHE.with(|cell| {
        {
            let hit = matches!(&*cell.borrow(), Some(c) if c.source == source);
            if !hit {
                let model = match &source {
                    ModelSource::Embedded => FelaTabModel::load(EMBEDDED_MODEL, EMBEDDED_CONFIG)
                        .unwrap_or_else(|e| error!("pg_fela: embedded model load failed: {e}")),
                    ModelSource::File {
                        model_path,
                        config_path,
                    } => {
                        let weights = std::fs::read(model_path).unwrap_or_else(|e| {
                            error!("pg_fela: cannot read model \"{model_path}\": {e}")
                        });
                        let cfg_json = std::fs::read_to_string(config_path).unwrap_or_else(|e| {
                            error!("pg_fela: cannot read config \"{config_path}\": {e}")
                        });
                        FelaTabModel::load(&weights, &cfg_json)
                            .unwrap_or_else(|e| error!("pg_fela: model load failed: {e}"))
                    }
                };
                *cell.borrow_mut() = Some(CachedModel { model, source });
            }
        }
        let slot = cell.borrow();
        f(&slot.as_ref().unwrap().model).unwrap_or_else(|e| error!("pg_fela: {e}"))
    })
}

fn infer_cls(
    m: &FelaTabModel,
    support_x: &[f32],
    support_y: &[f32],
    query_x: &[f32],
    n_feat: usize,
    ncls: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let ns = support_y.len();
    let nq = query_x.len() / n_feat;
    let mut x = Vec::with_capacity(support_x.len() + query_x.len());
    x.extend_from_slice(support_x);
    x.extend_from_slice(query_x);
    let flat = m.predict(&x, support_y, ns, nq, n_feat, 0, ncls)?;
    Ok(flat.chunks(ncls).map(|c| c.to_vec()).collect())
}

fn gbm_reg(sx: &[f32], sy: &[f32], qx: &[f32], n_feat: usize) -> Vec<(f32, f32)> {
    let ns = sy.len();
    let model = rust_gbm::GbmRegressor::fit(sx, sy, ns, n_feat);
    let fitted = model.predict(sx, ns);
    let ss: f32 = (0..ns).map(|i| (sy[i] - fitted[i]).powi(2)).sum();
    let std = if ns > 1 {
        (ss / (ns as f32 - 1.0)).sqrt()
    } else {
        0.0
    };
    let nq = qx.len() / n_feat;
    model
        .predict(qx, nq)
        .into_iter()
        .map(|p| (p, std))
        .collect()
}

#[pg_extern(immutable, strict, parallel_restricted)]
fn fela_classify(
    query_feats: Vec<f64>,
    support_feats: Vec<f64>,
    support_labels: Vec<i32>,
    n_feat: i32,
    n_class: i32,
) -> Vec<f64> {
    let (nf, nc) = check_dims(n_feat, n_class);
    let (_n_query, n_support) = split_counts(query_feats.len(), support_feats.len(), nf);
    if support_labels.len() != n_support {
        error!(
            "pg_fela: len(support_labels)={} != n_support={}",
            support_labels.len(),
            n_support
        );
    }
    let sx: Vec<f32> = support_feats.iter().map(|&v| v as f32).collect();
    let qx: Vec<f32> = query_feats.iter().map(|&v| v as f32).collect();
    let sy: Vec<f32> = support_labels.iter().map(|&v| v as f32).collect();
    let probs = with_model(|m| infer_cls(m, &sx, &sy, &qx, nf, nc));
    probs.into_iter().flatten().map(|v| v as f64).collect()
}

#[pg_extern(immutable, strict, parallel_restricted)]
fn fela_regress(
    query_feats: Vec<f64>,
    support_feats: Vec<f64>,
    support_labels: Vec<f64>,
    n_feat: i32,
) -> Vec<f64> {
    let nf = check_nfeat(n_feat);
    let (_n_query, n_support) = split_counts(query_feats.len(), support_feats.len(), nf);
    if support_labels.len() != n_support {
        error!(
            "pg_fela: len(support_labels)={} != n_support={}",
            support_labels.len(),
            n_support
        );
    }
    let sx: Vec<f32> = support_feats.iter().map(|&v| v as f32).collect();
    let qx: Vec<f32> = query_feats.iter().map(|&v| v as f32).collect();
    let sy: Vec<f32> = support_labels.iter().map(|&v| v as f32).collect();
    let out = gbm_reg(&sx, &sy, &qx, nf);
    out.into_iter()
        .flat_map(|(mean, std)| [mean as f64, std as f64])
        .collect()
}

#[pg_extern(immutable, strict, parallel_restricted)]
fn fela_classify_gated(
    query_feats: Vec<f64>,
    support_feats: Vec<f64>,
    support_labels: Vec<i32>,
    n_feat: i32,
    n_class: i32,
    threshold: f64,
) -> Option<i32> {
    let (nf, nc) = check_dims(n_feat, n_class);
    if query_feats.len() != nf {
        error!(
            "pg_fela: fela_classify_gated expects ONE query row (len {} != n_feat {})",
            query_feats.len(),
            nf
        );
    }
    let n_support = support_feats.len() / nf;
    if !support_feats.len().is_multiple_of(nf) || n_support < 1 {
        error!("pg_fela: bad support_feats length");
    }
    if support_labels.len() != n_support {
        error!(
            "pg_fela: len(support_labels)={} != n_support={}",
            support_labels.len(),
            n_support
        );
    }
    let sx: Vec<f32> = support_feats.iter().map(|&v| v as f32).collect();
    let qx: Vec<f32> = query_feats.iter().map(|&v| v as f32).collect();
    let sy: Vec<f32> = support_labels.iter().map(|&v| v as f32).collect();
    let probs = with_model(|m| infer_cls(m, &sx, &sy, &qx, nf, nc));
    let (cls, p) = argmax_prob(&probs[0]);
    if (p as f64) < threshold {
        None
    } else {
        Some(cls as i32)
    }
}

#[pg_extern(immutable, strict, parallel_safe)]
fn fela_similar(
    query_feats: Vec<f64>,
    support_feats: Vec<f64>,
    n_feat: i32,
    k: i32,
) -> TableIterator<'static, (name!(support_idx, i32), name!(distance, f64))> {
    let nf = check_nfeat(n_feat);
    if query_feats.len() != nf {
        error!(
            "pg_fela: fela_similar expects ONE query row (len {} != n_feat {})",
            query_feats.len(),
            nf
        );
    }
    let ns = support_feats.len() / nf;
    if !support_feats.len().is_multiple_of(nf) || ns < 1 {
        error!("pg_fela: bad support_feats length");
    }
    let (mean, scale) = col_stats(&support_feats, ns, nf);
    let zq: Vec<f64> = (0..nf)
        .map(|c| (query_feats[c] - mean[c]) / scale[c])
        .collect();
    let mut dists: Vec<(i32, f64)> = (0..ns)
        .map(|r| {
            let d2: f64 = (0..nf)
                .map(|c| {
                    let z = (support_feats[r * nf + c] - mean[c]) / scale[c] - zq[c];
                    z * z
                })
                .sum();
            (r as i32, d2.sqrt())
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k.max(0) as usize);
    TableIterator::new(dists)
}

#[pg_extern(immutable, strict, parallel_safe)]
fn fela_argmax(probs: Vec<f64>) -> i32 {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in probs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

#[pg_extern(immutable, strict, parallel_safe)]
fn fela_confidence(probs: Vec<f64>) -> f64 {
    probs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[pg_extern(immutable, strict, parallel_safe)]
fn fela_version() -> String {
    format!(
        "pg_fela {} (pgrx; in-tree FelaTab)",
        env!("CARGO_PKG_VERSION")
    )
}

#[pg_extern(stable, parallel_restricted)]
fn fela_model_info() -> String {
    let source = resolve_source();
    with_model(|_m| Ok::<(), String>(()));
    let c = read_caps();
    let loc = match source {
        ModelSource::Embedded => {
            "source=embedded (bundled small int8; SET fela.model_path to override)".to_string()
        }
        ModelSource::File {
            model_path,
            config_path,
        } => format!("source=file; model_path={model_path}; config_path={config_path}"),
    };
    format!(
        "loaded=yes; {loc}; \
         dim={}; layers={}; heads={}; landmarks={}; max_features={}; max_classes={}",
        c.dim, c.n_layers, c.n_heads, c.n_landmarks, c.max_features, c.max_classes
    )
}

#[pg_extern(stable, parallel_restricted)]
fn fela_caps() -> TableIterator<
    'static,
    (
        name!(max_features, i32),
        name!(max_classes, i32),
        name!(dim, i32),
        name!(n_layers, i32),
        name!(n_heads, i32),
        name!(n_landmarks, i32),
    ),
> {
    let c = read_caps();
    TableIterator::once((
        c.max_features as i32,
        c.max_classes as i32,
        c.dim as i32,
        c.n_layers as i32,
        c.n_heads as i32,
        c.n_landmarks as i32,
    ))
}

struct TableData {
    features: Vec<String>,
    feature_kinds: Vec<ColKind>,
    row_ids: Vec<i64>,
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    n: usize,
    nf: usize,
    target_is_int: bool,
    encoding_notes: Vec<String>,
}

fn is_numeric_typ(t: &str) -> bool {
    matches!(
        t,
        "int2" | "int4" | "int8" | "float4" | "float8" | "numeric"
    )
}
fn is_int_typ(t: &str) -> bool {
    matches!(t, "int2" | "int4" | "int8")
}
fn is_text_typ(t: &str) -> bool {
    matches!(t, "text" | "varchar" | "bpchar")
}
fn is_date_typ(t: &str) -> bool {
    matches!(t, "date" | "timestamp" | "timestamptz")
}

const BOOL_TRUE_TOKENS: &[&str] = &["true", "yes", "y", "t"];
const BOOL_FALSE_TOKENS: &[&str] = &["false", "no", "n", "f"];

fn norm_tok(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

fn is_bool_like_text(distinct_vals: &[String]) -> bool {
    !distinct_vals.is_empty()
        && distinct_vals.iter().all(|v| {
            let n = norm_tok(v);
            BOOL_TRUE_TOKENS.contains(&n.as_str()) || BOOL_FALSE_TOKENS.contains(&n.as_str())
        })
}
fn bool_text_to_f64(v: &str) -> Option<f64> {
    let n = norm_tok(v);
    if BOOL_TRUE_TOKENS.contains(&n.as_str()) {
        Some(1.0)
    } else if BOOL_FALSE_TOKENS.contains(&n.as_str()) {
        Some(0.0)
    } else {
        None
    }
}

fn parse_finite_f64(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

const MAX_CATEGORICAL_CARDINALITY: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColKind {
    Numeric,
    Boolean,
    BoolText,
    Categorical,
    Date,
}

struct FeatCol {
    name: String,
    kind: ColKind,
}

fn feat_sql_expr(ident: &str, kind: ColKind) -> String {
    match kind {
        ColKind::Numeric => format!("{ident}::float8"),
        ColKind::Boolean => format!(
            "(CASE WHEN {ident} IS NULL THEN NULL::float8 WHEN {ident} THEN 1.0 ELSE 0.0 END)"
        ),
        ColKind::BoolText => format!(
            "(CASE WHEN {ident} IS NULL THEN NULL::float8 \
              WHEN lower(trim({ident}::text)) IN ('true','yes','y','t') THEN 1.0 \
              WHEN lower(trim({ident}::text)) IN ('false','no','n','f') THEN 0.0 \
              ELSE NULL::float8 END)"
        ),
        ColKind::Date => format!("EXTRACT(EPOCH FROM {ident})::float8"),
        ColKind::Categorical => format!("{ident}::text"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetMode {
    NumericF64,
    BoolCoerced,
    DateCoerced,
    TextRaw,
}

fn target_sql_expr(ident: &str, mode: TargetMode) -> String {
    match mode {
        TargetMode::NumericF64 => format!("{ident}::float8"),
        TargetMode::BoolCoerced => feat_sql_expr(ident, ColKind::Boolean),
        TargetMode::DateCoerced => feat_sql_expr(ident, ColKind::Date),
        TargetMode::TextRaw => format!("{ident}::text"),
    }
}

fn load_table(tbl: &str, target: Option<&str>) -> TableData {
    let res: SpiResult<TableData> = Spi::connect(|client| {
        let relname: String = client
            .select(
                &format!("SELECT {}::regclass::text", quote_literal(tbl)),
                None,
                &[],
            )?
            .first()
            .get_one::<String>()?
            .expect("regclass");

        let mut pk: Vec<String> = Vec::new();
        let pk_q = format!(
            "SELECT a.attname::text FROM pg_index i JOIN pg_attribute a \
             ON a.attrelid=i.indrelid AND a.attnum = ANY(i.indkey) \
             WHERE i.indrelid = {}::regclass AND i.indisprimary",
            quote_literal(tbl)
        );
        for row in client.select(&pk_q, None, &[])? {
            if let Some(n) = row.get::<String>(1)? {
                pk.push(n);
            }
        }

        let col_q = format!(
            "SELECT a.attname::text, t.typname::text FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid \
             WHERE a.attrelid = {}::regclass AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum",
            quote_literal(tbl)
        );
        let mut col_meta: Vec<(String, String)> = Vec::new();
        for row in client.select(&col_q, None, &[])? {
            col_meta.push((
                row.get::<String>(1)?.unwrap(),
                row.get::<String>(2)?.unwrap(),
            ));
        }

        let mut feat_cols: Vec<FeatCol> = Vec::new();
        let mut target_mode = TargetMode::NumericF64;
        let mut target_is_int = false;
        let mut target_found = false;
        let mut notes: Vec<String> = Vec::new();

        for (name, typ) in &col_meta {
            if let Some(t) = target {
                if name == t {
                    target_found = true;
                    if is_numeric_typ(typ) {
                        target_mode = TargetMode::NumericF64;
                        target_is_int = is_int_typ(typ);
                    } else if typ == "bool" {
                        target_mode = TargetMode::BoolCoerced;
                        target_is_int = true;
                    } else if is_date_typ(typ) {
                        target_mode = TargetMode::DateCoerced;
                        target_is_int = false;
                    } else if is_text_typ(typ) {
                        target_mode = TargetMode::TextRaw;
                        target_is_int = false;
                    } else {
                        error!(
                            "pg_fela: target column \"{t}\" has unsupported type \"{typ}\" \
                             (supported: numeric, boolean, date/timestamp, text)"
                        );
                    }
                    continue;
                }
            }
            if pk.contains(name) {
                continue;
            }
            if is_numeric_typ(typ) {
                feat_cols.push(FeatCol {
                    name: name.clone(),
                    kind: ColKind::Numeric,
                });
            } else if typ == "bool" {
                feat_cols.push(FeatCol {
                    name: name.clone(),
                    kind: ColKind::Boolean,
                });
            } else if is_date_typ(typ) {
                feat_cols.push(FeatCol {
                    name: name.clone(),
                    kind: ColKind::Date,
                });
            } else if is_text_typ(typ) {
                let scan_q = format!(
                    "SELECT {}::text AS v, count(*) FROM {relname} WHERE {} IS NOT NULL GROUP BY 1",
                    quote_identifier(name),
                    quote_identifier(name)
                );
                let mut distinct_vals: Vec<String> = Vec::new();
                let mut nonnull: i64 = 0;
                for row in client.select(&scan_q, None, &[])? {
                    if let Some(v) = row.get::<String>(1)? {
                        distinct_vals.push(v);
                    }
                    nonnull += row.get::<i64>(2)?.unwrap_or(0);
                }
                if distinct_vals.is_empty() {
                    notes.push(format!("{name}: all-NULL text column dropped"));
                    continue;
                }
                if is_bool_like_text(&distinct_vals) {
                    feat_cols.push(FeatCol {
                        name: name.clone(),
                        kind: ColKind::BoolText,
                    });
                    notes.push(format!("{name}: text -> boolean"));
                    continue;
                }
                let dcount = distinct_vals.len();
                let mostly_unique = nonnull >= 5 && (dcount as f64 / nonnull as f64) > 0.9;
                if dcount > MAX_CATEGORICAL_CARDINALITY || mostly_unique {
                    notes.push(format!(
                        "{name}: {dcount} distinct text values (high-cardinality) dropped"
                    ));
                    continue;
                }
                feat_cols.push(FeatCol {
                    name: name.clone(),
                    kind: ColKind::Categorical,
                });
                notes.push(format!("{name}: text -> categorical ({dcount} levels)"));
            }
        }
        if !target_found {
            if let Some(t) = target {
                error!("pg_fela: target column \"{t}\" not found in {relname}");
            }
        }
        if feat_cols.is_empty() {
            error!(
                "pg_fela: no usable feature columns found in {} (after excluding PK/target; \
                 numeric/boolean/date/low-cardinality-text columns qualify)",
                relname
            );
        }

        let nf = feat_cols.len();
        let cat_positions: Vec<usize> = feat_cols
            .iter()
            .enumerate()
            .filter(|(_, f)| f.kind == ColKind::Categorical)
            .map(|(j, _)| j)
            .collect();

        let feat_sel = feat_cols
            .iter()
            .map(|f| feat_sql_expr(&quote_identifier(&f.name), f.kind))
            .collect::<Vec<_>>()
            .join(", ");
        let data_q = match target {
            Some(t) => format!(
                "SELECT (row_number() OVER (ORDER BY ctid))::bigint, {feat_sel}, {} FROM {relname}",
                target_sql_expr(&quote_identifier(t), target_mode)
            ),
            None => format!(
                "SELECT (row_number() OVER (ORDER BY ctid))::bigint, {feat_sel} FROM {relname}"
            ),
        };

        let mut row_ids: Vec<i64> = Vec::new();
        let mut num_x: Vec<Option<f64>> = Vec::new();
        let mut cat_raw: Vec<Vec<Option<String>>> = vec![Vec::new(); cat_positions.len()];
        let mut y: Vec<Option<f64>> = Vec::new();
        let mut y_text: Vec<Option<String>> = Vec::new();

        for row in client.select(&data_q, None, &[])? {
            let rid = row.get::<i64>(1)?.unwrap();
            row_ids.push(rid);
            for (j, f) in feat_cols.iter().enumerate() {
                if f.kind == ColKind::Categorical {
                    let ci = cat_positions.iter().position(|&p| p == j).unwrap();
                    cat_raw[ci].push(row.get::<String>(2 + j)?);
                    num_x.push(None);
                } else {
                    num_x.push(row.get::<f64>(2 + j)?);
                }
            }
            if target.is_some() {
                if target_mode == TargetMode::TextRaw {
                    y_text.push(row.get::<String>(2 + nf)?);
                    y.push(None);
                } else {
                    y_text.push(None);
                    y.push(row.get::<f64>(2 + nf)?);
                }
            } else {
                y_text.push(None);
                y.push(None);
            }
        }
        let n = row_ids.len();

        let target_known = |i: usize| -> bool {
            if target.is_none() {
                true
            } else if target_mode == TargetMode::TextRaw {
                y_text[i].is_some()
            } else {
                y[i].is_some()
            }
        };
        let sup_rows: Vec<usize> = (0..n).filter(|&i| target_known(i)).collect();

        for (ci, &j) in cat_positions.iter().enumerate() {
            let raw = &cat_raw[ci];
            let mut vocab: Vec<String> = sup_rows.iter().filter_map(|&i| raw[i].clone()).collect();
            vocab.sort();
            vocab.dedup();
            let unseen_code = vocab.len() as f64;
            let mode_code = if vocab.is_empty() {
                unseen_code
            } else {
                let mut best = 0usize;
                let mut best_n = 0usize;
                for (vi, v) in vocab.iter().enumerate() {
                    let c = sup_rows
                        .iter()
                        .filter(|&&i| raw[i].as_deref() == Some(v.as_str()))
                        .count();
                    if c > best_n {
                        best_n = c;
                        best = vi;
                    }
                }
                best as f64
            };
            for i in 0..n {
                let code = match &raw[i] {
                    Some(s) => vocab
                        .iter()
                        .position(|v| v == s)
                        .map(|p| p as f64)
                        .unwrap_or(unseen_code),
                    None => mode_code,
                };
                num_x[i * nf + j] = Some(code);
            }
        }

        for (j, fc) in feat_cols.iter().enumerate() {
            if fc.kind == ColKind::Categorical {
                continue;
            }
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for &i in &sup_rows {
                if let Some(v) = num_x[i * nf + j] {
                    sum += v;
                    cnt += 1;
                }
            }
            let missing_total = (0..n).filter(|&i| num_x[i * nf + j].is_none()).count();
            if missing_total > 0 {
                let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
                for i in 0..n {
                    if num_x[i * nf + j].is_none() {
                        num_x[i * nf + j] = Some(mean);
                    }
                }
                notes.push(format!(
                    "{}: imputed {missing_total} missing value(s) (mean={mean:.4})",
                    fc.name
                ));
            }
        }

        let x: Vec<f64> = num_x.into_iter().map(|v| v.unwrap_or(0.0)).collect();

        if let Some(t) = target.filter(|_| target_mode == TargetMode::TextRaw) {
            let sup_strs: Vec<&str> = sup_rows
                .iter()
                .filter_map(|&i| y_text[i].as_deref())
                .collect();
            let distinct_sup: Vec<String> = {
                let mut v: Vec<String> = sup_strs.iter().map(|s| s.to_string()).collect();
                v.sort();
                v.dedup();
                v
            };
            if is_bool_like_text(&distinct_sup) {
                target_is_int = true;
                for i in 0..n {
                    y[i] = y_text[i].as_deref().and_then(bool_text_to_f64);
                }
                notes.push(format!("target \"{t}\": text -> boolean"));
            } else if !sup_strs.is_empty() && sup_strs.iter().all(|s| parse_finite_f64(s).is_some())
            {
                target_is_int = false;
                for i in 0..n {
                    y[i] = y_text[i].as_deref().and_then(parse_finite_f64);
                }
                notes.push(format!(
                    "target \"{t}\": numeric-parseable text -> regression target"
                ));
            } else {
                let caps = read_caps();
                if distinct_sup.len() > caps.max_classes {
                    error!(
                        "pg_fela: target column \"{t}\" is text, not numeric, and has {} distinct \
                         labels, which exceeds max_classes={}: cannot classify (too many classes), \
                         and cannot regress either (non-numeric text does not parse as a number)",
                        distinct_sup.len(),
                        caps.max_classes
                    );
                }
                target_is_int = true;
                for i in 0..n {
                    y[i] = y_text[i]
                        .as_deref()
                        .and_then(|s| distinct_sup.iter().position(|v| v == s))
                        .map(|p| p as f64);
                }
                let mapping = distinct_sup
                    .iter()
                    .enumerate()
                    .map(|(code, label)| format!("{code}={label}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                notes.push(format!(
                    "target \"{t}\": text -> {} classes [{mapping}]",
                    distinct_sup.len()
                ));
            }
        }

        if !notes.is_empty() {
            notice!("pg_fela: auto-encoded in {relname}: {}", notes.join("; "));
        }

        let feature_kinds: Vec<ColKind> = feat_cols.iter().map(|f| f.kind).collect();
        Ok(TableData {
            features: feat_cols.into_iter().map(|f| f.name).collect(),
            feature_kinds,
            row_ids,
            x,
            y,
            n,
            nf,
            target_is_int,
            encoding_notes: notes,
        })
    });
    res.unwrap_or_else(|e| error!("pg_fela: SPI error: {e}"))
}

fn support_query_idx(td: &TableData) -> (Vec<usize>, Vec<usize>) {
    let mut sup = Vec::new();
    let mut qry = Vec::new();
    for i in 0..td.n {
        if td.y[i].is_some() {
            sup.push(i);
        } else {
            qry.push(i);
        }
    }
    (sup, qry)
}

fn gather_x(td: &TableData, idx: &[usize]) -> Vec<f32> {
    let mut v = Vec::with_capacity(idx.len() * td.nf);
    for &i in idx {
        for c in 0..td.nf {
            v.push(td.x[i * td.nf + c] as f32);
        }
    }
    v
}

fn label_encode(td: &TableData, sup: &[usize]) -> Vec<f64> {
    let mut vals: Vec<f64> = sup.iter().map(|&i| td.y[i].unwrap()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup();
    vals
}

fn is_classification(td: &TableData, sup: &[usize], caps: &Caps) -> bool {
    let classes = label_encode(td, sup);
    td.target_is_int && classes.len() >= 2 && classes.len() <= caps.max_classes
}

fn require_support(sup: &[usize]) {
    if sup.is_empty() {
        error!("pg_fela: no labeled (non-NULL target) rows to learn from");
    }
}

const MAX_PRECISION_PROBE: usize = 6;

fn decimal_places(v: f64) -> usize {
    if !v.is_finite() {
        return 0;
    }
    let eps = (v.abs() * 1e-9).max(1e-9);
    for d in 0..=MAX_PRECISION_PROBE {
        let scale = 10f64.powi(d as i32);
        let rounded = (v * scale).round() / scale;
        if (rounded - v).abs() <= eps {
            return d;
        }
    }
    MAX_PRECISION_PROBE
}

fn target_precision(td: &TableData, sup: &[usize]) -> usize {
    if td.target_is_int {
        return 0;
    }
    sup.iter()
        .filter_map(|&i| td.y[i])
        .map(decimal_places)
        .max()
        .unwrap_or(0)
}

fn round_to(v: f64, decimals: usize) -> f64 {
    let scale = 10f64.powi(decimals as i32);
    (v * scale).round() / scale
}

fn regress_precision_for(tbl: &str, target: &str) -> (bool, usize) {
    let td = load_table(tbl, Some(target));
    let (sup, _qry) = support_query_idx(&td);
    if sup.is_empty() {
        return (false, 0);
    }
    let caps = read_caps();
    if is_classification(&td, &sup, &caps) {
        (false, 0)
    } else {
        (true, target_precision(&td, &sup))
    }
}

fn trust_label_str(trust: f64, ood: bool) -> &'static str {
    if ood || trust < 0.34 {
        "low trust"
    } else if trust < 0.67 {
        "check"
    } else {
        "trusted"
    }
}

#[pg_extern(stable, parallel_restricted)]
fn fela_detect_task(tbl: &str, target: &str) -> String {
    let td = load_table(tbl, Some(target));
    let (sup, _q) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    if is_classification(&td, &sup, &caps) {
        format!(
            "classify ({} classes, {} labeled rows, {} features)",
            label_encode(&td, &sup).len(),
            sup.len(),
            td.nf
        )
    } else {
        format!("regress ({} labeled rows, {} features)", sup.len(), td.nf)
    }
}

#[pg_extern(stable, parallel_restricted)]
fn fela_automl(
    tbl: &str,
    target: &str,
) -> TableIterator<
    'static,
    (
        name!(row_id, i64),
        name!(prediction, f64),
        name!(confidence, f64),
        name!(task, String),
    ),
> {
    let td = load_table(tbl, Some(target));
    let (sup, qry) = support_query_idx(&td);
    require_support(&sup);
    if qry.is_empty() {
        return TableIterator::new(Vec::new());
    }
    let caps = read_caps();
    let sx = gather_x(&td, &sup);
    let qx = gather_x(&td, &qry);
    let mut out: Vec<(i64, f64, f64, String)> = Vec::new();

    if is_classification(&td, &sup, &caps) {
        let classes = label_encode(&td, &sup);
        let sy: Vec<f32> = sup
            .iter()
            .map(|&i| encode_one(&classes, td.y[i].unwrap()) as f32)
            .collect();
        let probs = with_model(|m| infer_cls(m, &sx, &sy, &qx, td.nf, classes.len()));
        for (k, &qi) in qry.iter().enumerate() {
            let (cls, p) = argmax_prob(&probs[k]);
            out.push((td.row_ids[qi], classes[cls], p as f64, "classify".into()));
        }
    } else {
        let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
        let preds = gbm_reg(&sx, &sy, &qx, td.nf);
        let trust_ood: Vec<(f64, bool)> = if sup.len() >= 2 {
            compute_trust_ood(&td, &sup, &qry)
        } else {
            qry.iter().map(|_| (0.5, false)).collect()
        };
        let prec = target_precision(&td, &sup);
        for (k, &qi) in qry.iter().enumerate() {
            let (mean, _std) = preds[k];
            let (trust, _ood) = trust_ood[k];
            out.push((
                td.row_ids[qi],
                round_to(mean as f64, prec),
                trust,
                "regress".into(),
            ));
        }
    }
    TableIterator::new(out)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_impute(
    tbl: &str,
    target: &str,
) -> TableIterator<'static, (name!(row_id, i64), name!(imputed, f64))> {
    let rows: Vec<(i64, f64)> = fela_automl(tbl, target)
        .map(|(rid, pred, _c, _t)| (rid, pred))
        .collect();
    TableIterator::new(rows)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_anomaly(
    tbl: &str,
    target: &str,
) -> TableIterator<
    'static,
    (
        name!(row_id, i64),
        name!(score, f64),
        name!(is_outlier, bool),
    ),
> {
    let td = load_table(tbl, Some(target));
    let (sup, _q) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    let sx = gather_x(&td, &sup);
    let mut out: Vec<(i64, f64, bool)> = Vec::new();

    if is_classification(&td, &sup, &caps) {
        let classes = label_encode(&td, &sup);
        let sy: Vec<f32> = sup
            .iter()
            .map(|&i| encode_one(&classes, td.y[i].unwrap()) as f32)
            .collect();
        let probs = with_model(|m| infer_cls(m, &sx, &sy, &sx, td.nf, classes.len()));
        for (k, &si) in sup.iter().enumerate() {
            let true_c = encode_one(&classes, td.y[si].unwrap());
            let p_true = probs[k].get(true_c).copied().unwrap_or(0.0) as f64;
            let score = 1.0 - p_true;
            out.push((td.row_ids[si], score, score > 0.5));
        }
    } else {
        let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
        let preds = gbm_reg(&sx, &sy, &sx, td.nf);
        let resid: Vec<f64> = sup
            .iter()
            .enumerate()
            .map(|(k, &si)| (td.y[si].unwrap() - preds[k].0 as f64).abs())
            .collect();
        let mu = resid.iter().sum::<f64>() / resid.len().max(1) as f64;
        let var =
            resid.iter().map(|r| (r - mu) * (r - mu)).sum::<f64>() / resid.len().max(1) as f64;
        let sd = var.sqrt().max(1e-9);
        for (k, &si) in sup.iter().enumerate() {
            let z = (resid[k] - mu) / sd;
            out.push((td.row_ids[si], z, z > 2.0));
        }
    }
    TableIterator::new(out)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_cluster(
    tbl: &str,
    k: i32,
) -> TableIterator<'static, (name!(row_id, i64), name!(cluster, i32))> {
    cluster_table(&load_table(tbl, None), k)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_cluster_ex(
    tbl: &str,
    k: i32,
    exclude: &str,
) -> TableIterator<'static, (name!(row_id, i64), name!(cluster, i32))> {
    cluster_table(&load_table(tbl, Some(exclude)), k)
}

fn cluster_table(
    td: &TableData,
    k: i32,
) -> TableIterator<'static, (name!(row_id, i64), name!(cluster, i32))> {
    if k < 1 {
        error!("pg_fela: k must be >= 1");
    }
    if td.n == 0 {
        return TableIterator::new(Vec::new());
    }
    let (mean, scale) = col_stats(&td.x, td.n, td.nf);
    let mut z = vec![0f64; td.n * td.nf];
    for r in 0..td.n {
        for c in 0..td.nf {
            z[r * td.nf + c] = (td.x[r * td.nf + c] - mean[c]) / scale[c];
        }
    }
    let assign = cluster_labels(&z, td.n, td.nf, (k as usize).min(td.n));
    let out: Vec<(i64, i32)> = (0..td.n)
        .map(|r| (td.row_ids[r], assign[r] as i32))
        .collect();
    TableIterator::new(out)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_importance(
    tbl: &str,
    target: &str,
) -> TableIterator<'static, (name!(feature, String), name!(importance, f64))> {
    let (names, imps) = compute_importance(tbl, target);
    let out: Vec<(String, f64)> = names.into_iter().zip(imps).collect();
    TableIterator::new(out)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_explain(tbl: &str, target: &str) -> String {
    let td = load_table(tbl, Some(target));
    let (sup, qry) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    let cls = is_classification(&td, &sup, &caps);
    let (names, imps) = compute_importance(tbl, target);
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by(|&a, &b| imps[b].partial_cmp(&imps[a]).unwrap());

    let head = if cls {
        format!(
            "Predicting \"{target}\" is a CLASSIFICATION task with {} classes, learned in-context from {} labeled rows ({} to predict).",
            label_encode(&td, &sup).len(), sup.len(), qry.len()
        )
    } else {
        format!(
            "Predicting \"{target}\" is a REGRESSION task (gradient-boosted, fit on the labeled rows), learned from {} labeled rows ({} to predict).",
            sup.len(), qry.len()
        )
    };
    let encoded_suffix = if td.encoding_notes.is_empty() {
        String::new()
    } else {
        format!(" Auto-encoded: {}.", td.encoding_notes.join("; "))
    };
    let regress_suffix = if cls {
        String::new()
    } else {
        regress_explain_suffix(tbl, target, &td, &sup, &qry)
    };
    if order.is_empty() || imps[order[0]] <= 0.0 {
        return format!(
            "{head} No single feature stood out (every feature's permutation importance was <= 0 on this data).{regress_suffix}{encoded_suffix}"
        );
    }
    let top = &names[order[0]];
    let mut story = format!(
        "{head} {} mattered most (permutation importance {:.3})",
        top, imps[order[0]]
    );
    if order.len() > 1 && imps[order[1]] > 0.0 {
        story.push_str(&format!(
            ", followed by {} ({:.3})",
            names[order[1]], imps[order[1]]
        ));
    }
    story.push('.');
    story.push_str(&regress_suffix);
    story.push_str(&encoded_suffix);
    story
}

fn regress_explain_suffix(
    tbl: &str,
    target: &str,
    td: &TableData,
    sup: &[usize],
    qry: &[usize],
) -> String {
    if qry.is_empty() {
        return String::new();
    }
    let prec = target_precision(td, sup);
    let sx = gather_x(td, sup);
    let qx = gather_x(td, qry);
    let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
    let preds = gbm_reg(&sx, &sy, &qx, td.nf);
    let val = round_to(preds[0].0 as f64, prec);
    let example_rid = td.row_ids[qry[0]];

    let (trust, ood) = if sup.len() >= 2 {
        compute_trust_ood(td, sup, qry)[0]
    } else {
        (0.5, false)
    };
    let label = trust_label_str(trust, ood);
    let reason = if label == "low trust" {
        " (out of distribution - unlike your rows)"
    } else {
        ""
    };

    if sup.len() >= 4 {
        let bands: Vec<(i64, f64, f64, f64)> = fela_conformal_regress(tbl, target, 0.8).collect();
        if let Some(&(_, _, lo, hi)) = bands.iter().find(|(rid, ..)| *rid == example_rid) {
            let band = round_to((hi - lo) / 2.0, prec);
            return format!(
                " For example, row {example_rid}: {val}, usually within \u{00b1}{band}, {label}{reason}."
            );
        }
    }
    format!(" For example, row {example_rid}: {val}, {label}{reason}.")
}

fn feature_baseline(td: &TableData, sup: &[usize], j: usize) -> f64 {
    if td.feature_kinds[j] == ColKind::Categorical {
        let mut counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for &i in sup {
            let code = td.x[i * td.nf + j].round() as i64;
            *counts.entry(code).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(code, _)| code as f64)
            .unwrap_or(0.0)
    } else {
        let sum: f64 = sup.iter().map(|&i| td.x[i * td.nf + j]).sum();
        sum / sup.len().max(1) as f64
    }
}

#[pg_extern(stable, parallel_restricted)]
fn fela_explain_row(
    tbl: &str,
    target: &str,
    row_id: i64,
) -> TableIterator<
    'static,
    (
        name!(feature, String),
        name!(contribution, f64),
        name!(direction, String),
        name!(value, f64),
    ),
> {
    let td = load_table(tbl, Some(target));
    let (sup, _qry) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    let nf = td.nf;

    let ri = match td.row_ids.iter().position(|&r| r == row_id) {
        Some(i) => i,
        None => error!("pg_fela: fela_explain_row: no row with row_id={row_id} in \"{tbl}\""),
    };

    let actual: Vec<f64> = (0..nf).map(|c| td.x[ri * nf + c]).collect();
    let baseline: Vec<f64> = (0..nf).map(|j| feature_baseline(&td, &sup, j)).collect();

    let mut qx: Vec<f32> = Vec::with_capacity((nf + 1) * nf);
    qx.extend(actual.iter().map(|&v| v as f32));
    for j in 0..nf {
        let mut row = actual.clone();
        row[j] = baseline[j];
        qx.extend(row.iter().map(|&v| v as f32));
    }
    let sx = gather_x(&td, &sup);

    let mut contrib = vec![0f64; nf];
    let classify = is_classification(&td, &sup, &caps);
    if classify {
        let classes = label_encode(&td, &sup);
        let sy: Vec<f32> = sup
            .iter()
            .map(|&i| encode_one(&classes, td.y[i].unwrap()) as f32)
            .collect();
        let probs = with_model(|m| infer_cls(m, &sx, &sy, &qx, nf, classes.len()));
        let (cls_idx, _) = argmax_prob(&probs[0]);
        let p_actual = probs[0][cls_idx] as f64;
        for j in 0..nf {
            contrib[j] = p_actual - probs[1 + j][cls_idx] as f64;
        }
    } else {
        let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
        let preds = gbm_reg(&sx, &sy, &qx, nf);
        let pred_actual = preds[0].0 as f64;
        for j in 0..nf {
            contrib[j] = pred_actual - preds[1 + j].0 as f64;
        }
    }

    let mut order: Vec<usize> = (0..nf).collect();
    order.sort_by(|&a, &b| contrib[b].abs().partial_cmp(&contrib[a].abs()).unwrap());

    let (pos_word, neg_word) = if classify {
        ("toward", "away")
    } else {
        ("increases", "decreases")
    };
    let out: Vec<(String, f64, String, f64)> = order
        .into_iter()
        .map(|j| {
            let dir = if contrib[j] >= 0.0 {
                pos_word
            } else {
                neg_word
            };
            (
                td.features[j].clone(),
                contrib[j],
                dir.to_string(),
                actual[j],
            )
        })
        .collect();
    TableIterator::new(out)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_conformal_regress(
    tbl: &str,
    target: &str,
    coverage: f64,
) -> TableIterator<
    'static,
    (
        name!(row_id, i64),
        name!(prediction, f64),
        name!(lo, f64),
        name!(hi, f64),
    ),
> {
    if !(0.0..1.0).contains(&coverage) {
        error!("pg_fela: coverage must be in (0,1)");
    }
    let td = load_table(tbl, Some(target));
    let (sup, qry) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    if is_classification(&td, &sup, &caps) {
        error!("pg_fela: fela_conformal_regress: \"{target}\" looks categorical; use fela_conformal_threshold for classification intervals");
    }
    if sup.len() < 4 {
        error!("pg_fela: need >= 4 labeled rows for split-conformal");
    }
    let ctx: Vec<usize> = sup.iter().copied().step_by(2).collect();
    let cal: Vec<usize> = sup.iter().copied().skip(1).step_by(2).collect();
    let cx = gather_x(&td, &ctx);
    let cy: Vec<f32> = ctx.iter().map(|&i| td.y[i].unwrap() as f32).collect();
    let calx = gather_x(&td, &cal);
    let preds = gbm_reg(&cx, &cy, &calx, td.nf);
    let mut resid: Vec<f64> = cal
        .iter()
        .enumerate()
        .map(|(k, &ci)| (td.y[ci].unwrap() - preds[k].0 as f64).abs())
        .collect();
    resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = quantile_sorted(&resid, coverage);
    let sx = gather_x(&td, &sup);
    let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
    let qx = gather_x(&td, &qry);
    let mut out: Vec<(i64, f64, f64, f64)> = Vec::new();
    if !qry.is_empty() {
        let preds = gbm_reg(&sx, &sy, &qx, td.nf);
        for (k, &qi) in qry.iter().enumerate() {
            let mean = preds[k].0 as f64;
            out.push((td.row_ids[qi], mean, mean - q, mean + q));
        }
    }
    TableIterator::new(out)
}

#[pg_extern(stable, parallel_restricted)]
fn fela_conformal_threshold(tbl: &str, target: &str, coverage: f64) -> f64 {
    if !(0.0..1.0).contains(&coverage) {
        error!("pg_fela: coverage must be in (0,1)");
    }
    let td = load_table(tbl, Some(target));
    let (sup, _q) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    if !is_classification(&td, &sup, &caps) {
        error!("pg_fela: fela_conformal_threshold is for classification targets");
    }
    if sup.len() < 4 {
        error!("pg_fela: need >= 4 labeled rows for split-conformal");
    }
    let classes = label_encode(&td, &sup);
    let ctx: Vec<usize> = sup.iter().copied().step_by(2).collect();
    let cal: Vec<usize> = sup.iter().copied().skip(1).step_by(2).collect();
    let cx = gather_x(&td, &ctx);
    let cy: Vec<f32> = ctx
        .iter()
        .map(|&i| encode_one(&classes, td.y[i].unwrap()) as f32)
        .collect();
    let calx = gather_x(&td, &cal);
    with_model(|m| {
        let probs = infer_cls(m, &cx, &cy, &calx, td.nf, classes.len())?;
        let mut nc: Vec<f64> = cal
            .iter()
            .enumerate()
            .map(|(k, &ci)| {
                let tc = encode_one(&classes, td.y[ci].unwrap());
                1.0 - probs[k].get(tc).copied().unwrap_or(0.0) as f64
            })
            .collect();
        nc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tau = quantile_sorted(&nc, coverage);
        Ok((1.0 - tau).clamp(0.0, 1.0))
    })
}

fn standardize(x: &[f64], n: usize, d: usize, mean: &[f64], scale: &[f64]) -> Vec<f64> {
    let mut z = vec![0f64; n * d];
    for r in 0..n {
        for c in 0..d {
            z[r * d + c] = (x[r * d + c] - mean[c]) / scale[c];
        }
    }
    z
}

fn knn_mean_dist_loo(z: &[f64], n: usize, d: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0f64; n];
    for i in 0..n {
        let mut ds: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let s: f64 = (0..d).map(|c| (z[i * d + c] - z[j * d + c]).powi(2)).sum();
                s.sqrt()
            })
            .collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let kk = k.clamp(1, ds.len().max(1));
        out[i] = ds[..kk].iter().sum::<f64>() / kk as f64;
    }
    out
}

fn knn_mean_dist_to_set(zq: &[f64], zset: &[f64], n_set: usize, d: usize, k: usize) -> f64 {
    let mut ds: Vec<f64> = (0..n_set)
        .map(|j| {
            let s: f64 = (0..d).map(|c| (zq[c] - zset[j * d + c]).powi(2)).sum();
            s.sqrt()
        })
        .collect();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let kk = k.clamp(1, ds.len().max(1));
    ds[..kk].iter().sum::<f64>() / kk as f64
}

const OOD_Z_THRESHOLD: f64 = 2.0;

fn compute_trust_ood(td: &TableData, sup: &[usize], qry: &[usize]) -> Vec<(f64, bool)> {
    let nf = td.nf;
    let sup_x64: Vec<f64> = sup
        .iter()
        .flat_map(|&i| td.x[i * nf..(i + 1) * nf].iter().copied())
        .collect();
    let qry_x64: Vec<f64> = qry
        .iter()
        .flat_map(|&i| td.x[i * nf..(i + 1) * nf].iter().copied())
        .collect();
    let (mean, scale) = col_stats(&sup_x64, sup.len(), nf);
    let sz = standardize(&sup_x64, sup.len(), nf, &mean, &scale);
    let qz = standardize(&qry_x64, qry.len(), nf, &mean, &scale);

    let k = (sup.len() - 1).clamp(1, 5);
    let ref_dist = knn_mean_dist_loo(&sz, sup.len(), nf, k);
    let ref_mu = ref_dist.iter().sum::<f64>() / ref_dist.len().max(1) as f64;
    let ref_var = ref_dist
        .iter()
        .map(|d| (d - ref_mu) * (d - ref_mu))
        .sum::<f64>()
        / ref_dist.len().max(1) as f64;
    let ref_sd = ref_var.sqrt().max(1e-9);
    let ref_max = ref_dist.iter().cloned().fold(f64::MIN, f64::max);

    (0..qry.len())
        .map(|qk| {
            let dist = knn_mean_dist_to_set(&qz[qk * nf..(qk + 1) * nf], &sz, sup.len(), nf, k);
            let z = (dist - ref_mu) / ref_sd;
            let ood = dist > ref_max || z > OOD_Z_THRESHOLD;
            let trust = (1.0 - (z / OOD_Z_THRESHOLD).max(0.0)).clamp(0.0, 1.0);
            (trust, ood)
        })
        .collect()
}

#[pg_extern(stable, parallel_restricted)]
fn fela_predict_trust(
    tbl: &str,
    target: &str,
) -> TableIterator<
    'static,
    (
        name!(row_id, i64),
        name!(prediction, f64),
        name!(confidence, f64),
        name!(trust, f64),
        name!(ood, bool),
        name!(task, String),
    ),
> {
    let td = load_table(tbl, Some(target));
    let (sup, qry) = support_query_idx(&td);
    require_support(&sup);
    if sup.len() < 2 {
        error!("pg_fela: fela_predict_trust: need >= 2 labeled rows to build an OOD reference");
    }
    if qry.is_empty() {
        return TableIterator::new(Vec::new());
    }
    let caps = read_caps();
    let nf = td.nf;
    let sx = gather_x(&td, &sup);
    let qx = gather_x(&td, &qry);

    let trust_ood = compute_trust_ood(&td, &sup, &qry);

    let mut out: Vec<(i64, f64, f64, f64, bool, String)> = Vec::new();
    if is_classification(&td, &sup, &caps) {
        let classes = label_encode(&td, &sup);
        let sy: Vec<f32> = sup
            .iter()
            .map(|&i| encode_one(&classes, td.y[i].unwrap()) as f32)
            .collect();
        let probs = with_model(|m| infer_cls(m, &sx, &sy, &qx, nf, classes.len()));
        for (qk, &qi) in qry.iter().enumerate() {
            let (cls, p) = argmax_prob(&probs[qk]);
            let (trust, ood) = trust_ood[qk];
            out.push((
                td.row_ids[qi],
                classes[cls],
                p as f64,
                trust,
                ood,
                "classify".into(),
            ));
        }
    } else {
        let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
        let preds = gbm_reg(&sx, &sy, &qx, nf);
        for (qk, &qi) in qry.iter().enumerate() {
            let (mean_p, _std_p) = preds[qk];
            let (trust, ood) = trust_ood[qk];
            out.push((
                td.row_ids[qi],
                mean_p as f64,
                trust,
                trust,
                ood,
                "regress".into(),
            ));
        }
    }
    TableIterator::new(out)
}

fn check_nfeat(n_feat: i32) -> usize {
    if n_feat <= 0 {
        error!("pg_fela: n_feat must be > 0");
    }
    n_feat as usize
}
fn check_dims(n_feat: i32, n_class: i32) -> (usize, usize) {
    let nf = check_nfeat(n_feat);
    if n_class <= 0 {
        error!("pg_fela: n_class must be > 0");
    }
    (nf, n_class as usize)
}
fn split_counts(qlen: usize, slen: usize, nf: usize) -> (usize, usize) {
    if !qlen.is_multiple_of(nf) {
        error!("pg_fela: len(query_feats)={qlen} not divisible by n_feat={nf}");
    }
    if !slen.is_multiple_of(nf) {
        error!("pg_fela: len(support_feats)={slen} not divisible by n_feat={nf}");
    }
    let nq = qlen / nf;
    let ns = slen / nf;
    if ns < 1 {
        error!("pg_fela: need >= 1 support row");
    }
    if nq < 1 {
        error!("pg_fela: need >= 1 query row");
    }
    (nq, ns)
}

fn argmax_prob(p: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in p.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    (best, bv)
}

fn encode_one(classes: &[f64], v: f64) -> usize {
    classes.iter().position(|&c| c == v).unwrap_or(0)
}

fn col_stats(x: &[f64], n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0f64; d];
    for r in 0..n {
        for c in 0..d {
            mean[c] += x[r * d + c];
        }
    }
    for m in mean.iter_mut() {
        *m /= n.max(1) as f64;
    }
    let mut scale = vec![0f64; d];
    for r in 0..n {
        for c in 0..d {
            let dd = x[r * d + c] - mean[c];
            scale[c] += dd * dd;
        }
    }
    for s in scale.iter_mut() {
        let sd = (*s / n.max(1) as f64).sqrt();
        *s = if sd == 0.0 { 1.0 } else { sd };
    }
    (mean, scale)
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn cluster_labels(z: &[f64], n: usize, d: usize, k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        for c in 0..d {
            match z[a * d + c]
                .partial_cmp(&z[b * d + c])
                .unwrap_or(std::cmp::Ordering::Equal)
            {
                std::cmp::Ordering::Equal => continue,
                o => return o,
            }
        }
        a.cmp(&b)
    });
    let mut zs = vec![0f64; n * d];
    for (newi, &oi) in order.iter().enumerate() {
        zs[newi * d..newi * d + d].copy_from_slice(&z[oi * d..oi * d + d]);
    }
    let lab_sorted = kmeans(&zs, n, d, k, 100);
    let mut out = vec![0usize; n];
    for (newi, &oi) in order.iter().enumerate() {
        out[oi] = lab_sorted[newi];
    }
    out
}

fn kmeans(z: &[f64], n: usize, d: usize, k: usize, iters: usize) -> Vec<usize> {
    let k = k.max(1).min(n);
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let row = |i: usize| &z[i * d..i * d + d];
    let dist2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();

    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = (next() * n as f64) as usize % n;
    centers.push(row(first).to_vec());
    while centers.len() < k {
        let d2: Vec<f64> = (0..n)
            .map(|i| {
                centers
                    .iter()
                    .map(|c| dist2(row(i), c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let sum: f64 = d2.iter().sum();
        if sum <= 0.0 {
            centers.push(row((next() * n as f64) as usize % n).to_vec());
            continue;
        }
        let mut target = next() * sum;
        let mut chosen = n - 1;
        for (i, &w) in d2.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.push(row(chosen).to_vec());
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let mut best = 0;
            let mut bd = f64::INFINITY;
            for (c, ctr) in centers.iter().enumerate() {
                let dd = dist2(row(i), ctr);
                if dd < bd {
                    bd = dd;
                    best = c;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        let mut sums = vec![vec![0f64; d]; k];
        let mut cnt = vec![0usize; k];
        for i in 0..n {
            let a = assign[i];
            cnt[a] += 1;
            for c in 0..d {
                sums[a][c] += z[i * d + c];
            }
        }
        for c in 0..k {
            if cnt[c] > 0 {
                for x in 0..d {
                    centers[c][x] = sums[c][x] / cnt[c] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    assign
}

fn compute_importance(tbl: &str, target: &str) -> (Vec<String>, Vec<f64>) {
    let td = load_table(tbl, Some(target));
    let (sup, _q) = support_query_idx(&td);
    require_support(&sup);
    let caps = read_caps();
    let cls = is_classification(&td, &sup, &caps);
    let sx = gather_x(&td, &sup);
    let ns = sup.len();
    let nf = td.nf;

    let perm: Vec<usize> = (0..ns).rev().collect();

    let names = td.features.clone();
    let imps = if cls {
        with_model(|m| {
            let classes = label_encode(&td, &sup);
            let sy: Vec<f32> = sup
                .iter()
                .map(|&i| encode_one(&classes, td.y[i].unwrap()) as f32)
                .collect();
            let base_probs = infer_cls(m, &sx, &sy, &sx, nf, classes.len())?;
            let base_acc = accuracy(&base_probs, &sy);
            let mut out = vec![0f64; nf];
            for j in 0..nf {
                let mut qx = sx.clone();
                for r in 0..ns {
                    qx[r * nf + j] = sx[perm[r] * nf + j];
                }
                let probs = infer_cls(m, &sx, &sy, &qx, nf, classes.len())?;
                let acc = accuracy(&probs, &sy);
                out[j] = (base_acc - acc).max(0.0);
            }
            Ok(out)
        })
    } else {
        let sy: Vec<f32> = sup.iter().map(|&i| td.y[i].unwrap() as f32).collect();
        let base = gbm_reg(&sx, &sy, &sx, nf);
        let base_mse = mse(&base, &sy);
        let mut out = vec![0f64; nf];
        for j in 0..nf {
            let mut qx = sx.clone();
            for r in 0..ns {
                qx[r * nf + j] = sx[perm[r] * nf + j];
            }
            let pr = gbm_reg(&sx, &sy, &qx, nf);
            let m2 = mse(&pr, &sy);
            out[j] = ((m2 - base_mse) / (base_mse + 1e-9)).max(0.0);
        }
        out
    };
    (names, imps)
}

fn accuracy(probs: &[Vec<f32>], y: &[f32]) -> f64 {
    let mut correct = 0usize;
    for (p, &yc) in probs.iter().zip(y) {
        if argmax_prob(p).0 == yc as usize {
            correct += 1;
        }
    }
    correct as f64 / y.len().max(1) as f64
}
fn mse(preds: &[(f32, f32)], y: &[f32]) -> f64 {
    let mut s = 0f64;
    for (p, &yv) in preds.iter().zip(y) {
        let e = p.0 as f64 - yv as f64;
        s += e * e;
    }
    s / y.len().max(1) as f64
}

const WINDOW_SEEK_HEAD: std::ffi::c_int = 1;

extern "C" {
    fn WinGetPartitionRowCount(winobj: *mut pgrx::pg_sys::WindowObjectData) -> i64;
    fn WinGetCurrentPosition(winobj: *mut pgrx::pg_sys::WindowObjectData) -> i64;
    fn WinGetPartitionLocalMemory(
        winobj: *mut pgrx::pg_sys::WindowObjectData,
        sz: usize,
    ) -> *mut std::ffi::c_void;
    fn WinGetFuncArgInPartition(
        winobj: *mut pgrx::pg_sys::WindowObjectData,
        argno: std::ffi::c_int,
        relpos: std::ffi::c_int,
        seektype: std::ffi::c_int,
        set_mark: bool,
        isnull: *mut bool,
        isout: *mut bool,
    ) -> pgrx::pg_sys::Datum;
}

unsafe fn window_read_frame(
    winobj: *mut pgrx::pg_sys::WindowObjectData,
) -> (usize, usize, Vec<f64>) {
    use pgrx::datum::FromDatum;
    let n = WinGetPartitionRowCount(winobj).max(0) as usize;
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut d = 0usize;
    for i in 0..n {
        let mut isnull = false;
        let mut isout = false;
        let datum = WinGetFuncArgInPartition(
            winobj,
            0,
            i as std::ffi::c_int,
            WINDOW_SEEK_HEAD,
            false,
            &mut isnull,
            &mut isout,
        );
        if isnull || isout {
            error!(
                "pg_fela: *_over window functions require a non-NULL feature vector in every row"
            );
        }
        let v = Vec::<f64>::from_datum(datum, false).unwrap_or_default();
        if v.is_empty() {
            error!("pg_fela: *_over got an empty feature vector");
        }
        if d == 0 {
            d = v.len();
        } else if v.len() != d {
            error!(
                "pg_fela: *_over rows have inconsistent feature counts ({} vs {})",
                v.len(),
                d
            );
        }
        rows.push(v);
    }
    let mut data = Vec::with_capacity(n * d);
    for r in &rows {
        data.extend_from_slice(r);
    }
    (n, d, data)
}

#[repr(C)]
struct WinHeader {
    computed: i64,
}

#[no_mangle]
pub extern "C" fn pg_finfo_fela_cluster_over_wf() -> &'static pgrx::pg_sys::Pg_finfo_record {
    const V1: pgrx::pg_sys::Pg_finfo_record = pgrx::pg_sys::Pg_finfo_record { api_version: 1 };
    &V1
}
#[no_mangle]
pub extern "C" fn pg_finfo_fela_anomaly_over_wf() -> &'static pgrx::pg_sys::Pg_finfo_record {
    const V1: pgrx::pg_sys::Pg_finfo_record = pgrx::pg_sys::Pg_finfo_record { api_version: 1 };
    &V1
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn fela_cluster_over_wf(
    fcinfo: pgrx::pg_sys::FunctionCallInfo,
) -> pgrx::pg_sys::Datum {
    use pgrx::datum::IntoDatum;
    let winobj = (*fcinfo).context as *mut pgrx::pg_sys::WindowObjectData;
    let n = WinGetPartitionRowCount(winobj).max(0) as usize;
    let sz = std::mem::size_of::<WinHeader>() + n * std::mem::size_of::<i32>();
    let mem = WinGetPartitionLocalMemory(winobj, sz) as *mut u8;
    let header = mem as *mut WinHeader;
    let labels = mem.add(std::mem::size_of::<WinHeader>()) as *mut i32;
    if (*header).computed == 0 {
        let (nn, d, data) = window_read_frame(winobj);
        let (mean, scale) = col_stats(&data, nn, d);
        let mut z = vec![0f64; nn * d];
        for r in 0..nn {
            for c in 0..d {
                z[r * d + c] = (data[r * d + c] - mean[c]) / scale[c];
            }
        }
        let assign = cluster_labels(&z, nn, d, 3usize.min(nn.max(1)));
        for (i, a) in assign.iter().enumerate() {
            *labels.add(i) = *a as i32;
        }
        (*header).computed = 1;
    }
    let pos = WinGetCurrentPosition(winobj).max(0) as usize;
    (*fcinfo).isnull = false;
    (*labels.add(pos)).into_datum().unwrap()
}

#[pg_guard]
#[no_mangle]
pub unsafe extern "C-unwind" fn fela_anomaly_over_wf(
    fcinfo: pgrx::pg_sys::FunctionCallInfo,
) -> pgrx::pg_sys::Datum {
    use pgrx::datum::IntoDatum;
    let winobj = (*fcinfo).context as *mut pgrx::pg_sys::WindowObjectData;
    let n = WinGetPartitionRowCount(winobj).max(0) as usize;
    let sz = std::mem::size_of::<WinHeader>() + n * std::mem::size_of::<f64>();
    let mem = WinGetPartitionLocalMemory(winobj, sz) as *mut u8;
    let header = mem as *mut WinHeader;
    let scores = mem.add(std::mem::size_of::<WinHeader>()) as *mut f64;
    if (*header).computed == 0 {
        let (nn, d, data) = window_read_frame(winobj);
        let (mean, scale) = col_stats(&data, nn, d);
        for r in 0..nn {
            let mut ss = 0f64;
            for c in 0..d {
                let zc = (data[r * d + c] - mean[c]) / scale[c];
                ss += zc * zc;
            }
            *scores.add(r) = (ss / d.max(1) as f64).sqrt();
        }
        (*header).computed = 1;
    }
    let pos = WinGetCurrentPosition(winobj).max(0) as usize;
    (*fcinfo).isnull = false;
    (*scores.add(pos)).into_datum().unwrap()
}

extension_sql!(
    r#"
CREATE FUNCTION fela_cluster_over(VARIADIC float8[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'fela_cluster_over_wf' LANGUAGE c WINDOW STRICT;
CREATE FUNCTION fela_anomaly_over(VARIADIC float8[]) RETURNS float8
    AS 'MODULE_PATHNAME', 'fela_anomaly_over_wf' LANGUAGE c WINDOW STRICT;
"#,
    name = "fela_over_window_functions",
);

#[pg_extern]
fn fela_create_view(
    tbl: &str,
    target: Option<&str>,
    add: default!(&str, "'cluster,anomaly,predict'"),
    k: default!(i32, 3),
    materialized: default!(bool, true),
) -> String {
    let (nsp, rel, pk): (String, String, Option<String>) = Spi::connect(|c| {
        let r = c
            .select(
                &format!(
                    "SELECT n.nspname::text, c.relname::text FROM pg_class c \
                     JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid = {}::regclass",
                    quote_literal(tbl)
                ),
                None,
                &[],
            )?
            .first();
        let nsp = r.get::<String>(1)?.expect("nspname");
        let rel = r.get::<String>(2)?.expect("relname");
        let mut pks: Vec<String> = Vec::new();
        for row in c.select(
            &format!(
                "SELECT a.attname::text FROM pg_index i JOIN pg_attribute a \
                 ON a.attrelid=i.indrelid AND a.attnum = ANY(i.indkey) \
                 WHERE i.indrelid = {}::regclass AND i.indisprimary",
                quote_literal(tbl)
            ),
            None,
            &[],
        )? {
            if let Some(n) = row.get::<String>(1)? {
                pks.push(n);
            }
        }
        let pk = if pks.len() == 1 {
            Some(pks.remove(0))
        } else {
            None
        };
        Ok::<_, pgrx::spi::SpiError>((nsp, rel, pk))
    })
    .unwrap_or_else(|e| error!("pg_fela: SPI error: {e}"));

    let base_ref = quote_qualified_identifier(&nsp, &rel);
    let view_ref = quote_qualified_identifier(&nsp, &format!("{rel}_ml"));
    let tlit = quote_literal(&base_ref);

    let base_cols: Vec<String> = Spi::connect(|c| {
        let mut v = Vec::new();
        for row in c.select(
            &format!(
                "SELECT a.attname::text FROM pg_attribute a WHERE a.attrelid = {}::regclass \
                 AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum",
                quote_literal(tbl)
            ),
            None,
            &[],
        )? {
            v.push(row.get::<String>(1)?.unwrap());
        }
        Ok::<_, pgrx::spi::SpiError>(v)
    })
    .unwrap_or_else(|e| error!("pg_fela: SPI error: {e}"));

    let want = |name: &str| add.split(',').map(|s| s.trim()).any(|s| s == name);
    let want_cluster = want("cluster");
    let want_anomaly = want("anomaly");
    let want_predict = want("predict") || want("impute");

    let mut ctes: Vec<String> = vec![format!(
        "base AS (SELECT b.*, (row_number() OVER (ORDER BY b.ctid))::bigint AS __rid FROM {base_ref} b)"
    )];
    let mut sel: Vec<String> = base_cols
        .iter()
        .map(|c| format!("base.{}", quote_identifier(c)))
        .collect();
    let mut joins: Vec<String> = Vec::new();

    if want_cluster {
        let call = match target {
            Some(t) => format!("fela_cluster_ex({tlit}, {k}, {})", quote_literal(t)),
            None => format!("fela_cluster({tlit}, {k})"),
        };
        ctes.push(format!("cl AS (SELECT row_id, cluster FROM {call})"));
        joins.push("LEFT JOIN cl ON cl.row_id = base.__rid".into());
        sel.push("cl.cluster AS cluster".into());
    }
    if want_anomaly {
        match target {
            Some(t) => {
                ctes.push(format!(
                    "an AS (SELECT row_id, score AS anomaly_score FROM fela_anomaly({tlit}, {}))",
                    quote_literal(t)
                ));
                joins.push("LEFT JOIN an ON an.row_id = base.__rid".into());
                sel.push("an.anomaly_score AS anomaly_score".into());
            }
            None => notice!("pg_fela: fela_create_view: 'anomaly' skipped (needs a target column)"),
        }
    }
    if want_predict {
        match target {
            Some(t) => {
                let labeled: i64 = Spi::get_one::<i64>(&format!(
                    "SELECT count(*) FROM {base_ref} WHERE {} IS NOT NULL",
                    quote_identifier(t)
                ))
                .unwrap_or_else(|e| error!("pg_fela: SPI error: {e}"))
                .unwrap_or(0);

                let (is_regress, prec) = regress_precision_for(tbl, t);

                if labeled >= 2 {
                    ctes.push(format!(
                        "pr AS (SELECT row_id, prediction AS predicted, confidence, trust, ood \
                         FROM fela_predict_trust({tlit}, {}))",
                        quote_literal(t)
                    ));
                } else {
                    notice!(
                        "pg_fela: fela_create_view: only {labeled} labeled row(s); 'predict' falls back \
                         to predicted/confidence only (trust, ood, trust_label, and band all need >= 2 \
                         labeled rows to build an OOD reference, and are NULL here)"
                    );
                    ctes.push(format!(
                        "pr AS (SELECT row_id, prediction AS predicted, confidence, \
                         NULL::float8 AS trust, NULL::bool AS ood FROM fela_automl({tlit}, {}))",
                        quote_literal(t)
                    ));
                }
                joins.push("LEFT JOIN pr ON pr.row_id = base.__rid".into());
                if is_regress {
                    sel.push(format!(
                        "round(pr.predicted::numeric, {prec})::float8 AS predicted"
                    ));
                } else {
                    sel.push("pr.predicted AS predicted".into());
                }
                sel.push("pr.confidence AS confidence".into());
                sel.push("pr.trust AS trust".into());
                sel.push("pr.ood AS ood".into());
                sel.push(
                    "(CASE WHEN pr.trust IS NULL THEN NULL \
                      WHEN pr.trust < 0.34 OR pr.ood THEN 'low trust' \
                      WHEN pr.trust < 0.67 THEN 'check' \
                      ELSE 'trusted' END) AS trust_label"
                        .into(),
                );
                if is_regress && labeled >= 4 {
                    ctes.push(format!(
                        "bd AS (SELECT row_id, (hi - prediction) AS raw_band \
                         FROM fela_conformal_regress({tlit}, {}, 0.8))",
                        quote_literal(t)
                    ));
                    joins.push("LEFT JOIN bd ON bd.row_id = base.__rid".into());
                    sel.push(format!(
                        "round(bd.raw_band::numeric, {prec})::float8 AS band"
                    ));
                } else {
                    sel.push("NULL::float8 AS band".into());
                }
            }
            None => notice!("pg_fela: fela_create_view: 'predict' skipped (needs a target column)"),
        }
    }

    let existing_kind: Option<String> = Spi::get_one::<String>(&format!(
        "SELECT c.relkind::text FROM pg_class c WHERE c.oid = to_regclass({})",
        quote_literal(&view_ref)
    ))
    .unwrap_or(None);
    match existing_kind.as_deref() {
        Some("m") => Spi::run(&format!("DROP MATERIALIZED VIEW {view_ref}")).ok(),
        Some("v") => Spi::run(&format!("DROP VIEW {view_ref}")).ok(),
        _ => None,
    };

    let kind = if materialized {
        "MATERIALIZED VIEW"
    } else {
        "VIEW"
    };
    let create_sql = format!(
        "CREATE {kind} {view_ref} AS WITH {} SELECT {} FROM base {}",
        ctes.join(", "),
        sel.join(", "),
        joins.join(" ")
    );
    Spi::run(&create_sql).unwrap_or_else(|e| error!("pg_fela: create view failed: {e}"));

    let key_note = match &pk {
        Some(p) => format!("keyed by primary key \"{p}\" (carried through)"),
        None => "no single-column primary key; rows aligned by scan ordinal".into(),
    };
    let refresh = if materialized {
        format!(" Refresh with: REFRESH MATERIALIZED VIEW {view_ref};")
    } else {
        String::new()
    };
    format!(
        "created {} {view_ref} [{}]; {key_note}.{refresh}",
        if materialized {
            "materialized view"
        } else {
            "view"
        },
        {
            let mut cols = Vec::new();
            if want_cluster {
                cols.push("cluster");
            }
            if want_anomaly && target.is_some() {
                cols.push("anomaly_score");
            }
            if want_predict && target.is_some() {
                cols.push("predicted");
                cols.push("confidence");
                cols.push("trust");
                cols.push("ood");
                cols.push("trust_label");
                cols.push("band");
            }
            cols.join(", ")
        }
    )
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    fn set_gucs() {
        Spi::run("RESET fela.model_path; RESET fela.config_path;").unwrap();
    }

    fn iris_table() {
        Spi::run(
            "CREATE TEMP TABLE flowers(id serial PRIMARY KEY, sepal_len float8, sepal_wid float8, petal_len float8, species int);
             INSERT INTO flowers(sepal_len,sepal_wid,petal_len,species) VALUES
             (5.1,3.5,1.4,0),(4.9,3.0,1.4,0),(4.7,3.2,1.3,0),
             (7.0,3.2,4.7,1),(6.4,3.2,4.5,1),(6.9,3.1,4.9,1),
             (6.3,3.3,6.0,2),(5.8,2.7,5.1,2),(7.1,3.0,5.9,2),
             (5.0,3.4,1.5,NULL),(6.5,3.0,4.6,NULL),(6.2,3.4,5.4,NULL);",
        )
        .unwrap();
    }

    #[pg_test]
    fn version_and_caps() {
        set_gucs();
        let v = crate::fela_version();
        assert!(
            v.starts_with("pg_fela 1.0") && v.contains("pgrx"),
            "got {v}"
        );
        let mf: i32 = Spi::get_one("SELECT max_features FROM fela_caps()")
            .unwrap()
            .unwrap();
        assert_eq!(mf, 100);
        let mc: i32 = Spi::get_one("SELECT max_classes FROM fela_caps()")
            .unwrap()
            .unwrap();
        assert_eq!(mc, 10);
    }

    #[pg_test]
    fn embedded_default_model_with_no_guc_set() {
        Spi::run("RESET fela.model_path; RESET fela.config_path;").unwrap();

        let mf: i32 = Spi::get_one("SELECT max_features FROM fela_caps()")
            .unwrap()
            .unwrap();
        assert_eq!(mf, 100);
        let mc: i32 = Spi::get_one("SELECT max_classes FROM fela_caps()")
            .unwrap()
            .unwrap();
        assert_eq!(mc, 10);
        let info = crate::fela_model_info();
        assert!(
            info.contains("embedded"),
            "expected fela_model_info to report the embedded source, got: {info}"
        );

        iris_table();
        assert_eq!(
            crate::fela_detect_task("flowers", "species")
                .split(' ')
                .next(),
            Some("classify")
        );
        let preds: Vec<f64> = Spi::connect(|c| {
            c.select(
                "SELECT prediction FROM fela_automl('flowers','species') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .filter_map(|r| r.get::<f64>(1).unwrap())
            .collect()
        });
        assert_eq!(
            preds.len(),
            3,
            "one prediction per NULL row (embedded model)"
        );
        let mut sorted = preds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            sorted,
            vec![0.0, 1.0, 2.0],
            "embedded-model automl predicts all three classes with no GUC set"
        );
    }

    #[pg_test]
    fn argmax_and_confidence() {
        assert_eq!(crate::fela_argmax(vec![0.1, 0.7, 0.2]), 1);
        assert!((crate::fela_confidence(vec![0.1, 0.7, 0.2]) - 0.7).abs() < 1e-12);
    }

    #[pg_test]
    fn gbm_regression_is_accurate() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE nums(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO nums(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,60) g;
             INSERT INTO nums(a,b,y) VALUES (1.0,1.0,NULL),(2.0,0.5,NULL),(3.0,2.0,NULL);",
        )
        .unwrap();
        let task: String = Spi::get_one("SELECT task FROM fela_automl('nums','y') LIMIT 1")
            .unwrap()
            .unwrap();
        assert_eq!(task, "regress");
        let preds: Vec<f64> = Spi::connect(|c| {
            c.select(
                "SELECT prediction FROM fela_automl('nums','y') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .filter_map(|r| r.get::<f64>(1).unwrap())
            .collect()
        });
        let truth = [2.0f64, 6.0, 6.0];
        assert_eq!(preds.len(), 3, "three NULL rows predicted");
        for (p, t) in preds.iter().zip(truth) {
            assert!((p - t).abs() < 1.2, "gbm regress {p} vs truth {t}");
        }
    }

    #[pg_test]
    fn automl_regress_confidence_is_per_row_trust_not_global_std() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE regconf(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO regconf(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,40) g;
             INSERT INTO regconf(a,b,y) VALUES (2.0, 1.5, NULL);        -- in-distribution query
             INSERT INTO regconf(a,b,y) VALUES (5000.0, -3000.0, NULL); -- far OOD query",
        )
        .unwrap();

        let task: String = Spi::get_one("SELECT task FROM fela_automl('regconf','y') LIMIT 1")
            .unwrap()
            .unwrap();
        assert_eq!(task, "regress");

        let confs: Vec<f64> = Spi::connect(|c| {
            c.select(
                "SELECT confidence FROM fela_automl('regconf','y') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .filter_map(|r| r.get::<f64>(1).unwrap())
            .collect()
        });
        assert_eq!(confs.len(), 2, "one confidence per NULL row");
        for c in &confs {
            assert!(
                (0.0..=1.0).contains(c),
                "regression confidence must be a per-row trust score in [0,1], got {confs:?}"
            );
        }
        assert!(
            (confs[0] - confs[1]).abs() > 0.01,
            "confidence must VARY per row (in-distribution vs. far-OOD); a constant value here \
             would mean the old global-in-sample-std bug is back: {confs:?}"
        );
        assert!(
            confs[1] < confs[0],
            "the far-OOD row must have lower trust than the in-distribution row: {confs:?}"
        );
    }

    #[pg_test]
    fn automl_regress_rounds_prediction_to_integer_target_precision() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE regint(id serial PRIMARY KEY, a float8, b float8, y int4);
             INSERT INTO regint(a,b,y) SELECT g*1.0, (g%5)*1.0, (3*g + 10)
               FROM generate_series(1,30) g;
             INSERT INTO regint(a,b,y) VALUES (31.0, 1.0, NULL);",
        )
        .unwrap();

        let task: String = Spi::get_one("SELECT task FROM fela_automl('regint','y') LIMIT 1")
            .unwrap()
            .unwrap();
        assert_eq!(
            task, "regress",
            "30 distinct integer labels exceeds max_classes; must fall back to regression"
        );

        let pred: f64 = Spi::get_one("SELECT prediction FROM fela_automl('regint','y')")
            .unwrap()
            .unwrap();
        assert_eq!(
            pred.fract(),
            0.0,
            "prediction for an integer-typed target must be rounded to an integer, got {pred}"
        );
    }

    #[pg_test]
    fn create_view_exposes_trust_label_and_band() {
        set_gucs();
        Spi::run(
            "CREATE TABLE oodband(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO oodband(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,40) g;
             INSERT INTO oodband(a,b,y) VALUES (2.0, 1.5, NULL);        -- in-distribution query
             INSERT INTO oodband(a,b,y) VALUES (5000.0, -3000.0, NULL); -- far OOD query",
        )
        .unwrap();

        let msg = crate::fela_create_view("oodband", Some("y"), "predict", 3, true);
        assert!(
            msg.contains("trust_label") && msg.contains("band"),
            "msg should advertise the new columns: {msg}"
        );

        let (indist_label, indist_band): (String, Option<f64>) = Spi::connect(|c| {
            let r = c
                .select(
                    "SELECT trust_label, band FROM oodband_ml WHERE a = 2.0 AND y IS NULL",
                    None,
                    &[],
                )?
                .first();
            Ok::<_, pgrx::spi::SpiError>((
                r.get::<String>(1)?.expect("trust_label"),
                r.get::<f64>(2)?,
            ))
        })
        .unwrap();
        assert_ne!(
            indist_label, "low trust",
            "in-distribution row should not be flagged low trust"
        );
        assert!(
            indist_band.is_some() && indist_band.unwrap() > 0.0,
            "band must be a positive typical-miss half-width, got {indist_band:?}"
        );

        let (ood_label, ood_band): (String, Option<f64>) = Spi::connect(|c| {
            let r = c
                .select(
                    "SELECT trust_label, band FROM oodband_ml WHERE a = 5000.0 AND y IS NULL",
                    None,
                    &[],
                )?
                .first();
            Ok::<_, pgrx::spi::SpiError>((
                r.get::<String>(1)?.expect("trust_label"),
                r.get::<f64>(2)?,
            ))
        })
        .unwrap();
        assert_eq!(
            ood_label, "low trust",
            "far-OOD row must be labeled 'low trust'"
        );
        assert!(
            ood_band.is_some(),
            "band should still be populated (per-row conformal half-width) for the OOD row"
        );
    }

    #[pg_test]
    fn explain_regression_uses_shared_trust_and_band_vocabulary() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE regexpl(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO regexpl(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,40) g;
             INSERT INTO regexpl(a,b,y) VALUES (2.0, 1.5, NULL);",
        )
        .unwrap();
        let msg = crate::fela_explain("regexpl", "y");
        assert!(
            msg.contains("usually within"),
            "regression explain text should mention the 'usually within +/- X' band: {msg}"
        );
        assert!(
            !msg.contains("typical miss"),
            "the conformal band is a coverage interval, not the average/typical miss: {msg}"
        );
        assert!(
            msg.contains("trusted") || msg.contains("check") || msg.contains("low trust"),
            "regression explain text should use the shared trust vocabulary: {msg}"
        );
    }

    #[pg_test]
    fn gbm_anomaly_uses_gbm() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE nums2(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO nums2(a,b,y) SELECT g*0.1, (g%7)*0.5,
                    2.0*(g*0.1)*((g%7)*0.5) - (g*0.1)*(g*0.1) + 1.0
               FROM generate_series(1,30) g;
             -- outlier: an off-grid (a,b) point with a moderately (not absurdly) wrong label.
             INSERT INTO nums2(a,b,y) VALUES (1.12, 1.42, 6.9264);",
        )
        .unwrap();
        let outlier_id: i64 = Spi::get_one("SELECT id FROM nums2 WHERE y = 6.9264")
            .unwrap()
            .unwrap();
        let inlier_id: i64 = Spi::get_one("SELECT id FROM nums2 WHERE a > 0.39 AND a < 0.41")
            .unwrap()
            .unwrap();
        let (top_id, top_score, top_flag) = Spi::get_three::<i64, f64, bool>(
            "SELECT row_id, score, is_outlier FROM fela_anomaly('nums2','y') \
             ORDER BY score DESC LIMIT 1",
        )
        .unwrap();
        assert_eq!(
            top_id,
            Some(outlier_id),
            "the injected outlier row must score highest"
        );
        assert_eq!(
            top_flag,
            Some(true),
            "the injected outlier must be flagged is_outlier"
        );
        assert!(
            top_score.unwrap_or(0.0) > 3.0,
            "expected a large standardized residual for the outlier, got {top_score:?}"
        );

        let (inlier_score, inlier_flag) = Spi::get_two::<f64, bool>(&format!(
            "SELECT score, is_outlier FROM fela_anomaly('nums2','y') WHERE row_id = {inlier_id}"
        ))
        .unwrap();
        assert_eq!(
            inlier_flag,
            Some(false),
            "a well-modeled inlier row must not be flagged is_outlier, got score {inlier_score:?}"
        );
        assert!(
            inlier_score.unwrap_or(f64::MAX) < 2.0,
            "expected a small standardized residual for the well-modeled inlier, got {inlier_score:?}"
        );
    }

    #[pg_test]
    fn gbm_conformal_covers() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE numsc(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO numsc(a,b,y) SELECT g*0.1, (g%7)*0.5,
                    2.0*(g*0.1)*((g%7)*0.5) - (g*0.1)*(g*0.1) + 1.0
               FROM generate_series(1,60) g;
             INSERT INTO numsc(a,b,y) VALUES (1.0,1.0,NULL),(2.0,0.5,NULL),(3.0,2.0,NULL);",
        )
        .unwrap();
        let rows: Vec<(f64, f64, f64)> = Spi::connect(|c| {
            c.select(
                "SELECT prediction, lo, hi FROM fela_conformal_regress('numsc','y',0.9) ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .map(|r| {
                (
                    r.get::<f64>(1).unwrap().unwrap(),
                    r.get::<f64>(2).unwrap().unwrap(),
                    r.get::<f64>(3).unwrap().unwrap(),
                )
            })
            .collect()
        });
        assert_eq!(rows.len(), 3, "three NULL rows predicted");
        let truth = [2.0f64, -1.0, 4.0];
        for ((pred, lo, hi), t) in rows.iter().zip(truth) {
            assert!(
                (pred - t).abs() < 1.2,
                "gbm conformal point estimate {pred} vs truth {t}"
            );
            assert!(lo <= hi, "interval must be well-formed: lo={lo} hi={hi}");
            assert!(
                *lo <= t && t <= *hi,
                "predicted interval [{lo},{hi}] must cover truth {t}"
            );
        }
    }

    #[pg_test]
    fn predict_trust_flags_ood_row() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE oodtbl(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO oodtbl(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,40) g;
             -- in-distribution query row: well inside the observed (a,b) grid.
             INSERT INTO oodtbl(a,b,y) VALUES (2.0, 1.5, NULL);
             -- far-OOD query row: orders of magnitude outside the support range on both features.
             INSERT INTO oodtbl(a,b,y) VALUES (5000.0, -3000.0, NULL);",
        )
        .unwrap();
        let indist_id: i64 = Spi::get_one("SELECT id FROM oodtbl WHERE a = 2.0 AND y IS NULL")
            .unwrap()
            .unwrap();
        let ood_id: i64 = Spi::get_one("SELECT id FROM oodtbl WHERE a = 5000.0 AND y IS NULL")
            .unwrap()
            .unwrap();

        let rows: Vec<(i64, f64, bool)> = Spi::connect(|c| {
            c.select(
                "SELECT row_id, trust, ood FROM fela_predict_trust('oodtbl','y')",
                None,
                &[],
            )
            .unwrap()
            .map(|r| {
                (
                    r.get::<i64>(1).unwrap().unwrap(),
                    r.get::<f64>(2).unwrap().unwrap(),
                    r.get::<bool>(3).unwrap().unwrap(),
                )
            })
            .collect()
        });
        assert_eq!(rows.len(), 2, "two NULL-y rows to predict");

        let (_, indist_trust, indist_ood) =
            *rows.iter().find(|(rid, _, _)| *rid == indist_id).unwrap();
        let (_, ood_trust, ood_ood) = *rows.iter().find(|(rid, _, _)| *rid == ood_id).unwrap();

        assert!(
            !indist_ood,
            "in-distribution row must NOT be flagged ood, trust={indist_trust}"
        );
        assert!(
            indist_trust > 0.5,
            "in-distribution row should have high trust, got {indist_trust}"
        );

        assert!(ood_ood, "far-OOD row must be flagged ood");
        assert!(
            ood_trust < 0.5,
            "far-OOD row should have low trust, got {ood_trust}"
        );
    }

    #[pg_test]
    fn predict_trust_regress_confidence_is_per_row_trust_not_global_std() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE ptregconf(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO ptregconf(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,40) g;
             INSERT INTO ptregconf(a,b,y) VALUES (2.0, 1.5, NULL);        -- in-distribution query
             INSERT INTO ptregconf(a,b,y) VALUES (5000.0, -3000.0, NULL); -- far OOD query",
        )
        .unwrap();

        let task: String =
            Spi::get_one("SELECT task FROM fela_predict_trust('ptregconf','y') LIMIT 1")
                .unwrap()
                .unwrap();
        assert_eq!(task, "regress");

        let rows: Vec<(f64, f64)> = Spi::connect(|c| {
            c.select(
                "SELECT confidence, trust FROM fela_predict_trust('ptregconf','y') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .map(|r| {
                (
                    r.get::<f64>(1).unwrap().unwrap(),
                    r.get::<f64>(2).unwrap().unwrap(),
                )
            })
            .collect()
        });
        assert_eq!(rows.len(), 2, "one (confidence, trust) pair per NULL row");

        for (conf, trust) in &rows {
            assert!(
                (0.0..=1.0).contains(conf),
                "regression confidence from fela_predict_trust must be a per-row trust \
                 score in [0,1], got {rows:?}"
            );
            assert_eq!(
                conf, trust,
                "fela_predict_trust regress confidence must equal trust exactly \
                 (same value, same machinery), got {rows:?}"
            );
        }
        let confs: Vec<f64> = rows.iter().map(|(c, _)| *c).collect();
        assert!(
            (confs[0] - confs[1]).abs() > 0.01,
            "confidence must VARY per row (in-distribution vs. far-OOD); a constant value here \
             would mean the old global-in-sample-std bug is back: {confs:?}"
        );
        assert!(
            confs[1] < confs[0],
            "the far-OOD row must have lower trust/confidence than the in-distribution row: {confs:?}"
        );
    }

    #[pg_test]
    fn classify_matches_direct_rust_reference() {
        set_gucs();
        let support = "ARRAY[5.1,3.5,1.4, 4.9,3.0,1.4, 7.0,3.2,4.7, 6.4,3.2,4.5, 6.3,3.3,6.0, 5.8,2.7,5.1]::float8[]";
        let query = "ARRAY[5.0,3.4,1.5, 6.5,3.0,4.6, 6.2,3.4,5.4]::float8[]";
        let labels = "ARRAY[0,0,1,1,2,2]::int[]";
        let got: Vec<f64> = Spi::get_one::<Vec<f64>>(&format!(
            "SELECT fela_classify({query}, {support}, {labels}, 3, 3)"
        ))
        .unwrap()
        .unwrap();
        let reference = direct_rust_reference();
        assert_eq!(got.len(), reference.len());
        let maxerr = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            maxerr < 1e-5,
            "HONESTY GATE FAIL: max|pgrx-direct| = {maxerr:e}"
        );
        for qi in 0..3 {
            let row = &got[qi * 3..qi * 3 + 3];
            let ref_row = &reference[qi * 3..qi * 3 + 3];
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-4);
            assert_eq!(
                crate::fela_argmax(row.to_vec()),
                crate::fela_argmax(ref_row.to_vec())
            );
        }
    }

    #[pg_test]
    fn automl_predicts_null_rows() {
        set_gucs();
        iris_table();
        assert_eq!(
            crate::fela_detect_task("flowers", "species")
                .split(' ')
                .next(),
            Some("classify")
        );
        let preds: Vec<f64> = Spi::connect(|c| {
            c.select(
                "SELECT prediction FROM fela_automl('flowers','species') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .filter_map(|r| r.get::<f64>(1).unwrap())
            .collect()
        });
        assert_eq!(preds.len(), 3, "one prediction per NULL row");
        let mut sorted = preds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            sorted,
            vec![0.0, 1.0, 2.0],
            "automl predicts all three classes"
        );
    }

    #[pg_test]
    fn gbm_importance_ranks_relevant_feature() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE nums3(id serial PRIMARY KEY, a float8, b float8, c float8, y float8);
             INSERT INTO nums3(a,b,c,y) SELECT g*0.1, (g*37 % 11)::float8, (g*53 % 13)::float8,
                    sin(g*0.1*3.5) * 5.0
               FROM generate_series(1,40) g;",
        )
        .unwrap();
        let top: String = Spi::get_one(
            "SELECT feature FROM fela_importance('nums3','y') ORDER BY importance DESC, feature LIMIT 1",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            top, "a",
            "expected the driving feature 'a' to have top permutation importance"
        );
    }

    #[pg_test]
    fn importance_flags_petal() {
        set_gucs();
        iris_table();
        let top: String = Spi::get_one(
            "SELECT feature FROM fela_importance('flowers','species') ORDER BY importance DESC, feature LIMIT 1",
        )
        .unwrap()
        .unwrap();
        assert_eq!(top, "petal_len", "expected petal_len most important");
    }

    #[pg_test]
    fn explain_row_ranks_dominant_feature() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE domtbl(id serial PRIMARY KEY, dom float8, noise float8, y float8);
             INSERT INTO domtbl(dom,noise,y)
               SELECT g::float8, 5.0, 3.0*g::float8 + 7.0 FROM generate_series(1,30) g;
             INSERT INTO domtbl(dom,noise,y) VALUES (100.0, 5.0, NULL);",
        )
        .unwrap();
        let row_id: i64 = Spi::get_one("SELECT row_id FROM fela_automl('domtbl','y')")
            .unwrap()
            .unwrap();

        let rows: Vec<(String, f64, String, f64)> = Spi::connect(|c| {
            c.select(
                &format!(
                    "SELECT feature, contribution, direction, value FROM fela_explain_row('domtbl','y',{row_id}) \
                     ORDER BY abs(contribution) DESC"
                ),
                None,
                &[],
            )
            .unwrap()
            .map(|r| {
                (
                    r.get::<String>(1).unwrap().unwrap(),
                    r.get::<f64>(2).unwrap().unwrap(),
                    r.get::<String>(3).unwrap().unwrap(),
                    r.get::<f64>(4).unwrap().unwrap(),
                )
            })
            .collect()
        });

        assert_eq!(rows.len(), 2, "expected one row per feature (dom, noise)");
        assert_eq!(
            rows[0].0, "dom",
            "expected 'dom' to rank top by |contribution|"
        );
        assert!(
            rows[0].1 > 0.0,
            "expected 'dom' (100.0, far above the ~15.5 support mean) to INCREASE the prediction, got {}",
            rows[0].1
        );
        assert_eq!(rows[0].2, "increases");
        assert_eq!(
            rows[0].3, 100.0,
            "'value' should be the row's own dom value"
        );

        let noise_row = rows.iter().find(|r| r.0 == "noise").unwrap();
        assert!(
            noise_row.1.abs() < 1e-6,
            "constant feature 'noise' should contribute ~0 (baseline == actual value), got {}",
            noise_row.1
        );
    }

    #[pg_test]
    fn explain_row_ranks_dominant_feature_classify() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE domcls(id serial PRIMARY KEY, dom float8, noise float8, cls int);
             INSERT INTO domcls(dom,noise,cls)
               SELECT g::float8, 5.0, 0 FROM generate_series(1,15) g;
             INSERT INTO domcls(dom,noise,cls)
               SELECT (g+20)::float8, 5.0, 1 FROM generate_series(1,15) g;
             INSERT INTO domcls(dom,noise,cls) VALUES (100.0, 5.0, NULL);",
        )
        .unwrap();
        let row_id: i64 = Spi::get_one("SELECT row_id FROM fela_automl('domcls','cls')")
            .unwrap()
            .unwrap();
        let predicted_class: f64 = Spi::get_one(&format!(
            "SELECT prediction FROM fela_automl('domcls','cls') WHERE row_id = {row_id}"
        ))
        .unwrap()
        .unwrap();
        assert_eq!(
            predicted_class, 1.0,
            "dom=100.0 (far above the class-1 range) should predict class 1"
        );

        let rows: Vec<(String, f64, String, f64)> = Spi::connect(|c| {
            c.select(
                &format!(
                    "SELECT feature, contribution, direction, value FROM fela_explain_row('domcls','cls',{row_id}) \
                     ORDER BY abs(contribution) DESC"
                ),
                None,
                &[],
            )
            .unwrap()
            .map(|r| {
                (
                    r.get::<String>(1).unwrap().unwrap(),
                    r.get::<f64>(2).unwrap().unwrap(),
                    r.get::<String>(3).unwrap().unwrap(),
                    r.get::<f64>(4).unwrap().unwrap(),
                )
            })
            .collect()
        });

        assert_eq!(rows.len(), 2, "expected one row per feature (dom, noise)");
        assert_eq!(
            rows[0].0, "dom",
            "expected 'dom' to rank top by |contribution|"
        );
        assert!(
            rows[0].1 >= 0.0,
            "expected 'dom' (100.0) to contribute TOWARD the predicted class 1, got {}",
            rows[0].1
        );
        assert_eq!(rows[0].2, "toward");
        assert_eq!(
            rows[0].3, 100.0,
            "'value' should be the row's own dom value"
        );

        let noise_row = rows.iter().find(|r| r.0 == "noise").unwrap();
        assert!(
            noise_row.1.abs() < 1e-6,
            "constant feature 'noise' should contribute ~0 (baseline == actual value), got {}",
            noise_row.1
        );
    }

    #[pg_test]
    fn cluster_produces_k_groups() {
        set_gucs();
        iris_table();
        Spi::run("CREATE TEMP TABLE ff AS SELECT sepal_len, sepal_wid, petal_len FROM flowers;")
            .unwrap();
        let total: i64 = Spi::get_one("SELECT count(*) FROM fela_cluster('ff', 3)")
            .unwrap()
            .unwrap();
        assert_eq!(total, 12, "all 12 rows should be clustered");
        let ndistinct: i64 =
            Spi::get_one("SELECT count(DISTINCT cluster) FROM fela_cluster('ff', 3)")
                .unwrap()
                .unwrap();
        assert_eq!(
            ndistinct, 3,
            "k=3 should yield 3 non-empty clusters on iris"
        );
    }

    #[pg_test]
    fn similar_returns_nearest() {
        set_gucs();
        let idx: i32 = Spi::get_one(
            "SELECT support_idx FROM fela_similar(ARRAY[5.1,3.5,1.4]::float8[], \
             ARRAY[5.1,3.5,1.4, 7.0,3.2,4.7, 6.3,3.3,6.0]::float8[], 3, 1)",
        )
        .unwrap()
        .unwrap();
        assert_eq!(idx, 0);
    }

    #[pg_test]
    fn gated_abstains_below_threshold() {
        set_gucs();
        let support = "ARRAY[5.1,3.5,1.4, 4.9,3.0,1.4, 7.0,3.2,4.7, 6.4,3.2,4.5, 6.3,3.3,6.0, 5.8,2.7,5.1]::float8[]";
        let labels = "ARRAY[0,0,1,1,2,2]::int[]";
        let none: Option<i32> = Spi::get_one(&format!(
            "SELECT fela_classify_gated(ARRAY[5.0,3.4,1.5]::float8[], {support}, {labels}, 3, 3, 0.99)"
        ))
        .unwrap();
        assert!(none.is_none(), "should abstain at 0.99");
        let some: Option<i32> = Spi::get_one(&format!(
            "SELECT fela_classify_gated(ARRAY[5.0,3.4,1.5]::float8[], {support}, {labels}, 3, 3, 0.3)"
        ))
        .unwrap();
        assert_eq!(some, Some(0), "should answer class 0 at 0.3");
    }

    fn fruits_table() {
        Spi::run(
            "CREATE TEMP TABLE fruits(id serial PRIMARY KEY, sugar float8, acidity float8, weight float8, kind int);
             INSERT INTO fruits(sugar,acidity,weight,kind) VALUES
             (9.1,0.3,120,0),(9.4,0.25,130,0),(8.8,0.35,110,0),
             (3.1,0.9,60,1),(2.8,1.1,55,1),(3.4,0.85,62,1),
             (6.0,0.6,90,2),(5.7,0.65,85,2),(6.3,0.55,95,2),
             (9.0,0.3,125,NULL),(3.0,1.0,58,NULL);",
        )
        .unwrap();
    }

    #[pg_test]
    fn over_matches_srf() {
        set_gucs();
        fruits_table();
        let cluster_ok: bool = Spi::get_one(
            "SELECT bool_and(o.c = s.cluster) FROM \
             (SELECT id, fela_cluster_over(sugar,acidity,weight) OVER () AS c FROM fruits) o \
             JOIN fela_cluster_ex('fruits',3,'kind') s ON s.row_id = o.id",
        )
        .unwrap()
        .unwrap();
        assert!(
            cluster_ok,
            "cluster_over must match fela_cluster_ex exactly"
        );

        let maxerr: f64 = Spi::get_one(
            "WITH s AS (SELECT id, sugar, acidity, weight,
                 avg(sugar) OVER () a1, stddev_pop(sugar) OVER () d1,
                 avg(acidity) OVER () a2, stddev_pop(acidity) OVER () d2,
                 avg(weight) OVER () a3, stddev_pop(weight) OVER () d3 FROM fruits),
             ref AS (SELECT id, sqrt((
                 power(CASE WHEN d1=0 THEN 0 ELSE (sugar-a1)/d1 END,2)+
                 power(CASE WHEN d2=0 THEN 0 ELSE (acidity-a2)/d2 END,2)+
                 power(CASE WHEN d3=0 THEN 0 ELSE (weight-a3)/d3 END,2))/3) AS score FROM s),
             got AS (SELECT id, fela_anomaly_over(sugar,acidity,weight) OVER () AS score FROM fruits)
             SELECT max(abs(got.score - ref.score)) FROM got JOIN ref USING (id)",
        )
        .unwrap()
        .unwrap();
        assert!(
            maxerr < 1e-9,
            "anomaly_over vs SQL RMS-z max err = {maxerr:e}"
        );
    }

    #[pg_test]
    fn create_view_matches_direct() {
        set_gucs();
        Spi::run(
            "CREATE TABLE fruits(id serial PRIMARY KEY, sugar float8, acidity float8, weight float8, kind int);
             INSERT INTO fruits(sugar,acidity,weight,kind) VALUES
             (9.1,0.3,120,0),(9.4,0.25,130,0),(8.8,0.35,110,0),
             (3.1,0.9,60,1),(2.8,1.1,55,1),(3.4,0.85,62,1),
             (6.0,0.6,90,2),(5.7,0.65,85,2),(6.3,0.55,95,2),
             (9.0,0.3,125,NULL),(3.0,1.0,58,NULL);",
        )
        .unwrap();
        let msg =
            crate::fela_create_view("fruits", Some("kind"), "cluster,anomaly,predict", 3, true);
        assert!(msg.contains("predicted"), "msg: {msg}");
        let cluster_ok: bool = Spi::get_one(
            "SELECT bool_and(v.cluster = s.cluster) FROM fruits_ml v \
             JOIN fela_cluster_ex('fruits',3,'kind') s ON s.row_id = v.id",
        )
        .unwrap()
        .unwrap();
        assert!(cluster_ok, "view cluster must match SRF");
        let predict_ok: bool = Spi::get_one(
            "SELECT bool_and(v.predicted = a.prediction) FROM fruits_ml v \
             JOIN fela_automl('fruits','kind') a ON a.row_id = v.id WHERE v.predicted IS NOT NULL",
        )
        .unwrap()
        .unwrap();
        assert!(predict_ok, "view predicted must match fela_automl");
        assert!(
            msg.contains("confidence") && msg.contains("trust") && msg.contains("ood"),
            "msg should advertise the new columns: {msg}"
        );
        let trust_ok: bool = Spi::get_one(
            "SELECT bool_and(v.confidence = t.confidence AND v.trust = t.trust AND v.ood = t.ood) \
             FROM fruits_ml v JOIN fela_predict_trust('fruits','kind') t ON t.row_id = v.id \
             WHERE v.predicted IS NOT NULL",
        )
        .unwrap()
        .unwrap();
        assert!(
            trust_ok,
            "view confidence/trust/ood must match fela_predict_trust exactly"
        );
        let ranges_ok: bool = Spi::get_one(
            "SELECT bool_and(confidence BETWEEN 0 AND 1 AND trust BETWEEN 0 AND 1) FROM fruits_ml \
             WHERE predicted IS NOT NULL",
        )
        .unwrap()
        .unwrap();
        assert!(ranges_ok, "confidence/trust must be in [0,1]");
        Spi::run("REFRESH MATERIALIZED VIEW fruits_ml").unwrap();
        let n: i64 = Spi::get_one("SELECT count(*) FROM fruits_ml")
            .unwrap()
            .unwrap();
        assert_eq!(n, 11);
    }

    #[pg_test]
    fn create_view_surfaces_trust_and_ood() {
        set_gucs();
        Spi::run(
            "CREATE TABLE oodview(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO oodview(a,b,y) SELECT g*0.1, (g%7)*0.5, 3.0*(g*0.1) - 2.0*((g%7)*0.5) + 1.0
               FROM generate_series(1,40) g;
             INSERT INTO oodview(a,b,y) VALUES (2.0, 1.5, NULL);       -- in-distribution query row
             INSERT INTO oodview(a,b,y) VALUES (5000.0, -3000.0, NULL);",
        )
        .unwrap();
        let msg = crate::fela_create_view("oodview", Some("y"), "predict", 3, true);
        assert!(
            msg.contains("predicted") && msg.contains("trust") && msg.contains("ood"),
            "msg: {msg}"
        );

        let indist_ood: bool =
            Spi::get_one("SELECT ood FROM oodview_ml WHERE a = 2.0 AND y IS NULL")
                .unwrap()
                .unwrap();
        assert!(
            !indist_ood,
            "in-distribution row must not be flagged ood in the view"
        );

        let (ood_flag, ood_trust, ood_conf, ood_pred): (bool, f64, f64, Option<f64>) =
            Spi::connect(|c| {
                let r = c
                    .select(
                        "SELECT ood, trust, confidence, predicted FROM oodview_ml \
                         WHERE a = 5000.0 AND y IS NULL",
                        None,
                        &[],
                    )?
                    .first();
                Ok::<_, pgrx::spi::SpiError>((
                    r.get::<bool>(1)?.expect("ood"),
                    r.get::<f64>(2)?.expect("trust"),
                    r.get::<f64>(3)?.expect("confidence"),
                    r.get::<f64>(4)?,
                ))
            })
            .unwrap();
        assert!(ood_flag, "far-OOD row must be flagged ood in the view");
        assert!(
            ood_trust < 0.5,
            "far-OOD row should have low trust, got {ood_trust}"
        );
        assert!(
            ood_conf >= 0.0,
            "confidence must be present and non-negative"
        );
        assert_eq!(
            ood_conf, ood_trust,
            "regression confidence in the view must equal trust (both are the same \
             per-row trust score, sourced from fela_predict_trust)"
        );
        assert!(
            ood_pred.is_some(),
            "predicted must still be populated for the OOD row"
        );

        let ood_rows: i64 = Spi::get_one("SELECT count(*) FROM oodview_ml WHERE ood")
            .unwrap()
            .unwrap();
        assert_eq!(ood_rows, 1, "exactly the far-OOD row should trip WHERE ood");
    }

    #[pg_test]
    fn create_view_one_labeled_row_degrades_predict_not_view() {
        set_gucs();
        Spi::run(
            "CREATE TABLE oneLabel(id serial PRIMARY KEY, a float8, b float8, y float8);
             INSERT INTO oneLabel(a,b,y) VALUES (1.0, 2.0, 10.0);      -- the ONLY labeled row
             INSERT INTO oneLabel(a,b,y) VALUES (1.1, 2.1, NULL);
             INSERT INTO oneLabel(a,b,y) VALUES (5.0, 9.0, NULL);
             INSERT INTO oneLabel(a,b,y) VALUES (0.9, 1.9, NULL);",
        )
        .unwrap();

        let msg =
            crate::fela_create_view("onelabel", Some("y"), "cluster,anomaly,predict", 2, true);
        assert!(msg.contains("predicted"), "msg: {msg}");

        let exists: bool =
            Spi::get_one("SELECT relkind = 'm' FROM pg_class WHERE oid = 'onelabel_ml'::regclass")
                .unwrap()
                .unwrap();
        assert!(exists, "onelabel_ml materialized view must exist");

        let n: i64 = Spi::get_one("SELECT count(*) FROM onelabel_ml")
            .unwrap()
            .unwrap();
        assert_eq!(n, 4, "all base rows must be present in the view");

        let (pred_ok, trust_null_ok): (bool, bool) = Spi::connect(|c| {
            let r = c
                .select(
                    "SELECT bool_and(predicted IS NOT NULL AND confidence IS NOT NULL), \
                     bool_and(trust IS NULL AND ood IS NULL) \
                     FROM onelabel_ml WHERE y IS NULL",
                    None,
                    &[],
                )?
                .first();
            Ok::<_, pgrx::spi::SpiError>((
                r.get::<bool>(1)?.expect("pred_ok"),
                r.get::<bool>(2)?.expect("trust_null_ok"),
            ))
        })
        .unwrap();
        assert!(
            pred_ok,
            "predicted/confidence must be populated even with 1 labeled row"
        );
        assert!(
            trust_null_ok,
            "trust/ood must be NULL when support < 2 labeled rows"
        );

        let cluster_ok: bool =
            Spi::get_one("SELECT bool_and(cluster IS NOT NULL) FROM onelabel_ml")
                .unwrap()
                .unwrap();
        assert!(cluster_ok, "cluster column must be present/populated");

        let anomaly_ok: bool =
            Spi::get_one("SELECT anomaly_score IS NOT NULL FROM onelabel_ml WHERE y = 10.0")
                .unwrap()
                .unwrap();
        assert!(
            anomaly_ok,
            "anomaly_score must be present/populated for the labeled row (unaffected by predict's degraded support)"
        );
    }

    #[pg_test]
    fn embedding_classify_probe() {
        set_gucs();
        let d = 64usize;
        let vec_for = |cls: usize, seed: usize| -> Vec<f64> {
            (0..d)
                .map(|j| {
                    let base = if cls == 0 { 1.0 } else { -1.0 };
                    let sign = if j % 2 == 0 { base } else { -base };
                    let jitter = (((seed * 7 + j * 13) % 11) as f64 - 5.0) * 0.02;
                    sign + jitter
                })
                .collect()
        };
        let mut support: Vec<f64> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        for s in 0..8 {
            support.extend(vec_for(s % 2, s));
            labels.push((s % 2) as i32);
        }
        let mut correct = 0;
        let ntest = 6;
        for t in 0..ntest {
            let cls = t % 2;
            let q = vec_for(cls, 100 + t);
            let qs = format!(
                "ARRAY[{}]::float8[]",
                q.iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let ss = format!(
                "ARRAY[{}]::float8[]",
                support
                    .iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let ls = format!(
                "ARRAY[{}]::int[]",
                labels
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let probs: Vec<f64> =
                Spi::get_one(&format!("SELECT fela_classify({qs}, {ss}, {ls}, {d}, 2)"))
                    .unwrap()
                    .unwrap();
            assert_eq!(probs.len(), 2);
            assert!(
                (probs.iter().sum::<f64>() - 1.0).abs() < 1e-4,
                "valid distribution"
            );
            if crate::fela_argmax(probs) == cls as i32 {
                correct += 1;
            }
        }
        pgrx::log!(
            "EMBEDDING PROBE: fela_classify separated {correct}/{ntest} structured 64-d vectors"
        );
    }

    #[pg_test]
    fn automl_handles_categorical_bool_and_nulls() {
        set_gucs();
        Spi::run(
            "CREATE TEMP TABLE messy(id serial PRIMARY KEY, cat text, flag bool, num float8, target int);
             INSERT INTO messy(cat,flag,num,target) VALUES
             ('red',true,5.0,0),('red',true,NULL,0),('red',true,5.0,0),
             ('green',false,5.0,1),('green',NULL,5.0,1),('green',false,5.0,1),
             ('blue',true,5.0,2),('blue',true,5.0,2),('blue',true,5.0,2),
             ('red',true,5.0,NULL),('green',false,5.0,NULL),('blue',true,5.0,NULL),
             ('purple',true,5.0,NULL);",
        )
        .unwrap();

        let task = crate::fela_detect_task("messy", "target");
        assert!(task.starts_with("classify"), "got {task}");

        let preds: Vec<f64> = Spi::connect(|c| {
            c.select(
                "SELECT prediction FROM fela_automl('messy','target') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .map(|r| r.get::<f64>(1).unwrap().unwrap())
            .collect()
        });
        assert_eq!(
            preds.len(),
            4,
            "one prediction per NULL-target row, including the unseen-category row"
        );
        for &p in &preds {
            assert!(
                (0.0..=2.0).contains(&p) && p.fract() == 0.0,
                "prediction {p} must be one of the classes {{0,1,2}}"
            );
        }
        assert_eq!(
            preds[..3],
            [0.0, 1.0, 2.0],
            "expected the categorical signal alone (cat=red/green/blue) to drive the seen-category \
             query rows to their exact matching class {{0,1,2}}, got {preds:?}"
        );
    }

    #[pg_test]
    fn numeric_table_parity_with_direct_primitive() {
        set_gucs();
        fruits_table();
        let automl_preds: Vec<(i64, f64, f64)> = Spi::connect(|c| {
            c.select(
                "SELECT row_id, prediction, confidence FROM fela_automl('fruits','kind') ORDER BY row_id",
                None,
                &[],
            )
            .unwrap()
            .map(|r| {
                (
                    r.get::<i64>(1).unwrap().unwrap(),
                    r.get::<f64>(2).unwrap().unwrap(),
                    r.get::<f64>(3).unwrap().unwrap(),
                )
            })
            .collect()
        });
        assert_eq!(automl_preds.len(), 2, "two NULL-kind rows in fruits_table");

        let (support_feats, support_labels): (Vec<f64>, Vec<i32>) = Spi::connect(|c| {
            let mut sf = Vec::new();
            let mut sl = Vec::new();
            for row in c
                .select(
                    "SELECT sugar,acidity,weight,kind FROM fruits WHERE kind IS NOT NULL ORDER BY id",
                    None,
                    &[],
                )
                .unwrap()
            {
                sf.push(row.get::<f64>(1).unwrap().unwrap());
                sf.push(row.get::<f64>(2).unwrap().unwrap());
                sf.push(row.get::<f64>(3).unwrap().unwrap());
                sl.push(row.get::<i32>(4).unwrap().unwrap());
            }
            (sf, sl)
        });
        let (query_feats, query_ids): (Vec<f64>, Vec<i64>) = Spi::connect(|c| {
            let mut qf = Vec::new();
            let mut qi = Vec::new();
            for row in c
                .select(
                    "SELECT id,sugar,acidity,weight FROM fruits WHERE kind IS NULL ORDER BY id",
                    None,
                    &[],
                )
                .unwrap()
            {
                qi.push(row.get::<i64>(1).unwrap().unwrap());
                qf.push(row.get::<f64>(2).unwrap().unwrap());
                qf.push(row.get::<f64>(3).unwrap().unwrap());
                qf.push(row.get::<f64>(4).unwrap().unwrap());
            }
            (qf, qi)
        });

        let direct: Vec<f64> =
            crate::fela_classify(query_feats, support_feats, support_labels, 3, 3);
        for (k, &qid) in query_ids.iter().enumerate() {
            let probs = &direct[k * 3..k * 3 + 3];
            let direct_cls = crate::fela_argmax(probs.to_vec()) as f64;
            let direct_conf = crate::fela_confidence(probs.to_vec());
            let (automl_cls, automl_conf) = automl_preds
                .iter()
                .find(|(rid, _, _)| *rid == qid)
                .map(|(_, p, c)| (*p, *c))
                .unwrap_or_else(|| panic!("missing automl prediction for row {qid}"));
            assert_eq!(
                automl_cls, direct_cls,
                "row {qid}: fela_automl (table path) must match fela_classify (direct primitive) \
                 exactly on an all-numeric table - the F1 byte-parity guard"
            );
            assert!(
                (automl_conf - direct_conf).abs() < 1e-9,
                "row {qid}: fela_automl confidence ({automl_conf}) must numerically match \
                 fela_confidence() of the direct fela_classify probability vector ({direct_conf}) - \
                 a same-argmax-but-different-probs divergence would otherwise slip through"
            );
        }
    }

    fn direct_rust_reference() -> Vec<f64> {
        use crate::felatab::FelaTabModel;
        let model =
            FelaTabModel::load(crate::EMBEDDED_MODEL, crate::EMBEDDED_CONFIG).expect("load");
        let support: Vec<f32> = vec![
            5.1, 3.5, 1.4, 4.9, 3.0, 1.4, 7.0, 3.2, 4.7, 6.4, 3.2, 4.5, 6.3, 3.3, 6.0, 5.8, 2.7,
            5.1,
        ];
        let query: Vec<f32> = vec![5.0, 3.4, 1.5, 6.5, 3.0, 4.6, 6.2, 3.4, 5.4];
        let labels: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let mut x = support.clone();
        x.extend_from_slice(&query);
        model
            .predict(&x, &labels, 6, 3, 3, 0, 3)
            .expect("predict")
            .into_iter()
            .map(|v| v as f64)
            .collect()
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
