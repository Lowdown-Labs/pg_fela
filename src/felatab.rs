use crate::ops::{gelu_erf, layer_norm, linear, softmax_inplace};
use crate::qgemm::QInt8Matrix;
use crate::safetensors_io::dtype_to_f32;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct FelaTabConfig {
    pub max_features: usize,
    pub max_classes: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_mult: usize,
    #[serde(default = "default_true")]
    pub use_landmark: bool,
    #[serde(default = "default_landmarks")]
    pub n_landmarks: usize,
    #[serde(default = "default_eps")]
    pub ln_eps: f32,
}
fn default_true() -> bool {
    true
}
fn default_landmarks() -> usize {
    48
}
fn default_eps() -> f32 {
    1e-5
}

enum QLinear {
    F32 { w: Vec<f32>, out: usize, inp: usize },
    Int8(QInt8Matrix),
}
impl QLinear {
    fn matmul(&self, x: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        match self {
            QLinear::F32 { w, out, inp } => linear(x, m, *inp, w, *out, bias),
            QLinear::Int8(q) => q.matmul(x, m, bias),
        }
    }
}

struct Block {
    n1_w: Vec<f32>,
    n1_b: Vec<f32>,
    aq: QLinear,
    ak: QLinear,
    av: QLinear,
    ao: QLinear,
    ab_w: QLinear,
    ab_b: Vec<f32>,
    lq: Option<QLinear>,
    lk: Option<QLinear>,
    lv: Option<QLinear>,
    lo: Option<QLinear>,
    l_gate: f32,
    n2_w: Vec<f32>,
    n2_b: Vec<f32>,
    ff0: QLinear,
    ff0_b: Vec<f32>,
    ff2: QLinear,
    ff2_b: Vec<f32>,
}

pub struct FelaTabModel {
    cfg: FelaTabConfig,
    inner: usize,
    hid: usize,
    feat_w: QLinear,
    feat_b: Vec<f32>,
    cls_label_emb: Vec<f32>,
    reg_label_w: Vec<f32>,
    reg_label_b: Vec<f32>,
    reg_label_gate: f32,
    type_emb: Vec<f32>,
    blocks: Vec<Block>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    cls_head: QLinear,
    cls_head_b: Vec<f32>,
    reg_head: QLinear,
    reg_head_b: Vec<f32>,
    last_ops: std::cell::Cell<f64>,
}

fn get_f32(st: &SafeTensors, name: &str) -> Option<Vec<f32>> {
    let t = st.tensor(name).ok()?;
    dtype_to_f32(t.dtype(), t.data()).ok()
}
fn get_i8(st: &SafeTensors, name: &str) -> Option<Vec<i8>> {
    let t = st.tensor(name).ok()?;
    if t.dtype() != Dtype::I8 {
        return None;
    }
    Some(t.data().iter().map(|&b| b as i8).collect())
}

fn load_linear(st: &SafeTensors, name: &str, out: usize, inp: usize) -> Result<QLinear, String> {
    if let (Some(qw), Some(scale)) = (get_i8(st, name), get_f32(st, &format!("{name}.scale"))) {
        if qw.len() != out * inp {
            return Err(format!("{name}: int8 len {} != {out}*{inp}", qw.len()));
        }
        if scale.len() != out {
            return Err(format!("{name}.scale len {} != {out}", scale.len()));
        }
        return Ok(QLinear::Int8(QInt8Matrix::from_parts(qw, scale, out, inp)));
    }
    let w = get_f32(st, name).ok_or_else(|| format!("missing tensor `{name}`"))?;
    if w.len() != out * inp {
        return Err(format!("{name}: f32 len {} != {out}*{inp}", w.len()));
    }
    Ok(QLinear::F32 { w, out, inp })
}

impl FelaTabModel {
    pub fn load(weights: &[u8], config_json: &str) -> Result<FelaTabModel, String> {
        let cfg: FelaTabConfig =
            serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
        let st = SafeTensors::deserialize(weights).map_err(|e| format!("safetensors: {e}"))?;
        let (c, inner, hid) = (cfg.dim, cfg.n_heads * cfg.head_dim, cfg.dim * cfg.ffn_mult);
        let g = |name: &str| get_f32(&st, name).ok_or_else(|| format!("missing tensor `{name}`"));

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("blocks.{i}");
            let ll = |n: &str, o: usize, inp: usize| load_linear(&st, &format!("{p}.{n}"), o, inp);
            let landmark = cfg.use_landmark;
            blocks.push(Block {
                n1_w: g(&format!("{p}.n1.weight"))?,
                n1_b: g(&format!("{p}.n1.bias"))?,
                aq: ll("attn.q.weight", inner, c)?,
                ak: ll("attn.k.weight", inner, c)?,
                av: ll("attn.v.weight", inner, c)?,
                ao: ll("attn.o.weight", c, inner)?,
                ab_w: ll("attn.b.weight", cfg.n_heads, c)?,
                ab_b: g(&format!("{p}.attn.b.bias"))?,
                lq: if landmark {
                    Some(ll("landmark.q.weight", inner, c)?)
                } else {
                    None
                },
                lk: if landmark {
                    Some(ll("landmark.k.weight", inner, c)?)
                } else {
                    None
                },
                lv: if landmark {
                    Some(ll("landmark.v.weight", inner, c)?)
                } else {
                    None
                },
                lo: if landmark {
                    Some(ll("landmark.o.weight", c, inner)?)
                } else {
                    None
                },
                l_gate: if landmark {
                    g(&format!("{p}.landmark.gate"))?[0]
                } else {
                    0.0
                },
                n2_w: g(&format!("{p}.n2.weight"))?,
                n2_b: g(&format!("{p}.n2.bias"))?,
                ff0: ll("ff.0.weight", hid, c)?,
                ff0_b: g(&format!("{p}.ff.0.bias"))?,
                ff2: ll("ff.2.weight", c, hid)?,
                ff2_b: g(&format!("{p}.ff.2.bias"))?,
            });
        }

        Ok(FelaTabModel {
            inner,
            hid,
            feat_w: load_linear(&st, "feat_enc.weight", c, cfg.max_features)?,
            feat_b: g("feat_enc.bias")?,
            cls_label_emb: g("cls_label_emb.weight")?,
            reg_label_w: g("reg_label.weight")?,
            reg_label_b: g("reg_label.bias")?,
            reg_label_gate: g("reg_label_gate")?[0],
            type_emb: g("type_emb.weight")?,
            blocks,
            norm_w: g("norm.weight")?,
            norm_b: g("norm.bias")?,
            cls_head: load_linear(&st, "cls_head.weight", cfg.max_classes, c)?,
            cls_head_b: g("cls_head.bias")?,
            reg_head: load_linear(&st, "reg_head.weight", 2, c)?,
            reg_head_b: g("reg_head.bias")?,
            cfg,
            last_ops: std::cell::Cell::new(0.0),
        })
    }

    pub fn last_ops(&self) -> f64 {
        self.last_ops.get()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        x_raw: &[f32],
        y_support: &[f32],
        n_support: usize,
        n_query: usize,
        n_feat: usize,
        task_type: u32,
        ncls: usize,
    ) -> Result<Vec<f32>, String> {
        let t = n_support + n_query;
        if n_support == 0 {
            return Err("need >=1 support row".into());
        }
        if x_raw.len() != t * n_feat {
            return Err(format!("x_raw len {} != {t}*{n_feat}", x_raw.len()));
        }
        if y_support.len() != n_support {
            return Err(format!("y_support len {} != {n_support}", y_support.len()));
        }
        if n_feat > self.cfg.max_features {
            return Err(format!(
                "n_feat {n_feat} > max_features {} (PCA cap not ported to wasm)",
                self.cfg.max_features
            ));
        }
        let is_cls = task_type == 0;

        let mut mean = vec![0f32; n_feat];
        let mut scale = vec![0f32; n_feat];
        for c in 0..n_feat {
            let mut s = 0f32;
            for r in 0..n_support {
                s += x_raw[r * n_feat + c];
            }
            mean[c] = s / n_support as f32;
        }
        for c in 0..n_feat {
            let mut v = 0f32;
            for r in 0..n_support {
                let d = x_raw[r * n_feat + c] - mean[c];
                v += d * d;
            }
            let sd = (v / n_support as f32).sqrt();
            scale[c] = if sd == 0.0 { 1.0 } else { sd };
        }
        let f = self.cfg.max_features;
        let mut feats = vec![0f32; t * f];
        for r in 0..t {
            for c in 0..n_feat {
                feats[r * f + c] = (x_raw[r * n_feat + c] - mean[c]) / scale[c];
            }
        }

        let (y_norm, ymu, ysd) = if is_cls {
            (y_support.to_vec(), 0.0f32, 1.0f32)
        } else {
            let ymu = y_support.iter().sum::<f32>() / n_support as f32;
            let var = y_support
                .iter()
                .map(|&v| (v - ymu) * (v - ymu))
                .sum::<f32>()
                / n_support as f32;
            let ysd = var.sqrt() + 1e-6;
            let yn = y_support
                .iter()
                .map(|&v| (v - ymu) / ysd)
                .collect::<Vec<_>>();
            (yn, ymu, ysd)
        };

        let head_out = self.forward(&feats, &y_norm, n_support, t, task_type, ncls);

        let mut out = Vec::new();
        for qi in 0..n_query {
            let row = &head_out[qi * self.head_width(is_cls, ncls)
                ..qi * self.head_width(is_cls, ncls) + self.head_width(is_cls, ncls)];
            if is_cls {
                let mut probs = row[..ncls].to_vec();
                softmax_inplace(&mut probs);
                out.extend_from_slice(&probs);
            } else {
                let mean_v = row[0] * ysd + ymu;
                let logvar = row[1].clamp(-8.0, 8.0);
                let std_v = (logvar.exp()).sqrt() * ysd;
                out.push(mean_v);
                out.push(std_v);
            }
        }
        Ok(out)
    }

    fn head_width(&self, is_cls: bool, ncls: usize) -> usize {
        if is_cls {
            ncls
        } else {
            2
        }
    }

    fn forward(
        &self,
        feats: &[f32],
        y_norm: &[f32],
        ns: usize,
        t: usize,
        task_type: u32,
        ncls: usize,
    ) -> Vec<f32> {
        let c = self.cfg.dim;
        let is_cls = task_type == 0;
        let mut ops = 0f64;

        let mut x = self.feat_w.matmul(feats, t, &self.feat_b);
        ops += (t * c * self.cfg.max_features) as f64;
        let te = &self.type_emb[task_type as usize * c..task_type as usize * c + c];
        for r in 0..t {
            for (d, xv) in x[r * c..r * c + c].iter_mut().enumerate() {
                *xv += te[d];
            }
        }
        if is_cls {
            for r in 0..ns {
                let cls = (y_norm[r].round().max(0.0) as usize).min(self.cfg.max_classes);
                let emb = &self.cls_label_emb[cls * c..cls * c + c];
                for (d, xv) in x[r * c..r * c + c].iter_mut().enumerate() {
                    *xv += emb[d];
                }
            }
        } else {
            for r in 0..ns {
                let y = y_norm[r];
                for (d, xv) in x[r * c..r * c + c].iter_mut().enumerate() {
                    let lab = y * self.reg_label_w[d] + self.reg_label_b[d];
                    *xv += lab * self.reg_label_gate;
                }
            }
        }

        let wm: Vec<f32> = (0..t).map(|r| if r < ns { 1.0 } else { 0.0 }).collect();

        for blk in &self.blocks {
            let mut h = x.clone();
            layer_norm(&mut h, t, c, &blk.n1_w, &blk.n1_b, self.cfg.ln_eps);
            let mut a = self.delta_attn(blk, &h, &wm, t, &mut ops);
            if blk.lq.is_some() {
                let l = self.landmark_attn(blk, &h, &wm, ns, t, &mut ops);
                for (av, lv) in a.iter_mut().zip(&l) {
                    *av += lv;
                }
            }
            for (xv, av) in x.iter_mut().zip(&a) {
                *xv += av;
            }
            let mut h2 = x.clone();
            layer_norm(&mut h2, t, c, &blk.n2_w, &blk.n2_b, self.cfg.ln_eps);
            let mut g = blk.ff0.matmul(&h2, t, &blk.ff0_b);
            for v in g.iter_mut() {
                *v = gelu_erf(*v);
            }
            let f = blk.ff2.matmul(&g, t, &blk.ff2_b);
            ops += (2 * t * c * self.hid) as f64;
            for (xv, fv) in x.iter_mut().zip(&f) {
                *xv += fv;
            }
        }
        layer_norm(&mut x, t, c, &self.norm_w, &self.norm_b, self.cfg.ln_eps);

        let hw = self.head_width(is_cls, ncls);
        let mut out = vec![0f32; (t - ns) * hw];
        for qi in 0..(t - ns) {
            let row = &x[(ns + qi) * c..(ns + qi) * c + c];
            if is_cls {
                let logits = self.cls_head.matmul(row, 1, &self.cls_head_b);
                out[qi * hw..qi * hw + hw].copy_from_slice(&logits[..ncls]);
            } else {
                let mv = self.reg_head.matmul(row, 1, &self.reg_head_b);
                out[qi * 2..qi * 2 + 2].copy_from_slice(&mv[..2]);
            }
        }
        self.last_ops.set(ops);
        out
    }

    fn delta_attn(&self, blk: &Block, h: &[f32], wm: &[f32], t: usize, ops: &mut f64) -> Vec<f32> {
        let (c, hh, d) = (self.cfg.dim, self.cfg.n_heads, self.cfg.head_dim);
        let inner = self.inner;
        let mut q = blk.aq.matmul(h, t, &[]);
        let mut k = blk.ak.matmul(h, t, &[]);
        let v = blk.av.matmul(h, t, &[]);
        let beta_lin = blk.ab_w.matmul(h, t, &blk.ab_b);
        *ops += (3 * t * inner * c) as f64;

        for r in 0..t {
            for head in 0..hh {
                let off = r * inner + head * d;
                l2_normalize(&mut q[off..off + d]);
                l2_normalize(&mut k[off..off + d]);
            }
        }

        let mut o = vec![0f32; t * inner];
        for head in 0..hh {
            let mut s = vec![0f32; d * d];
            for r in 0..t {
                let off = r * inner + head * d;
                let qh = &q[off..off + d];
                let kh = &k[off..off + d];
                let vh = &v[off..off + d];
                let beta = sigmoid(beta_lin[r * hh + head]) * wm[r];
                for e in 0..d {
                    let mut acc = 0f32;
                    for dd in 0..d {
                        acc += qh[dd] * s[dd * d + e];
                    }
                    o[off + e] = acc;
                }
                if beta != 0.0 {
                    let mut u = vec![0f32; d];
                    for e in 0..d {
                        let mut ks = 0f32;
                        for dd in 0..d {
                            ks += kh[dd] * s[dd * d + e];
                        }
                        u[e] = beta * (vh[e] - ks);
                    }
                    for dd in 0..d {
                        let kd = kh[dd];
                        if kd != 0.0 {
                            let row = &mut s[dd * d..dd * d + d];
                            for e in 0..d {
                                row[e] += kd * u[e];
                            }
                        }
                    }
                }
            }
        }
        *ops += (2 * t * hh * d * d) as f64;
        blk.ao.matmul(&o, t, &[])
    }

    fn landmark_attn(
        &self,
        blk: &Block,
        h: &[f32],
        _wm: &[f32],
        ns: usize,
        t: usize,
        ops: &mut f64,
    ) -> Vec<f32> {
        let (c, hh, d, gtot) = (
            self.cfg.dim,
            self.cfg.n_heads,
            self.cfg.head_dim,
            self.cfg.n_landmarks,
        );
        let inner = self.inner;

        let gs = ((ns as f32) / (gtot as f32)).ceil().max(1.0);
        let mut land = vec![0f32; gtot * c];
        let mut cnt = vec![0f32; gtot];
        for r in 0..ns {
            let gid = ((r as f32 / gs).floor() as usize).min(gtot - 1);
            for dd in 0..c {
                land[gid * c + dd] += h[r * c + dd];
            }
            cnt[gid] += 1.0;
        }
        for g in 0..gtot {
            let ct = cnt[g].max(1.0);
            for dd in 0..c {
                land[g * c + dd] /= ct;
            }
        }
        let populated: Vec<bool> = cnt.iter().map(|&x| x > 0.0).collect();

        let lq = blk.lq.as_ref().unwrap();
        let lk = blk.lk.as_ref().unwrap();
        let lv = blk.lv.as_ref().unwrap();
        let lo = blk.lo.as_ref().unwrap();
        let q = lq.matmul(h, t, &[]);
        let kk = lk.matmul(&land, gtot, &[]);
        let vv = lv.matmul(&land, gtot, &[]);
        *ops += (t * inner * c + 2 * gtot * inner * c) as f64;

        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let mut o = vec![0f32; t * inner];
        for r in 0..t {
            for head in 0..hh {
                let qoff = r * inner + head * d;
                let mut att = vec![f32::NEG_INFINITY; gtot];
                for g in 0..gtot {
                    if !populated[g] {
                        att[g] = -1e9;
                        continue;
                    }
                    let koff = g * inner + head * d;
                    let mut dotp = 0f32;
                    for dd in 0..d {
                        dotp += q[qoff + dd] * kk[koff + dd];
                    }
                    att[g] = dotp * inv_sqrt_d;
                }
                softmax_inplace(&mut att);
                for e in 0..d {
                    let mut acc = 0f32;
                    for g in 0..gtot {
                        acc += att[g] * vv[g * inner + head * d + e];
                    }
                    o[qoff + e] = acc;
                }
            }
        }
        *ops += (2 * t * hh * gtot * d) as f64;
        let mut out = lo.matmul(&o, t, &[]);
        let gate = blk.l_gate.tanh();
        for v in out.iter_mut() {
            *v *= gate;
        }
        out
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn l2_normalize(x: &mut [f32]) {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for v in x.iter_mut() {
        *v /= n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_and_l2() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }
}
