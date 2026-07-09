#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline]
fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn sum_sumsq(x: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        return unsafe { sum_sumsq_avx512(x) };
    }
    let mut s = 0.0f32;
    let mut q = 0.0f32;
    for &v in x {
        s += v;
        q += v * v;
    }
    (s, q)
}

#[inline]
pub fn sum_squares(x: &[f32]) -> f32 {
    sum_sumsq(x).1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_sumsq_avx512(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut s = _mm512_setzero_ps();
    let mut q = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        s = _mm512_add_ps(s, v);
        q = _mm512_fmadd_ps(v, v, q);
        i += 16;
    }
    let mut ts = _mm512_reduce_add_ps(s);
    let mut tq = _mm512_reduce_add_ps(q);
    while i < n {
        let v = *x.get_unchecked(i);
        ts += v;
        tq += v * v;
        i += 1;
    }
    (ts, tq)
}

#[inline]
fn scale_weight(row: &mut [f32], scale: f32, weight: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        unsafe {
            scale_weight_avx512(row, scale, weight);
        }
        return;
    }
    for (d, v) in row.iter_mut().enumerate() {
        let w = if weight.is_empty() { 1.0 } else { weight[d] };
        *v = *v * scale * w;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scale_weight_avx512(row: &mut [f32], scale: f32, weight: &[f32]) {
    let n = row.len();
    let vs = _mm512_set1_ps(scale);
    let mut i = 0;
    if weight.is_empty() {
        while i + 16 <= n {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            _mm512_storeu_ps(row.as_mut_ptr().add(i), _mm512_mul_ps(v, vs));
            i += 16;
        }
        while i < n {
            *row.get_unchecked_mut(i) *= scale;
            i += 1;
        }
        return;
    }
    while i + 16 <= n {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        let w = _mm512_loadu_ps(weight.as_ptr().add(i));
        _mm512_storeu_ps(
            row.as_mut_ptr().add(i),
            _mm512_mul_ps(_mm512_mul_ps(v, vs), w),
        );
        i += 16;
    }
    while i < n {
        *row.get_unchecked_mut(i) = *row.get_unchecked(i) * scale * *weight.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn affine_avx512(row: &mut [f32], mean: f32, inv: f32, weight: &[f32], bias: &[f32]) {
    let n = row.len();
    let vm = _mm512_set1_ps(mean);
    let vi = _mm512_set1_ps(inv);
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        let w = _mm512_loadu_ps(weight.as_ptr().add(i));
        let b = _mm512_loadu_ps(bias.as_ptr().add(i));
        let c = _mm512_mul_ps(_mm512_sub_ps(v, vm), vi);
        _mm512_storeu_ps(row.as_mut_ptr().add(i), _mm512_fmadd_ps(c, w, b));
        i += 16;
    }
    while i < n {
        *row.get_unchecked_mut(i) = (*row.get_unchecked(i) - mean) * inv * *weight.get_unchecked(i)
            + *bias.get_unchecked(i);
        i += 1;
    }
}

pub fn softmax_inplace(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in xs.iter_mut() {
        *x *= inv;
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn rms_norm(x: &mut [f32], rows: usize, dim: usize, weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), rows * dim);
    for r in 0..rows {
        let row = &mut x[r * dim..r * dim + dim];
        let ms = sum_squares(row) / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        scale_weight(row, inv, weight);
    }
}

pub fn layer_norm(x: &mut [f32], rows: usize, dim: usize, weight: &[f32], bias: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), rows * dim);
    let dn = dim as f32;
    for r in 0..rows {
        let row = &mut x[r * dim..r * dim + dim];
        let (s, sq) = sum_sumsq(row);
        let mean = s / dn;
        let var = (sq / dn - mean * mean).max(0.0);
        let inv = 1.0 / (var + eps).sqrt();
        #[cfg(target_arch = "x86_64")]
        if has_avx512() && !weight.is_empty() && !bias.is_empty() {
            unsafe { affine_avx512(row, mean, inv, weight, bias) };
            continue;
        }
        for (d, v) in row.iter_mut().enumerate() {
            let w = if weight.is_empty() { 1.0 } else { weight[d] };
            let b = if bias.is_empty() { 0.0 } else { bias[d] };
            *v = (*v - mean) * inv * w + b;
        }
    }
}

#[allow(clippy::excessive_precision)]
pub fn erf(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    let erfc = ans;
    if x >= 0.0 {
        1.0 - erfc
    } else {
        erfc - 1.0
    }
}

pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * core::f32::consts::FRAC_1_SQRT_2))
}

pub fn linear(
    x: &[f32],
    rows: usize,
    in_f: usize,
    w: &[f32],
    out_f: usize,
    bias: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * in_f);
    debug_assert_eq!(w.len(), out_f * in_f);
    let mut y = vec![0.0f32; rows * out_f];

    #[cfg(feature = "parallel")]
    {
        if (rows as u64) * (out_f as u64) * (in_f as u64) >= 4_000_000 {
            use rayon::prelude::*;
            y.par_iter_mut().enumerate().for_each(|(idx, yv)| {
                let r = idx / out_f;
                let o = idx % out_f;
                let xr = &x[r * in_f..r * in_f + in_f];
                let wr = &w[o * in_f..o * in_f + in_f];
                let acc = dot(xr, wr);
                *yv = if bias.is_empty() { acc } else { acc + bias[o] };
            });
            return y;
        }
    }

    for r in 0..rows {
        let xr = &x[r * in_f..r * in_f + in_f];
        for o in 0..out_f {
            let wr = &w[o * in_f..o * in_f + in_f];
            let mut acc = dot(xr, wr);
            if !bias.is_empty() {
                acc += bias[o];
            }
            y[r * out_f + o] = acc;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn linear_identity() {
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let y = linear(&x, 1, 2, &w, 2, &[]);
        assert_eq!(y, vec![1.0, 2.0]);
    }
}
