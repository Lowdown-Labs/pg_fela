#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "parallel")]
struct SendPtr(*mut f32);
#[cfg(feature = "parallel")]
unsafe impl Send for SendPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtr {}
#[cfg(feature = "parallel")]
impl SendPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

pub struct QInt8Matrix {
    pub n: usize,
    pub k: usize,
    qw: Vec<i8>,
    wscale: Vec<f32>,
    wsum: Vec<i32>,
}

impl QInt8Matrix {
    pub fn from_f32(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k, "weight len != n*k");
        let mut qw = vec![0i8; n * k];
        let mut wscale = vec![0f32; n];
        let mut wsum = vec![0i32; n];
        for r in 0..n {
            let row = &w[r * k..r * k + k];
            let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = 1.0 / s;
            let mut sum = 0i32;
            for c in 0..k {
                let q = (row[c] * inv).round().clamp(-127.0, 127.0) as i8;
                qw[r * k + c] = q;
                sum += q as i32;
            }
            wscale[r] = s;
            wsum[r] = sum;
        }
        Self {
            n,
            k,
            qw,
            wscale,
            wsum,
        }
    }

    pub fn from_parts(qw: Vec<i8>, wscale: Vec<f32>, n: usize, k: usize) -> Self {
        assert_eq!(qw.len(), n * k, "qw len != n*k");
        assert_eq!(wscale.len(), n, "wscale len != n");
        let mut wsum = vec![0i32; n];
        for (r, s) in wsum.iter_mut().enumerate() {
            *s = qw[r * k..r * k + k].iter().map(|&q| q as i32).sum();
        }
        Self {
            n,
            k,
            qw,
            wscale,
            wsum,
        }
    }

    fn quant_row(x: &[f32], out: &mut [u8]) -> f32 {
        let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv = 1.0 / s;
        for (o, &v) in out.iter_mut().zip(x) {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
            *o = (q + 128) as u8;
        }
        s
    }

    pub fn matmul(&self, x: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        self.run(x, m, bias, vnni_available())
    }

    pub fn matmul_scalar(&self, x: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        self.run(x, m, bias, false)
    }

    fn run(&self, x: &[f32], m: usize, bias: &[f32], use_vnni: bool) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "x len != m*k");
        let (xq, ascale) = quantize_activations(x, m, self.k);
        self.compute(&xq, &ascale, m, bias, use_vnni)
    }

    pub fn matmul_prequant(&self, xq: &[u8], ascale: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        assert_eq!(xq.len(), m * self.k, "xq len != m*k");
        assert_eq!(ascale.len(), m, "ascale len != m");
        self.compute(xq, ascale, m, bias, vnni_available())
    }

    fn compute(
        &self,
        xq: &[u8],
        ascale: &[f32],
        m: usize,
        bias: &[f32],
        use_vnni: bool,
    ) -> Vec<f32> {
        let (n, k) = (self.n, self.k);
        assert!(bias.is_empty() || bias.len() == n, "bias len != n");

        let mut y = vec![0f32; m * n];
        const PARALLEL_WORK_MIN: u64 = 4_000_000;
        let threads = if (m as u64) * (n as u64) * (k as u64) >= PARALLEL_WORK_MIN {
            thread_count(n)
        } else {
            1
        };
        if threads <= 1 {
            self.fill_cols(0, n, y.as_mut_ptr(), n, m, xq, ascale, bias, use_vnni);
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let cols_per = n.div_ceil(threads);
                let yp = SendPtr(y.as_mut_ptr());
                (0..threads).into_par_iter().for_each(|t| {
                    let base = t * cols_per;
                    if base >= n {
                        return;
                    }
                    let ncols = cols_per.min(n - base);
                    self.fill_cols(base, ncols, yp.get(), n, m, xq, ascale, bias, use_vnni);
                });
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_cols(
        &self,
        base: usize,
        ncols: usize,
        yptr: *mut f32,
        n: usize,
        m: usize,
        xq: &[u8],
        ascale: &[f32],
        bias: &[f32],
        use_vnni: bool,
    ) {
        let k = self.k;
        let row = |r: usize| &xq[r * k..r * k + k];
        let wcol = |col: usize| &self.qw[col * k..col * k + k];
        let deq = |acc: i32, col: usize, r: usize| {
            let bz = if bias.is_empty() { 0.0 } else { bias[col] };
            ((acc - 128 * self.wsum[col]) as f32) * ascale[r] * self.wscale[col] + bz
        };
        let put = |col: usize, r: usize, val: f32| unsafe { *yptr.add(r * n + col) = val };
        let end = base + ncols;

        #[cfg(target_arch = "x86_64")]
        if use_vnni {
            let mut j = base;
            while j + 4 <= end {
                let w = [wcol(j), wcol(j + 1), wcol(j + 2), wcol(j + 3)];
                let mut r = 0;
                while r + 4 <= m {
                    let a = [row(r), row(r + 1), row(r + 2), row(r + 3)];
                    let acc = unsafe { vnni_tile_4x4(a, w) };
                    for (rr, accr) in acc.iter().enumerate() {
                        for (cc, &av) in accr.iter().enumerate() {
                            put(j + cc, r + rr, deq(av, j + cc, r + rr));
                        }
                    }
                    r += 4;
                }
                while r < m {
                    for (cc, &wc) in w.iter().enumerate() {
                        put(j + cc, r, deq(unsafe { vnni_dot(row(r), wc) }, j + cc, r));
                    }
                    r += 1;
                }
                j += 4;
            }
            while j < end {
                let wr = wcol(j);
                let mut r = 0;
                while r + 4 <= m {
                    let acc =
                        unsafe { vnni_dot4([row(r), row(r + 1), row(r + 2), row(r + 3)], wr) };
                    for (rr, &av) in acc.iter().enumerate() {
                        put(j, r + rr, deq(av, j, r + rr));
                    }
                    r += 4;
                }
                while r < m {
                    put(j, r, deq(unsafe { vnni_dot(row(r), wr) }, j, r));
                    r += 1;
                }
                j += 1;
            }
            return;
        }
        for j in base..end {
            let wr = wcol(j);
            for r in 0..m {
                put(j, r, deq(scalar_dot(row(r), wr), j, r));
            }
        }
    }
}

fn thread_count(n: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        let avail = std::thread::available_parallelism().map_or(1, |t| t.get());
        let cap = std::env::var("FELA_QGEMM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&t| t > 0)
            .unwrap_or(avail);
        cap.min(avail).min(n.max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        1
    }
}

fn scalar_dot(a: &[u8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &w)| (x as i32) * (w as i32))
        .sum()
}

fn vnni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn vnni_dot(a: &[u8], b: &[i8]) -> i32 {
    let k = a.len();
    let (mut c0, mut c1, mut c2, mut c3) = (
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
    );
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    while i + 256 <= k {
        c0 = _mm512_dpbusd_epi32(c0, ld(ap, i), ld8(bp, i));
        c1 = _mm512_dpbusd_epi32(c1, ld(ap, i + 64), ld8(bp, i + 64));
        c2 = _mm512_dpbusd_epi32(c2, ld(ap, i + 128), ld8(bp, i + 128));
        c3 = _mm512_dpbusd_epi32(c3, ld(ap, i + 192), ld8(bp, i + 192));
        i += 256;
    }
    while i + 64 <= k {
        c0 = _mm512_dpbusd_epi32(c0, ld(ap, i), ld8(bp, i));
        i += 64;
    }
    let acc = _mm512_add_epi32(_mm512_add_epi32(c0, c1), _mm512_add_epi32(c2, c3));
    let mut sum = _mm512_reduce_add_epi32(acc);
    while i < k {
        sum += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn ld(p: *const u8, i: usize) -> __m512i {
    _mm512_loadu_si512(p.add(i) as *const __m512i)
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn ld8(p: *const i8, i: usize) -> __m512i {
    _mm512_loadu_si512(p.add(i) as *const __m512i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn vnni_dot4(a: [&[u8]; 4], b: &[i8]) -> [i32; 4] {
    let k = b.len();
    let (mut c0, mut c1, mut c2, mut c3) = (
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
    );
    let bp = b.as_ptr();
    let (p0, p1, p2, p3) = (a[0].as_ptr(), a[1].as_ptr(), a[2].as_ptr(), a[3].as_ptr());
    let mut i = 0;
    while i + 64 <= k {
        let vb = ld8(bp, i);
        c0 = _mm512_dpbusd_epi32(c0, ld(p0, i), vb);
        c1 = _mm512_dpbusd_epi32(c1, ld(p1, i), vb);
        c2 = _mm512_dpbusd_epi32(c2, ld(p2, i), vb);
        c3 = _mm512_dpbusd_epi32(c3, ld(p3, i), vb);
        i += 64;
    }
    let mut s = [
        _mm512_reduce_add_epi32(c0),
        _mm512_reduce_add_epi32(c1),
        _mm512_reduce_add_epi32(c2),
        _mm512_reduce_add_epi32(c3),
    ];
    while i < k {
        let w = b[i] as i32;
        for t in 0..4 {
            s[t] += a[t][i] as i32 * w;
        }
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn vnni_tile_4x4(a: [&[u8]; 4], w: [&[i8]; 4]) -> [[i32; 4]; 4] {
    let k = w[0].len();
    let mut acc = [[_mm512_setzero_si512(); 4]; 4];
    let ap = [a[0].as_ptr(), a[1].as_ptr(), a[2].as_ptr(), a[3].as_ptr()];
    let wp = [w[0].as_ptr(), w[1].as_ptr(), w[2].as_ptr(), w[3].as_ptr()];
    let mut i = 0;
    while i + 64 <= k {
        let av = [ld(ap[0], i), ld(ap[1], i), ld(ap[2], i), ld(ap[3], i)];
        let wv = [ld8(wp[0], i), ld8(wp[1], i), ld8(wp[2], i), ld8(wp[3], i)];
        for r in 0..4 {
            for (c, &wc) in wv.iter().enumerate() {
                acc[r][c] = _mm512_dpbusd_epi32(acc[r][c], av[r], wc);
            }
        }
        i += 64;
    }
    let mut out = [[0i32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = _mm512_reduce_add_epi32(acc[r][c]);
        }
    }
    while i < k {
        for r in 0..4 {
            let av = a[r][i] as i32;
            for c in 0..4 {
                out[r][c] += av * w[c][i] as i32;
            }
        }
        i += 1;
    }
    out
}

pub struct QInt4Matrix {
    pub n: usize,
    pub k: usize,
    qw4: Vec<u8>,
    wscale: Vec<f32>,
    wsum: Vec<i32>,
}

#[inline]
fn sign_ext_i4(nib: u8) -> i32 {
    ((nib ^ 0x08) as i32) - 8
}

pub fn quantize_activations(x: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(x.len(), m * k, "x len != m*k");
    let mut xq = vec![0u8; m * k];
    let mut ascale = vec![0f32; m];
    for r in 0..m {
        ascale[r] = QInt8Matrix::quant_row(&x[r * k..r * k + k], &mut xq[r * k..r * k + k]);
    }
    (xq, ascale)
}

impl QInt4Matrix {
    pub fn from_f32(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k, "weight len != n*k");
        assert_eq!(k % 64, 0, "int4 path requires K a multiple of 64");
        let kb = k / 2;
        let mut qw4 = vec![0u8; n * kb];
        let mut wscale = vec![0f32; n];
        let mut wsum = vec![0i32; n];
        for r in 0..n {
            let row = &w[r * k..r * k + k];
            let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            let inv = 1.0 / s;
            let q: Vec<i8> = row
                .iter()
                .map(|&v| (v * inv).round().clamp(-7.0, 7.0) as i8)
                .collect();
            wsum[r] = q.iter().map(|&x| x as i32).sum();
            let dst = &mut qw4[r * kb..r * kb + kb];
            for b in 0..k / 64 {
                let base = b * 64;
                for j in 0..32 {
                    let lo = (q[base + j] & 0xF) as u8;
                    let hi = (q[base + 32 + j] & 0xF) as u8;
                    dst[b * 32 + j] = (hi << 4) | lo;
                }
            }
            wscale[r] = s;
        }
        Self {
            n,
            k,
            qw4,
            wscale,
            wsum,
        }
    }

    pub fn matmul(&self, x: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        self.run(x, m, bias, i4_vnni_available())
    }
    pub fn matmul_scalar(&self, x: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        self.run(x, m, bias, false)
    }

    fn run(&self, x: &[f32], m: usize, bias: &[f32], use_vnni: bool) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "x len != m*k");
        let (xq, ascale) = quantize_activations(x, m, self.k);
        self.compute(&xq, &ascale, m, bias, use_vnni)
    }

    pub fn matmul_prequant(&self, xq: &[u8], ascale: &[f32], m: usize, bias: &[f32]) -> Vec<f32> {
        assert_eq!(xq.len(), m * self.k, "xq len != m*k");
        assert_eq!(ascale.len(), m, "ascale len != m");
        self.compute(xq, ascale, m, bias, i4_vnni_available())
    }

    fn compute(
        &self,
        xq: &[u8],
        ascale: &[f32],
        m: usize,
        bias: &[f32],
        use_vnni: bool,
    ) -> Vec<f32> {
        let (n, k, kb) = (self.n, self.k, self.k / 2);
        assert!(bias.is_empty() || bias.len() == n, "bias len != n");
        let mut y = vec![0f32; m * n];
        let threads = if (m as u64) * (n as u64) * (k as u64) >= 4_000_000 {
            thread_count(n)
        } else {
            1
        };
        let fill = |base: usize, ncols: usize, yptr: *mut f32| {
            let row = |r: usize| &xq[r * k..r * k + k];
            let wc = |col: usize| &self.qw4[col * kb..col * kb + kb];
            let deq = |acc: i32, col: usize, r: usize| {
                let bz = if bias.is_empty() { 0.0 } else { bias[col] };
                ((acc - 128 * self.wsum[col]) as f32) * ascale[r] * self.wscale[col] + bz
            };
            let put = |col: usize, r: usize, v: f32| unsafe { *yptr.add(r * n + col) = v };
            for col in base..base + ncols {
                let wp = wc(col);
                let mut r = 0;
                #[cfg(target_arch = "x86_64")]
                if use_vnni {
                    while r + 4 <= m {
                        let acc =
                            unsafe { i4_dot4([row(r), row(r + 1), row(r + 2), row(r + 3)], wp) };
                        for (rr, &a) in acc.iter().enumerate() {
                            put(col, r + rr, deq(a, col, r + rr));
                        }
                        r += 4;
                    }
                    while r < m {
                        put(col, r, deq(unsafe { i4_dot(row(r), wp) }, col, r));
                        r += 1;
                    }
                    continue;
                }
                for r in 0..m {
                    put(col, r, deq(i4_scalar_dot(row(r), wp, k), col, r));
                }
            }
        };
        if threads <= 1 {
            fill(0, n, y.as_mut_ptr());
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let cols_per = n.div_ceil(threads);
                let yp = SendPtr(y.as_mut_ptr());
                (0..threads).into_par_iter().for_each(|t| {
                    let base = t * cols_per;
                    if base < n {
                        fill(base, cols_per.min(n - base), yp.get());
                    }
                });
            }
        }
        y
    }
}

fn i4_scalar_dot(a: &[u8], wp: &[u8], k: usize) -> i32 {
    let mut acc = 0i32;
    for b in 0..k / 64 {
        for j in 0..32 {
            let byte = wp[b * 32 + j];
            acc += a[b * 64 + j] as i32 * sign_ext_i4(byte & 0x0F);
            acc += a[b * 64 + 32 + j] as i32 * sign_ext_i4((byte >> 4) & 0x0F);
        }
    }
    acc
}

fn i4_vnni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
unsafe fn unpack_i4(p: *const u8) -> __m512i {
    let packed = _mm256_loadu_si256(p as *const __m256i);
    let mask = _mm256_set1_epi8(0x0F);
    let eight = _mm256_set1_epi8(8);
    let lo = _mm256_sub_epi8(
        _mm256_xor_si256(_mm256_and_si256(packed, mask), eight),
        eight,
    );
    let hi_n = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    let hi = _mm256_sub_epi8(_mm256_xor_si256(hi_n, eight), eight);
    _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
unsafe fn i4_dot(a: &[u8], wp: &[u8]) -> i32 {
    let k = a.len();
    let (mut c0, mut c1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    let (ap, wpp) = (a.as_ptr(), wp.as_ptr());
    let (mut i, mut wi) = (0, 0);
    while i + 128 <= k {
        c0 = _mm512_dpbusd_epi32(c0, ld(ap, i), unpack_i4(wpp.add(wi)));
        c1 = _mm512_dpbusd_epi32(c1, ld(ap, i + 64), unpack_i4(wpp.add(wi + 32)));
        i += 128;
        wi += 64;
    }
    while i + 64 <= k {
        c0 = _mm512_dpbusd_epi32(c0, ld(ap, i), unpack_i4(wpp.add(wi)));
        i += 64;
        wi += 32;
    }
    _mm512_reduce_add_epi32(_mm512_add_epi32(c0, c1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
unsafe fn i4_dot4(a: [&[u8]; 4], wp: &[u8]) -> [i32; 4] {
    let k = a[0].len();
    let mut c = [_mm512_setzero_si512(); 4];
    let ap = [a[0].as_ptr(), a[1].as_ptr(), a[2].as_ptr(), a[3].as_ptr()];
    let wpp = wp.as_ptr();
    let (mut i, mut wi) = (0, 0);
    while i + 64 <= k {
        let wv = unpack_i4(wpp.add(wi));
        for r in 0..4 {
            c[r] = _mm512_dpbusd_epi32(c[r], ld(ap[r], i), wv);
        }
        i += 64;
        wi += 32;
    }
    [
        _mm512_reduce_add_epi32(c[0]),
        _mm512_reduce_add_epi32(c[1]),
        _mm512_reduce_add_epi32(c[2]),
        _mm512_reduce_add_epi32(c[3]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_ref(x: &[f32], m: usize, w: &[f32], n: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for j in 0..k {
                    acc += x[r * k + j] * w[c * k + j];
                }
                y[r * n + c] = acc;
            }
        }
        y
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn qint8_matmul_approximates_f32() {
        let (m, n, k) = (5usize, 192usize, 256usize);
        let w = fill(1, n * k, 1.0);
        let x = fill(2, m * k, 1.0);
        let q = QInt8Matrix::from_f32(&w, n, k);
        let y = q.matmul(&x, m, &[]);
        let r = f32_ref(&x, m, &w, n, k);
        let c = cos(&y, &r);
        assert!(c > 0.99, "cos {c} too low (int8 GEMM != f32)");
        let rel = {
            let num: f32 = y
                .iter()
                .zip(&r)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let den: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            num / den
        };
        assert!(rel < 0.05, "rel error {rel} too high");
    }

    #[test]
    fn from_parts_reconstructs_from_f32() {
        let (m, n, k) = (4usize, 96usize, 128usize);
        let w = fill(9, n * k, 0.9);
        let x = fill(10, m * k, 1.1);
        let q = QInt8Matrix::from_f32(&w, n, k);
        let rebuilt = QInt8Matrix::from_parts(q.qw.clone(), q.wscale.clone(), n, k);
        assert_eq!(rebuilt.wsum, q.wsum, "recomputed wsum mismatch");
        assert_eq!(
            q.matmul_scalar(&x, m, &[]),
            rebuilt.matmul_scalar(&x, m, &[]),
            "from_parts GEMM != from_f32 GEMM"
        );
    }

    #[test]
    fn vnni_matches_scalar_bit_for_bit() {
        let (m, n, k) = (4usize, 130usize, 320usize);
        let w = fill(3, n * k, 0.7);
        let x = fill(4, m * k, 1.3);
        let q = QInt8Matrix::from_f32(&w, n, k);
        let bias = fill(5, n, 0.2);
        let y_vnni = q.matmul(&x, m, &bias);
        let y_scalar = q.matmul_scalar(&x, m, &bias);
        for (a, b) in y_vnni.iter().zip(&y_scalar) {
            assert!((a - b).abs() < 1e-3, "vnni {a} != scalar {b}");
        }
    }

    #[test]
    fn qint4_matmul_approximates_f32() {
        let (m, n, k) = (5usize, 192usize, 256usize);
        let w = fill(1, n * k, 1.0);
        let x = fill(2, m * k, 1.0);
        let y = QInt4Matrix::from_f32(&w, n, k).matmul(&x, m, &[]);
        let c = cos(&y, &f32_ref(&x, m, &w, n, k));
        assert!(
            c > 0.96,
            "int4 cos {c} too low (lossier than int8, but should track f32)"
        );
    }

    #[test]
    fn i4_vnni_matches_scalar_bit_for_bit() {
        let (m, n, k) = (4usize, 130usize, 256usize);
        let w = fill(3, n * k, 0.7);
        let x = fill(4, m * k, 1.3);
        let q = QInt4Matrix::from_f32(&w, n, k);
        let bias = fill(5, n, 0.2);
        let (yv, ys) = (q.matmul(&x, m, &bias), q.matmul_scalar(&x, m, &bias));
        for (a, b) in yv.iter().zip(&ys) {
            assert!((a - b).abs() < 1e-3, "i4 vnni {a} != scalar {b}");
        }
    }

    #[test]
    fn prequant_matches_matmul_int8_and_int4() {
        let (m, n, k) = (4usize, 130usize, 256usize);
        let x = fill(4, m * k, 1.3);
        let (xq, ascale) = quantize_activations(&x, m, k);
        let w8 = fill(3, n * k, 0.7);
        let q8 = QInt8Matrix::from_f32(&w8, n, k);
        let bias = fill(5, n, 0.2);
        let a = q8.matmul(&x, m, &bias);
        let b = q8.matmul_prequant(&xq, &ascale, m, &bias);
        assert_eq!(a, b, "int8 prequant != matmul");
        let q4 = QInt4Matrix::from_f32(&w8, n, k);
        let c = q4.matmul(&x, m, &bias);
        let d = q4.matmul_prequant(&xq, &ascale, m, &bias);
        assert_eq!(c, d, "int4 prequant != matmul");
    }

    #[test]
    fn handles_non_multiple_of_64_k() {
        let (m, n, k) = (3usize, 64usize, 100usize);
        let w = fill(6, n * k, 1.0);
        let x = fill(7, m * k, 1.0);
        let q = QInt8Matrix::from_f32(&w, n, k);
        let c = cos(&q.matmul(&x, m, &[]), &f32_ref(&x, m, &w, n, k));
        assert!(c > 0.99, "cos {c} with tail K");
    }
}
