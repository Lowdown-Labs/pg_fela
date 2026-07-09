use std::collections::HashMap;

use safetensors::{Dtype, SafeTensors};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum LoadError {
    Parse(String),
    UnsupportedDtype(Dtype),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Parse(e) => write!(f, "safetensors parse error: {e}"),
            LoadError::UnsupportedDtype(d) => write!(f, "unsupported dtype: {d:?}"),
        }
    }
}

impl std::error::Error for LoadError {}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let val = match exp {
        0 => (frac as f32) * 2f32.powi(-24),
        0x1f => {
            if frac == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

fn to_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>, LoadError> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        other => return Err(LoadError::UnsupportedDtype(other)),
    })
}

pub fn dtype_to_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>, LoadError> {
    to_f32(dtype, bytes)
}

pub fn load_f32_map(buffer: &[u8]) -> Result<HashMap<String, Tensor>, LoadError> {
    let st = SafeTensors::deserialize(buffer).map_err(|e| LoadError::Parse(e.to_string()))?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        let data = to_f32(view.dtype(), view.data())?;
        out.insert(
            name.to_string(),
            Tensor {
                shape: view.shape().to_vec(),
                data,
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_one() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn f16_one() {
        assert!((f16_to_f32(0x3c00) - 1.0).abs() < 1e-6);
    }
}
