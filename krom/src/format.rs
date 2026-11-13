//! On-disk model format.
//!
//! A model is a directory of memory-mapped files:
//!
//! - `vocab.fst`: folded word → id, ids are lexicographic ranks.
//! - `ngram.fst`: 1–3-gram keys of 3-byte big-endian word ids → quantized
//!   log10 probability.
//! - `meta.txt`: `key=value` lines (format version, quantization, …).

use std::fmt;
use std::fs;
use std::path::Path;

use crate::Error;

pub const FORMAT_VERSION: u32 = 1;
pub const ID_BYTES: usize = 3;
/// Largest id that fits [`ID_BYTES`] and the 21-bit packed count keys.
pub const MAX_ID: u32 = (1 << 21) - 2;

/// Default quantization levels per log10 unit. Coarse steps keep FST outputs
/// small and repetitive; 32 costs no measurable ranking accuracy.
pub const DEFAULT_SCALE: u32 = 32;

pub fn push_id(key: &mut Vec<u8>, id: u32) {
    debug_assert!(id <= MAX_ID);
    key.extend_from_slice(&id.to_be_bytes()[1..]);
}

/// Quantize a log10 probability (≤ 0) to `-logp · scale`.
pub fn pack(logp: f32, scale: u32) -> u64 {
    (-logp * scale as f32).round().clamp(0.0, u16::MAX as f32) as u64
}

pub fn unpack(value: u64, scale: u32) -> f32 {
    -(value as f32) / scale as f32
}

#[derive(Clone, Debug, PartialEq)]
pub struct Meta {
    /// log10 Stupid Backoff penalty per backoff step.
    pub backoff: f32,
    /// log10 score of words missing from the model.
    pub unk_logp: f32,
    /// Training tokens.
    pub tokens: u64,
    /// Quantization levels per log10 unit of `ngram.fst` values.
    pub scale: u32,
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            backoff: 0.4f32.log10(),
            unk_logp: -9.0,
            tokens: 0,
            scale: DEFAULT_SCALE,
        }
    }
}

impl Meta {
    pub fn read(dir: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(dir.join("meta.txt"))?;
        let mut meta = Meta::default();
        let mut version = None;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| Error::Format(format!("meta.txt: {line}")))?;
            let bad = || Error::Format(format!("meta.txt: bad value for {key}"));
            match key {
                "format" => version = Some(value.parse::<u32>().map_err(|_| bad())?),
                "backoff" => meta.backoff = value.parse().map_err(|_| bad())?,
                "unk_logp" => meta.unk_logp = value.parse().map_err(|_| bad())?,
                "tokens" => meta.tokens = value.parse().map_err(|_| bad())?,
                "scale" => meta.scale = value.parse().map_err(|_| bad())?,
                _ => {}
            }
        }
        match version {
            Some(FORMAT_VERSION) if meta.scale > 0 => Ok(meta),
            other => Err(Error::Format(format!("unsupported model format {other:?}"))),
        }
    }

    pub fn write(&self, dir: &Path) -> Result<(), Error> {
        fs::write(dir.join("meta.txt"), self.to_string())?;
        Ok(())
    }
}

impl fmt::Display for Meta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "format={FORMAT_VERSION}")?;
        writeln!(f, "backoff={}", self.backoff)?;
        writeln!(f, "unk_logp={}", self.unk_logp)?;
        writeln!(f, "tokens={}", self.tokens)?;
        writeln!(f, "scale={}", self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        for p in [-0.0, -1.25, -7.9, -12.0] {
            assert!((unpack(pack(p, 2048), 2048) - p).abs() < 1e-3);
            assert!((unpack(pack(p, 32), 32) - p).abs() <= 0.5 / 32.0);
        }
    }

    #[test]
    fn meta_roundtrip() {
        let meta = Meta {
            tokens: 42,
            scale: 64,
            ..Meta::default()
        };
        let dir = std::env::temp_dir().join(format!("krom-meta-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        meta.write(&dir).unwrap();
        assert_eq!(Meta::read(&dir).unwrap(), meta);
        fs::remove_dir_all(&dir).unwrap();
    }
}
