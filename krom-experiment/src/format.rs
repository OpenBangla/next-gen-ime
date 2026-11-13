//! On-disk model format.
//!
//! A model is a directory of memory-mapped files:
//!
//! - `vocab.fst`: folded word → id, ids are lexicographic ranks.
//! - `ngram.fst`: 1–3-gram keys of 3-byte big-endian word ids → packed value.
//! - `stems.fst` + `stem_ngram.fst` (optional): stem-factored history tables.
//! - `meta.txt`: `key=value` lines (format version, method, analyzer, …).

use std::fmt;
use std::fs;
use std::path::Path;

use crate::Error;

pub const FORMAT_VERSION: u32 = 1;
pub const ID_BYTES: usize = 3;
/// Largest id that fits [`ID_BYTES`] and the 21-bit packed count keys.
pub const MAX_ID: u32 = (1 << 21) - 2;

/// Key tag bytes in `stem_ngram.fst`.
pub const STEM_TAG_BI: u8 = 2;
pub const STEM_TAG_TRI: u8 = 3;

/// Default quantization levels per log10 unit (full 16-bit precision).
pub const DEFAULT_SCALE: u32 = 2048;

pub fn push_id(key: &mut Vec<u8>, id: u32) {
    debug_assert!(id <= MAX_ID);
    key.extend_from_slice(&id.to_be_bytes()[1..]);
}

/// Pack a log10 probability (≤ 0) and an optional log10 backoff, quantized
/// to `scale` levels per log10 unit.
///
/// Bits 0–15: `-logp · scale`; bits 16–31: `bo · scale` as i16; bit 32: has
/// backoff. Leaf n-grams without backoff stay ≤ 16 bits; a coarse `scale`
/// makes outputs small and repetitive, which lets the FST share states.
pub fn pack(logp: f32, backoff: Option<f32>, scale: u32) -> u64 {
    let scale = scale as f32;
    let p = (-logp * scale).round().clamp(0.0, u16::MAX as f32) as u64;
    match backoff {
        None => p,
        Some(bo) => {
            let b = (bo * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16 as u16;
            p | (b as u64) << 16 | 1 << 32
        }
    }
}

pub fn unpack(value: u64, scale: u32) -> (f32, Option<f32>) {
    let scale = scale as f32;
    let logp = -((value & 0xFFFF) as f32) / scale;
    let backoff = (value >> 32 & 1 == 1).then(|| ((value >> 16) as u16 as i16) as f32 / scale);
    (logp, backoff)
}

/// How n-gram scores were estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Stupid Backoff (Brants et al. 2007): relative frequencies, constant
    /// backoff penalty applied at query time.
    StupidBackoff,
    /// Backoff model read from an ARPA file (e.g. KenLM `lmplz` modified
    /// Kneser-Ney), with stored backoff weights.
    Arpa,
}

impl Method {
    fn as_str(self) -> &'static str {
        match self {
            Method::StupidBackoff => "stupid-backoff",
            Method::Arpa => "arpa",
        }
    }
}

/// Which history analyzer built the stem tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnalyzerKind {
    Rule,
    /// SentencePiece first-piece stems; the value is the `.model` path.
    SentencePiece(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Meta {
    pub method: Method,
    /// log10 Stupid Backoff penalty per backoff step.
    pub sb_backoff: f32,
    /// log10 probability of words missing from the model.
    pub unk_logp: f32,
    pub analyzer: Option<AnalyzerKind>,
    pub tokens: u64,
    /// Quantization levels per log10 unit of `ngram.fst` values.
    pub scale: u32,
    /// Quantization levels per log10 unit of `stem_ngram.fst` values.
    pub stem_scale: u32,
}

impl Meta {
    pub fn new(method: Method) -> Self {
        Self {
            method,
            sb_backoff: 0.4f32.log10(),
            unk_logp: -9.0,
            analyzer: None,
            tokens: 0,
            scale: DEFAULT_SCALE,
            stem_scale: DEFAULT_SCALE,
        }
    }

    pub fn read(dir: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(dir.join("meta.txt"))?;
        let mut meta = Meta::new(Method::StupidBackoff);
        let mut version = None;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| Error::Format(format!("meta.txt: {line}")))?;
            let bad = || Error::Format(format!("meta.txt: bad value for {key}"));
            match key {
                "format" => version = Some(value.parse::<u32>().map_err(|_| bad())?),
                "method" => {
                    meta.method = match value {
                        "stupid-backoff" => Method::StupidBackoff,
                        "arpa" => Method::Arpa,
                        _ => return Err(bad()),
                    }
                }
                "sb_backoff" => meta.sb_backoff = value.parse().map_err(|_| bad())?,
                "unk_logp" => meta.unk_logp = value.parse().map_err(|_| bad())?,
                "tokens" => meta.tokens = value.parse().map_err(|_| bad())?,
                "scale" => meta.scale = value.parse().map_err(|_| bad())?,
                "stem_scale" => meta.stem_scale = value.parse().map_err(|_| bad())?,
                "analyzer" => {
                    meta.analyzer = match value.split_once(':') {
                        None if value == "rule" => Some(AnalyzerKind::Rule),
                        Some(("sp", path)) => Some(AnalyzerKind::SentencePiece(path.to_owned())),
                        _ => return Err(bad()),
                    }
                }
                _ => {}
            }
        }
        match version {
            Some(FORMAT_VERSION) => Ok(meta),
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
        writeln!(f, "method={}", self.method.as_str())?;
        writeln!(f, "sb_backoff={}", self.sb_backoff)?;
        writeln!(f, "unk_logp={}", self.unk_logp)?;
        writeln!(f, "tokens={}", self.tokens)?;
        writeln!(f, "scale={}", self.scale)?;
        writeln!(f, "stem_scale={}", self.stem_scale)?;
        match &self.analyzer {
            None => Ok(()),
            Some(AnalyzerKind::Rule) => writeln!(f, "analyzer=rule"),
            Some(AnalyzerKind::SentencePiece(path)) => writeln!(f, "analyzer=sp:{path}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        for (p, bo) in [
            (-0.0, None),
            (-1.25, Some(-0.5)),
            (-7.9, Some(0.3)),
            (-12.0, None),
        ] {
            let (q, qb) = unpack(pack(p, bo, DEFAULT_SCALE), DEFAULT_SCALE);
            assert!((q - p).abs() < 1e-3);
            assert_eq!(qb.is_some(), bo.is_some());
            if let (Some(a), Some(b)) = (qb, bo) {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn meta_roundtrip() {
        let mut meta = Meta::new(Method::Arpa);
        meta.analyzer = Some(AnalyzerKind::SentencePiece("/x/sp.model".into()));
        meta.tokens = 42;
        meta.scale = 32;
        let dir = std::env::temp_dir().join(format!("krom-meta-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        meta.write(&dir).unwrap();
        assert_eq!(Meta::read(&dir).unwrap(), meta);
        fs::remove_dir_all(&dir).unwrap();
    }
}
