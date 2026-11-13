//! Stem-factored history tables: `P(w | stem₋₂, stem₋₁, class₋₁)` and
//! `P(w | stem₋₁, class₋₁)` over surface words `w`.

use std::path::Path;

use fst::Map;
use memmap2::Mmap;

use crate::Error;
use crate::format::{self, AnalyzerKind, ID_BYTES, Meta, STEM_TAG_BI, STEM_TAG_TRI};
use crate::ngram::mmap_fst;
use crate::stem::{Analyzer, RuleStemmer};

pub struct StemModel {
    vocab: Map<Mmap>,
    stems: Map<Mmap>,
    table: Map<Mmap>,
    analyzer: Box<dyn Analyzer>,
    scale: u32,
}

/// Key prefixes of the history; `None` when a stem is unknown.
#[derive(Clone, Debug, Default)]
pub struct StemState {
    tri: Option<Vec<u8>>,
    bi: Option<Vec<u8>>,
}

/// Build the analyzer recorded in a model's metadata.
pub fn analyzer_for(kind: &AnalyzerKind) -> Result<Box<dyn Analyzer>, Error> {
    match kind {
        AnalyzerKind::Rule => Ok(Box::new(RuleStemmer)),
        #[cfg(feature = "sp-eval")]
        AnalyzerKind::SentencePiece(path) => {
            Ok(Box::new(crate::stem::SpAnalyzer::open(Path::new(path))?))
        }
        #[cfg(not(feature = "sp-eval"))]
        AnalyzerKind::SentencePiece(_) => {
            Err(Error::Format("model needs the `sp-eval` feature".into()))
        }
    }
}

impl StemModel {
    /// Open the stem tables of a model directory, if it has any.
    pub fn open(dir: &Path) -> Result<Option<Self>, Error> {
        let meta = Meta::read(dir)?;
        let Some(kind) = meta.analyzer else {
            return Ok(None);
        };
        Ok(Some(Self {
            vocab: mmap_fst(&dir.join("vocab.fst"))?,
            stems: mmap_fst(&dir.join("stems.fst"))?,
            table: mmap_fst(&dir.join("stem_ngram.fst"))?,
            analyzer: analyzer_for(&kind)?,
            scale: meta.stem_scale,
        }))
    }

    fn stem_id(&self, word: &str) -> Option<(u32, u8)> {
        let (stem, class) = self.analyzer.analyze(word);
        self.stems.get(stem.as_bytes()).map(|id| (id as u32, class))
    }

    pub fn state(&self, history: &[String]) -> StemState {
        let mut state = StemState::default();
        let Some((s1, c1)) = history.last().and_then(|w| self.stem_id(w)) else {
            return state;
        };
        let mut bi = vec![STEM_TAG_BI];
        format::push_id(&mut bi, s1);
        bi.push(c1);
        state.bi = Some(bi);
        if let Some((s2, _)) = history
            .len()
            .checked_sub(2)
            .and_then(|i| self.stem_id(&history[i]))
        {
            let mut tri = vec![STEM_TAG_TRI];
            format::push_id(&mut tri, s2);
            format::push_id(&mut tri, s1);
            tri.push(c1);
            state.tri = Some(tri);
        }
        state
    }

    /// log10 relative frequency and its order (3 or 2), if seen.
    pub fn score(&self, state: &StemState, word: &str) -> Option<(f32, u8)> {
        let id = self.vocab.get(word)? as u32;
        let mut key = [0u8; 16];
        for (prefix, order) in [(&state.tri, 3), (&state.bi, 2)] {
            let Some(prefix) = prefix else { continue };
            let len = prefix.len();
            key[..len].copy_from_slice(prefix);
            key[len..len + ID_BYTES].copy_from_slice(&id.to_be_bytes()[1..]);
            if let Some(value) = self.table.get(&key[..len + ID_BYTES]) {
                return Some((format::unpack(value, self.scale).0, order));
            }
        }
        None
    }
}
