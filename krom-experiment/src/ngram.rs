//! Surface-word backoff n-gram model stored in FSTs.

use std::fs::File;
use std::path::Path;

use fst::Map;
use memmap2::Mmap;

use crate::Error;
use crate::format::{self, ID_BYTES, Meta, Method};
use crate::scorer::{Score, Scorer};

pub(crate) fn mmap_fst(path: &Path) -> Result<Map<Mmap>, Error> {
    let file = File::open(path)?;
    // SAFETY: model files are written once by `krom-build` and not modified
    // while mapped; a concurrent writer would be a usage error.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(Map::new(mmap)?)
}

/// Trigram (or lower) backoff model over folded surface words.
pub struct NgramModel {
    vocab: Map<Mmap>,
    ngrams: Map<Mmap>,
    meta: Meta,
    max_order: usize,
}

/// History resolved against the vocabulary, with its backoff weights.
#[derive(Clone, Copy, Debug)]
pub struct NgramState {
    /// `[w₋₂, w₋₁]` ids; `None` when absent or out of vocabulary.
    history: [Option<u32>; 2],
    /// log10 penalty when backing off from the trigram / bigram context.
    backoff: [f32; 2],
}

impl NgramModel {
    pub fn open(dir: &Path) -> Result<Self, Error> {
        Ok(Self {
            vocab: mmap_fst(&dir.join("vocab.fst"))?,
            ngrams: mmap_fst(&dir.join("ngram.fst"))?,
            meta: Meta::read(dir)?,
            max_order: 3,
        })
    }

    /// Cap the n-gram order used for scoring (1 = unigram-only baseline).
    pub fn with_max_order(mut self, order: usize) -> Self {
        self.max_order = order.clamp(1, 3);
        self
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Id of a folded word.
    pub fn id(&self, word: &str) -> Option<u32> {
        self.vocab.get(word).map(|id| id as u32)
    }

    fn get(&self, ids: &[u32]) -> Option<(f32, Option<f32>)> {
        let mut key = [0u8; 3 * ID_BYTES];
        for (chunk, id) in key.as_chunks_mut::<ID_BYTES>().0.iter_mut().zip(ids) {
            chunk.copy_from_slice(&id.to_be_bytes()[1..]);
        }
        let scale = self.meta.scale;
        self.ngrams
            .get(&key[..ids.len() * ID_BYTES])
            .map(|v| format::unpack(v, scale))
    }

    fn context_backoff(&self, ids: &[u32]) -> f32 {
        match self.meta.method {
            Method::StupidBackoff => self.meta.sb_backoff,
            Method::Arpa => self.get(ids).and_then(|(_, bo)| bo).unwrap_or(0.0),
        }
    }
}

impl Scorer for NgramModel {
    type State = NgramState;

    fn state(&self, history: &[String]) -> NgramState {
        let usable = self.max_order - 1;
        let mut ids = [None, None];
        if usable >= 1 {
            ids[1] = history.last().and_then(|w| self.id(w));
        }
        if usable >= 2 && ids[1].is_some() && history.len() >= 2 {
            ids[0] = self.id(&history[history.len() - 2]);
        }
        let mut backoff = [0.0, 0.0];
        if let [Some(a), Some(b)] = ids {
            backoff[0] = self.context_backoff(&[a, b]);
        }
        if let Some(b) = ids[1] {
            backoff[1] = self.context_backoff(&[b]);
        }
        NgramState {
            history: ids,
            backoff,
        }
    }

    fn score(&self, state: &NgramState, word: &str) -> Score {
        let unknown = Score {
            logp: self.meta.unk_logp,
            order: 0,
        };
        let Some(id) = self.id(word) else {
            return unknown;
        };
        let mut acc = 0.0;
        if let [Some(a), Some(b)] = state.history {
            if let Some((p, _)) = self.get(&[a, b, id]) {
                return Score { logp: p, order: 3 };
            }
            acc += state.backoff[0];
        }
        if let Some(b) = state.history[1] {
            if let Some((p, _)) = self.get(&[b, id]) {
                return Score {
                    logp: acc + p,
                    order: 2,
                };
            }
            acc += state.backoff[1];
        }
        match self.get(&[id]) {
            Some((p, _)) => Score {
                logp: acc + p,
                order: 1,
            },
            None => unknown,
        }
    }
}
