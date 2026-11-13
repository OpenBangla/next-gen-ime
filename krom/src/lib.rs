//! Context-based ranking of Bengali word suggestions.
//!
//! `krom` does not generate words. A suggestion engine such as `upodesh`
//! produces candidates for what the user is typing; `krom` orders them by how
//! likely each is to follow the previous words, using a Stupid Backoff
//! trigram model stored in memory-mapped FSTs.
//!
//! ```no_run
//! let ranker = krom::Ranker::open("model".as_ref())?;
//! let candidates = ["সামনে", "সাথে", "সময়"];
//! for r in ranker.rank(&["আমি", "বাড়ির"], &candidates) {
//!     println!("{} {:.3}", r.word, r.score);
//! }
//! # Ok::<(), krom::Error>(())
//! ```

use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;

use fst::Map;
use memmap2::Mmap;

use crate::format::{ID_BYTES, Meta};

#[cfg(feature = "build")]
pub mod build;
pub mod format;
pub mod normalize;
pub mod tokenize;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Fst(fst::Error),
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Fst(e) => write!(f, "FST error: {e}"),
            Error::Format(msg) => write!(f, "invalid model: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<fst::Error> for Error {
    fn from(e: fst::Error) -> Self {
        Error::Fst(e)
    }
}

/// A ranked candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranked<'a> {
    /// Position in the input candidate list.
    pub index: usize,
    pub word: &'a str,
    /// log10 score; higher is better.
    pub score: f32,
}

fn mmap_fst(path: &Path) -> Result<Map<Mmap>, Error> {
    let file = File::open(path)?;
    // SAFETY: model files are written once by `krom-build` and not modified
    // while mapped; a concurrent writer would be a usage error.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(Map::new(mmap)?)
}

/// History resolved against the vocabulary: `[w₋₂, w₋₁]` ids, `None` when
/// absent or out of vocabulary.
type History = [Option<u32>; 2];

/// Orders candidates by their Stupid Backoff trigram score given the
/// preceding words.
pub struct Ranker {
    vocab: Map<Mmap>,
    ngrams: Map<Mmap>,
    meta: Meta,
    max_order: usize,
}

impl Ranker {
    /// Open a model directory written by `krom-build`.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        Ok(Self {
            vocab: mmap_fst(&dir.join("vocab.fst"))?,
            ngrams: mmap_fst(&dir.join("ngram.fst"))?,
            meta: Meta::read(dir)?,
            max_order: 3,
        })
    }

    /// Cap the n-gram order used for scoring (1 = word frequency only).
    pub fn with_max_order(mut self, order: usize) -> Self {
        self.max_order = order.clamp(1, 3);
        self
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Rank `candidates` given the preceding words (oldest first).
    ///
    /// An empty `context` means the start of a sentence. Results are sorted
    /// by score, ties keep the input order.
    pub fn rank<'a, T: AsRef<str>>(
        &self,
        context: &[&str],
        candidates: &'a [T],
    ) -> Vec<Ranked<'a>> {
        self.rank_history(&history_from_words(context), candidates)
    }

    /// Like [`rank`](Self::rank), taking free text (sentence punctuation is
    /// respected) as context.
    pub fn rank_text<'a, T: AsRef<str>>(&self, text: &str, candidates: &'a [T]) -> Vec<Ranked<'a>> {
        self.rank_history(&tokenize::history(text, 2), candidates)
    }

    /// Rank with an already tokenized, folded history (see [`tokenize`]).
    pub fn rank_history<'a, T: AsRef<str>>(
        &self,
        history: &[String],
        candidates: &'a [T],
    ) -> Vec<Ranked<'a>> {
        let history = self.resolve(history);
        let mut ranked: Vec<Ranked<'a>> = candidates
            .iter()
            .enumerate()
            .map(|(index, word)| {
                let word = word.as_ref();
                let score = self.score_folded(&history, &normalize::fold(word));
                Ranked { index, word, score }
            })
            .collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index)));
        ranked
    }

    /// Score one word given the preceding words.
    pub fn score(&self, context: &[&str], word: &str) -> f32 {
        let history = self.resolve(&history_from_words(context));
        self.score_folded(&history, &normalize::fold(word))
    }

    fn id(&self, word: &str) -> Option<u32> {
        self.vocab.get(word).map(|id| id as u32)
    }

    fn get(&self, ids: &[u32]) -> Option<f32> {
        let mut key = [0u8; 3 * ID_BYTES];
        for (chunk, id) in key.as_chunks_mut::<ID_BYTES>().0.iter_mut().zip(ids) {
            chunk.copy_from_slice(&id.to_be_bytes()[1..]);
        }
        self.ngrams
            .get(&key[..ids.len() * ID_BYTES])
            .map(|v| format::unpack(v, self.meta.scale))
    }

    fn resolve(&self, history: &[String]) -> History {
        let mut ids = [None, None];
        if self.max_order >= 2 {
            ids[1] = history.last().and_then(|w| self.id(w));
        }
        if self.max_order >= 3 && ids[1].is_some() && history.len() >= 2 {
            ids[0] = self.id(&history[history.len() - 2]);
        }
        ids
    }

    /// Stupid Backoff: the highest-order match, minus a constant penalty per
    /// order backed off.
    fn score_folded(&self, history: &History, word: &str) -> f32 {
        let Some(id) = self.id(word) else {
            return self.meta.unk_logp;
        };
        let mut penalty = 0.0;
        if let [Some(a), Some(b)] = *history {
            if let Some(p) = self.get(&[a, b, id]) {
                return p;
            }
            penalty += self.meta.backoff;
        }
        if let Some(b) = history[1] {
            if let Some(p) = self.get(&[b, id]) {
                return penalty + p;
            }
            penalty += self.meta.backoff;
        }
        match self.get(&[id]) {
            Some(p) => penalty + p,
            None => self.meta.unk_logp,
        }
    }
}

fn history_from_words(context: &[&str]) -> Vec<String> {
    let mut history: Vec<String> = context
        .iter()
        .filter_map(|w| tokenize::classify(w).map(|t| t.into_owned()))
        .collect();
    if history.is_empty() {
        history.push(tokenize::BOS.to_owned());
    }
    let keep = history.len().saturating_sub(2);
    history.drain(..keep);
    history
}
