//! Context-based ranking of Bengali word suggestions.
//!
//! `krom` does not generate words. A suggestion engine such as `upodesh`
//! produces candidates for what the user is typing; `krom` orders them by how
//! likely each is to follow the previous words, using a backoff trigram model
//! stored in memory-mapped FSTs (optionally backed by KenLM), with optional
//! stem-factored history tables for sparse contexts.
//!
//! ```no_run
//! let ranker = krom_experiment::Ranker::open("model".as_ref())?;
//! let candidates = ["সামনে", "সাথে", "সময়"];
//! for r in ranker.rank(&["আমি", "বাড়ির"], &candidates) {
//!     println!("{} {:.3}", r.word, r.score);
//! }
//! # Ok::<(), krom_experiment::Error>(())
//! ```

use std::fmt;
use std::io;
use std::path::Path;

pub mod format;
#[cfg(feature = "kenlm")]
pub mod kenlm;
pub mod ngram;
pub mod normalize;
pub mod scorer;
pub mod stem;
pub mod stem_model;
pub mod tokenize;

#[cfg(feature = "build")]
pub mod build;

pub use ngram::NgramModel;
pub use scorer::{Score, Scorer};
pub use stem_model::StemModel;

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

/// Default log10 penalty for using the stem tables over surface backoff.
pub const DEFAULT_STEM_PENALTY: f32 = -0.4;

/// Orders candidates by their score given the preceding words.
pub struct Ranker<S = NgramModel> {
    scorer: S,
    stems: Option<StemModel>,
    stem_penalty: f32,
}

impl Ranker<NgramModel> {
    /// Open a model directory written by `krom-build`, including its stem
    /// tables when present.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        Ok(Ranker::new(NgramModel::open(dir)?).with_stems(StemModel::open(dir)?))
    }
}

impl<S: Scorer> Ranker<S> {
    pub fn new(scorer: S) -> Self {
        Self {
            scorer,
            stems: None,
            stem_penalty: DEFAULT_STEM_PENALTY,
        }
    }

    pub fn with_stems(mut self, stems: Option<StemModel>) -> Self {
        self.stems = stems;
        self
    }

    pub fn with_stem_penalty(mut self, penalty: f32) -> Self {
        self.stem_penalty = penalty;
        self
    }

    pub fn scorer(&self) -> &S {
        &self.scorer
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
        let state = self.scorer.state(history);
        let stem_state = self.stems.as_ref().map(|s| s.state(history));
        let mut ranked: Vec<Ranked<'a>> = candidates
            .iter()
            .enumerate()
            .map(|(index, word)| {
                let word = word.as_ref();
                let score = self.score_folded(&state, stem_state.as_ref(), &normalize::fold(word));
                Ranked { index, word, score }
            })
            .collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index)));
        ranked
    }

    /// Score one word given the preceding words.
    pub fn score(&self, context: &[&str], word: &str) -> f32 {
        let history = history_from_words(context);
        let state = self.scorer.state(&history);
        let stem_state = self.stems.as_ref().map(|s| s.state(&history));
        self.score_folded(&state, stem_state.as_ref(), &normalize::fold(word))
    }

    fn score_folded(
        &self,
        state: &S::State,
        stem_state: Option<&stem_model::StemState>,
        word: &str,
    ) -> f32 {
        let surface = self.scorer.score(state, word);
        match (&self.stems, stem_state) {
            (Some(stems), Some(st)) => match stems.score(st, word) {
                Some((logp, order)) if order > surface.order => {
                    surface.logp.max(logp + self.stem_penalty)
                }
                _ => surface.logp,
            },
            _ => surface.logp,
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
