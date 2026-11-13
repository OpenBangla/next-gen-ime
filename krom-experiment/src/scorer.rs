//! Scoring interface shared by the FST and KenLM backends.

/// A candidate's log10 score and the n-gram order that produced it
/// (0 = unknown word, 1 = unigram, …).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub logp: f32,
    pub order: u8,
}

/// Scores folded words given a folded history.
///
/// `state` is computed once per ranking call so per-candidate scoring only
/// does the final lookups.
pub trait Scorer {
    type State;

    /// `history` is oldest → newest, may start with `<s>`.
    fn state(&self, history: &[String]) -> Self::State;

    fn score(&self, state: &Self::State, word: &str) -> Score;
}
