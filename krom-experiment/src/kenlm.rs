//! KenLM runtime scorer (feature `kenlm`), for comparison with the FST
//! backend. Loads ARPA or `build_binary` files through `kenlm-rs`.

use std::path::Path;

use kenlm_rs::{Model, State};

use crate::Error;
use crate::scorer::{Score, Scorer};
use crate::tokenize::BOS;

pub struct KenlmModel {
    model: Model,
}

impl KenlmModel {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let model = Model::new(path).map_err(|e| Error::Format(format!("kenlm: {e}")))?;
        Ok(Self { model })
    }
}

impl Scorer for KenlmModel {
    type State = State;

    fn state(&self, history: &[String]) -> State {
        let mut words = history;
        let mut state = if words.first().is_some_and(|w| w == BOS) {
            words = &words[1..];
            self.model.begin_sentence_state()
        } else {
            self.model.null_context_state()
        };
        let mut next = self.model.null_context_state();
        for word in words {
            let index = self
                .model
                .index(word)
                .unwrap_or_else(|_| self.model.not_found_index());
            if self.model.base_score(&state, index, &mut next).is_ok() {
                std::mem::swap(&mut state, &mut next);
            }
        }
        state
    }

    fn score(&self, state: &State, word: &str) -> Score {
        let unknown = self.model.not_found_index();
        let index = self.model.index(word).unwrap_or(unknown);
        let mut out = self.model.null_context_state();
        match self.model.base_full_score(state, index, &mut out) {
            Ok(full) if index != unknown => Score {
                logp: full.log_prob,
                order: full.ngram_length,
            },
            Ok(full) => Score {
                logp: full.log_prob,
                order: 0,
            },
            Err(_) => Score {
                logp: f32::NEG_INFINITY,
                order: 0,
            },
        }
    }
}
