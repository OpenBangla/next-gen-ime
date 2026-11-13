//! Corpus and history tokenization.
//!
//! Follows upodesh's `corpus_count` convention (whitespace split, punctuation
//! trim) and adds sentence splitting and token classes so the counter and the
//! ranker see identical tokens.

use std::borrow::Cow;

use crate::normalize::fold;

pub const BOS: &str = "<s>";
pub const EOS: &str = "</s>";
/// Out-of-vocabulary word (assigned by model building, never by the tokenizer).
pub const UNK: &str = "<unk>";
/// Non-Bengali token (Latin, mixed script).
pub const FOREIGN: &str = "<x>";
/// Number in Bengali or ASCII digits.
pub const NUM: &str = "<num>";

/// Sentence terminators. `৷` (U+09F7) is widely misused as a danda.
const TERMINATORS: &[char] = &['।', '॥', '৷', '?', '!', '؟'];

const PUNCT: &[char] = &[
    ',', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '‘', '’', '“', '”', '«', '»', '—', '–',
    '-', '…', '/', '\\', '*', '#', '@', '%', '&', '+', '=', '<', '>', '|', '~', '`', '_', '.', '·',
];

fn is_bengali(c: char) -> bool {
    ('\u{0980}'..='\u{09FF}').contains(&c)
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || ('\u{09E6}'..='\u{09EF}').contains(&c)
}

/// Map one whitespace-free piece to its token, or `None` when only
/// punctuation remains.
pub fn classify(piece: &str) -> Option<Cow<'_, str>> {
    let piece = piece.trim_matches(PUNCT);
    if piece.is_empty() {
        return None;
    }
    if piece.chars().any(is_digit) && piece.chars().all(|c| is_digit(c) || ",.:/".contains(c)) {
        return Some(Cow::Borrowed(NUM));
    }
    let key = fold(piece);
    if key.chars().all(is_bengali) && !key.chars().any(is_digit) {
        Some(key)
    } else {
        Some(Cow::Borrowed(FOREIGN))
    }
}

/// Split text into sentences of tokens (without `<s>`/`</s>`).
pub fn sentences(text: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    for word in text.split_whitespace() {
        // A '.' ends a sentence unless it closes a short abbreviation (মো., ডা.).
        let ends_with_period =
            word.ends_with('.') && word.trim_end_matches('.').chars().count() > 2;
        let mut parts = word.split(TERMINATORS).peekable();
        while let Some(part) = parts.next() {
            for sub in part.split(['-', '–', '—', '/']) {
                if let Some(token) = classify(sub) {
                    current.push(token.into_owned());
                }
            }
            if parts.peek().is_some() && !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
        }
        if ends_with_period && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Tokenize free-form preceding text into the history the ranker uses.
///
/// Returns the last `max` tokens of the current sentence, preceded by `<s>`
/// when the sentence start is within reach (including empty input).
pub fn history(text: &str, max: usize) -> Vec<String> {
    let ends_sentence = text.trim_end().ends_with(TERMINATORS);
    let mut sents = sentences(text);
    let current = if ends_sentence {
        Vec::new()
    } else {
        sents.pop().unwrap_or_default()
    };
    let mut hist: Vec<String> = Vec::with_capacity(max);
    if current.len() < max {
        hist.push(BOS.to_owned());
    }
    let skip = current.len().saturating_sub(max - hist.len());
    hist.extend(current.into_iter().skip(skip));
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences_and_trims() {
        let s = sentences("“বসে থাকেন, নির্বাচন ৫ বছর পর হবে। নানান সুতার টানে!");
        assert_eq!(
            s,
            vec![
                vec!["বসে", "থাকেন", "নির্বাচন", NUM, "বছর", "পর", "হবে"],
                vec!["নানান", "সুতার", "টানে"],
            ]
        );
    }

    #[test]
    fn foreign_and_hyphen() {
        assert_eq!(
            sentences("ভ্যালেন্টাইনস ডে Valentine আইন-শৃঙ্খলা"),
            vec![vec!["ভ্যালেন্টাইনস", "ডে", FOREIGN, "আইন", "শৃঙ্খলা"]]
        );
    }

    #[test]
    fn abbreviation_keeps_sentence() {
        assert_eq!(sentences("মো. রহিম এলেন.").len(), 1);
    }

    #[test]
    fn history_window() {
        assert_eq!(history("", 2), vec![BOS]);
        assert_eq!(history("আমি", 2), vec![BOS, "আমি"]);
        assert_eq!(history("সে বলল। আমি বাড়ির", 2), vec!["আমি", "বাড়ির"]);
        assert_eq!(history("সে বলল।", 2), vec![BOS]);
    }
}
