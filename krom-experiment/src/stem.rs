//! History-side word analysis: `word → (stem, suffix class)`.
//!
//! The ranker always scores *surface* candidates. Stems only generalize the
//! history: `P(w | stem₋₂, stem₋₁, class(suffix₋₁))` backs off from the
//! surface trigram. Suffix classes keep the case/tense signal of the previous
//! word (বাড়ির → সামনে) while merging spelling variants (`র`/`ের`/`এর`).

use std::borrow::Cow;
use std::sync::OnceLock;

use crate::normalize::fold;

/// Suffix class of a word; `NONE` when nothing was stripped.
pub type SuffixClass = u8;

pub mod class {
    use super::SuffixClass;
    pub const NONE: SuffixClass = 0;
    pub const GEN: SuffixClass = 1;
    pub const PL_GEN: SuffixClass = 2;
    pub const OBJ: SuffixClass = 3;
    pub const LOC: SuffixClass = 4;
    pub const PL: SuffixClass = 5;
    pub const CLS: SuffixClass = 6;
    pub const EMPH_O: SuffixClass = 7;
    pub const EMPH_I: SuffixClass = 8;
    pub const V_PROG: SuffixClass = 9;
    pub const V_PERF: SuffixClass = 10;
    pub const V_PAST: SuffixClass = 11;
    pub const V_HAB: SuffixClass = 12;
    pub const V_FUT: SuffixClass = 13;
    pub const V_PRES: SuffixClass = 14;
    pub const V_CAUS: SuffixClass = 15;
    /// Remainder that is not a known suffix (SentencePiece analyzer only).
    pub const OTHER: SuffixClass = 255;
}

/// Splits folded words into a stem and a suffix class.
pub trait Analyzer: Send + Sync {
    fn analyze<'a>(&self, word: &'a str) -> (Cow<'a, str>, SuffixClass);
}

/// When a suffix may be stripped.
#[derive(Clone, Copy, PartialEq)]
enum Cond {
    Any,
    /// Preceded by a vowel sign or independent vowel (বাংলা+র, বাংলা+য়).
    AfterVowel,
    /// Preceded by a consonant: verb roots (কর+লো, but not ভা+লো).
    AfterConsonant,
}

struct Rule {
    suffix: String,
    class: SuffixClass,
    cond: Cond,
}

/// Suffix table in source spelling; folded and length-sorted on first use.
/// Nukta letters are escaped so the source encoding cannot drift.
const TABLE: &[(&str, SuffixClass, Cond)] = {
    use Cond::*;
    use class::*;
    &[
        // Nominal case and number.
        ("র", GEN, AfterVowel),
        ("ের", GEN, Any),
        ("এর", GEN, Any),
        ("\u{09AF}\u{09BC}ের", GEN, AfterVowel),
        ("টার", GEN, Any),
        ("টির", GEN, Any),
        ("গুলোর", PL_GEN, Any),
        ("গুলির", PL_GEN, Any),
        ("দের", PL_GEN, Any),
        ("েদের", PL_GEN, Any),
        ("কে", OBJ, Any),
        ("দেরকে", OBJ, Any),
        ("গুলোকে", OBJ, Any),
        ("টাকে", OBJ, Any),
        ("টিকে", OBJ, Any),
        ("তে", LOC, Any),
        ("েতে", LOC, Any),
        ("গুলোতে", LOC, Any),
        ("টাতে", LOC, Any),
        ("\u{09AF}\u{09BC}", LOC, AfterVowel),
        ("রা", PL, AfterVowel),
        ("েরা", PL, Any),
        ("গুলো", PL, Any),
        ("গুলি", PL, Any),
        ("গুলা", PL, Any),
        ("সমূহ", PL, Any),
        ("টা", CLS, Any),
        ("টি", CLS, Any),
        ("খানা", CLS, Any),
        ("খানি", CLS, Any),
        ("ও", EMPH_O, Any),
        ("ই", EMPH_I, Any),
        // Verbal inflection.
        ("ছি", V_PROG, AfterConsonant),
        ("ছে", V_PROG, AfterConsonant),
        ("ছেন", V_PROG, AfterConsonant),
        ("ছো", V_PROG, AfterConsonant),
        ("ছিস", V_PROG, AfterConsonant),
        ("ছিল", V_PROG, AfterConsonant),
        ("ছিলো", V_PROG, AfterConsonant),
        ("ছিলাম", V_PROG, AfterConsonant),
        ("ছিলেন", V_PROG, AfterConsonant),
        ("েছি", V_PERF, AfterConsonant),
        ("েছে", V_PERF, AfterConsonant),
        ("েছেন", V_PERF, AfterConsonant),
        ("েছো", V_PERF, AfterConsonant),
        ("েছিল", V_PERF, AfterConsonant),
        ("েছিলো", V_PERF, AfterConsonant),
        ("েছিলাম", V_PERF, AfterConsonant),
        ("েছিলেন", V_PERF, AfterConsonant),
        ("লাম", V_PAST, AfterConsonant),
        ("লেন", V_PAST, AfterConsonant),
        ("লো", V_PAST, AfterConsonant),
        ("লি", V_PAST, AfterConsonant),
        ("তাম", V_HAB, AfterConsonant),
        ("তেন", V_HAB, AfterConsonant),
        ("তো", V_HAB, AfterConsonant),
        ("তিস", V_HAB, AfterConsonant),
        ("বে", V_FUT, AfterConsonant),
        ("বেন", V_FUT, AfterConsonant),
        ("বো", V_FUT, AfterConsonant),
        ("েন", V_PRES, AfterConsonant),
        ("ানো", V_CAUS, AfterConsonant),
    ]
};

/// Minimum stem length in code points after stripping.
const MIN_STEM_CHARS: usize = 2;
/// Strip at most this many suffixes (গাছ+গুলো+তে).
const MAX_STRIPS: usize = 2;

fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| {
        let mut rules: Vec<Rule> = TABLE
            .iter()
            .map(|&(suffix, class, cond)| Rule {
                suffix: fold(suffix).into_owned(),
                class,
                cond,
            })
            .collect();
        rules.sort_by_key(|r| std::cmp::Reverse(r.suffix.chars().count()));
        rules
    })
}

fn is_vowel_like(c: char) -> bool {
    matches!(c, '\u{0985}'..='\u{0994}' | '\u{09BE}'..='\u{09CC}' | '\u{09D7}')
}

fn is_consonant_like(c: char) -> bool {
    // Consonants, khanda-ta, and the nukta that ends decomposed ড়/ঢ়/য়.
    matches!(c, '\u{0995}'..='\u{09B9}' | '\u{09CE}' | '\u{09BC}')
}

/// The longest strippable suffix rule for `word`, if any.
fn match_rule(word: &str) -> Option<&'static Rule> {
    rules().iter().find(|rule| {
        let Some(stem) = word.strip_suffix(rule.suffix.as_str()) else {
            return false;
        };
        let last = stem.chars().next_back();
        stem.chars().count() >= MIN_STEM_CHARS
            && last != Some('\u{09CD}')
            && match rule.cond {
                Cond::Any => true,
                Cond::AfterVowel => last.is_some_and(is_vowel_like),
                Cond::AfterConsonant => last.is_some_and(is_consonant_like),
            }
    })
}

/// Rule-based suffix stripper (Mahmud et al. 2014 style noun rules, as in
/// OpenBangla `mul`, extended with verb paradigms and suffix classes).
#[derive(Clone, Copy, Debug, Default)]
pub struct RuleStemmer;

impl RuleStemmer {
    /// Class of a bare suffix string, e.g. a SentencePiece remainder.
    pub fn suffix_class(suffix: &str) -> Option<SuffixClass> {
        rules().iter().find(|r| r.suffix == suffix).map(|r| r.class)
    }
}

impl Analyzer for RuleStemmer {
    fn analyze<'a>(&self, word: &'a str) -> (Cow<'a, str>, SuffixClass) {
        if word.starts_with('<') {
            return (Cow::Borrowed(word), class::NONE);
        }
        let mut stem = word;
        let mut outer = class::NONE;
        for _ in 0..MAX_STRIPS {
            let Some(rule) = match_rule(stem) else { break };
            if outer == class::NONE {
                outer = rule.class;
            }
            stem = &stem[..stem.len() - rule.suffix.len()];
        }
        (Cow::Borrowed(stem), outer)
    }
}

/// "Stem = first SentencePiece piece" analyzer, for the ablation only.
#[cfg(feature = "sp-eval")]
pub struct SpAnalyzer {
    sp: sentencepiece::SentencePieceProcessor,
}

#[cfg(feature = "sp-eval")]
impl SpAnalyzer {
    pub fn open(model: &std::path::Path) -> Result<Self, crate::Error> {
        let sp = sentencepiece::SentencePieceProcessor::open(model)
            .map_err(|e| crate::Error::Format(format!("sentencepiece: {e}")))?;
        Ok(Self { sp })
    }
}

// SAFETY-free: the processor is only read after construction.
#[cfg(feature = "sp-eval")]
unsafe impl Sync for SpAnalyzer {}

#[cfg(feature = "sp-eval")]
impl Analyzer for SpAnalyzer {
    fn analyze<'a>(&self, word: &'a str) -> (Cow<'a, str>, SuffixClass) {
        if word.starts_with('<') {
            return (Cow::Borrowed(word), class::NONE);
        }
        let Ok(pieces) = self.sp.encode(word) else {
            return (Cow::Borrowed(word), class::NONE);
        };
        let mut pieces = pieces
            .into_iter()
            .map(|p| p.piece.trim_start_matches('▁').to_owned());
        let first = pieces.next().unwrap_or_default();
        let rest: String = pieces.collect();
        if first.is_empty() || rest.is_empty() {
            return (Cow::Borrowed(word), class::NONE);
        }
        let class = RuleStemmer::suffix_class(&fold(&rest)).unwrap_or(class::OTHER);
        (Cow::Owned(fold(&first).into_owned()), class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stem(word: &str) -> (String, SuffixClass) {
        let key = fold(word);
        let (s, c) = RuleStemmer.analyze(&key);
        (s.into_owned(), c)
    }

    #[test]
    fn nouns() {
        assert_eq!(stem("মানুষকে"), (fold("মানুষ").into_owned(), class::OBJ));
        assert_eq!(stem("গাছগুলোতে"), (fold("গাছ").into_owned(), class::LOC));
        assert_eq!(
            stem("বা\u{09DC}ির"),
            (fold("বা\u{09DC}ি").into_owned(), class::GEN)
        );
        assert_eq!(stem("বাংলা\u{09DF}"), (fold("বাংলা").into_owned(), class::LOC));
        assert_eq!(stem("লোকেরা"), (fold("লোক").into_owned(), class::PL));
        assert_eq!(stem("প্রয়োজন").1, class::NONE);
    }

    #[test]
    fn verbs() {
        assert_eq!(stem("করছিলাম").1, class::V_PROG);
        assert_eq!(stem("করেছেন"), (fold("কর").into_owned(), class::V_PERF));
        assert_eq!(stem("বললেন"), (fold("বল").into_owned(), class::V_PAST));
    }

    #[test]
    fn short_and_special_words_kept() {
        assert_eq!(stem("হবে"), (fold("হবে").into_owned(), class::NONE));
        assert_eq!(stem("ঘর"), (fold("ঘর").into_owned(), class::NONE));
        assert_eq!(stem("ভালো"), (fold("ভালো").into_owned(), class::NONE));
        assert_eq!(
            RuleStemmer.analyze("<s>"),
            (Cow::Borrowed("<s>"), class::NONE)
        );
    }
}
