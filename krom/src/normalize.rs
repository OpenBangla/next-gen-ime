//! Key folding shared by model building and ranking.
//!
//! Corpus text and upodesh candidates use different encodings of the same
//! spelling (upodesh 0.4.0 stores precomposed `য়`/`ড়`/`ঢ়`, the corpus mixes
//! both). Every token is folded to one canonical key before lookup; callers
//! always get their original strings back.

use std::borrow::Cow;

use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

const ZWNJ: char = '\u{200C}';
const ZWJ: char = '\u{200D}';
const KHANDA_TA_LEGACY: &str = "\u{09A4}\u{09CD}\u{200D}";
const KHANDA_TA: &str = "\u{09CE}";

/// Fold a word to its lookup key.
///
/// NFC (which decomposes the nukta letters, they are composition exclusions),
/// legacy khanda-ta `ত্‍` → `ৎ`, then ZWJ/ZWNJ removal. Borrows when the input
/// is already a key.
pub fn fold(word: &str) -> Cow<'_, str> {
    let needs_nfc = is_nfc_quick(word.chars()) != IsNormalized::Yes;
    let has_joiner = word.contains([ZWJ, ZWNJ]);
    if !needs_nfc && !has_joiner {
        return Cow::Borrowed(word);
    }
    let mut out: String = if needs_nfc {
        word.nfc().collect()
    } else {
        word.to_owned()
    };
    if has_joiner {
        if out.contains(KHANDA_TA_LEGACY) {
            out = out.replace(KHANDA_TA_LEGACY, KHANDA_TA);
        }
        out.retain(|c| c != ZWJ && c != ZWNJ);
    }
    Cow::Owned(out)
}

/// Inverse of the nukta part of [`fold`]: recompose `য়`/`ড়`/`ঢ়` into the
/// precomposed code points upodesh 0.4.0 indexes its dictionary with.
pub fn to_upodesh_form(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut chars = key.chars().peekable();
    while let Some(c) = chars.next() {
        if chars.peek() == Some(&'\u{09BC}') {
            let composed = match c {
                '\u{09AF}' => Some('\u{09DF}'),
                '\u{09A1}' => Some('\u{09DC}'),
                '\u{09A2}' => Some('\u{09DD}'),
                _ => None,
            };
            if let Some(composed) = composed {
                chars.next();
                out.push(composed);
                continue;
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precomposed_and_decomposed_fold_equal() {
        // বাড়িতে, precomposed ড় vs ড + nukta.
        assert_eq!(fold("বা\u{09DC}িতে"), fold("বা\u{09A1}\u{09BC}িতে"));
        assert_eq!(fold("ন\u{09DF}া"), fold("ন\u{09AF}\u{09BC}া"));
    }

    #[test]
    fn already_folded_borrows() {
        assert!(matches!(fold("আমি"), Cow::Borrowed(_)));
    }

    #[test]
    fn joiners_and_khanda_ta() {
        assert_eq!(fold("হঠাত\u{09CD}\u{200D}"), "হঠাৎ");
        assert_eq!(fold("র\u{200C}্য"), "র্য");
    }

    #[test]
    fn upodesh_form_roundtrip() {
        let word = "বা\u{09DC}িতে";
        assert_eq!(to_upodesh_form(&fold(word)), word);
    }
}
