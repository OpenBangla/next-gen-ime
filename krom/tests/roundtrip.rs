//! Build a tiny model from a fixture corpus and rank with it.
#![cfg(feature = "build")]

use std::fs;
use std::path::PathBuf;

use krom::Ranker;
use krom::build::{self, Options};

/// Build a model from the fixture text with no pruning.
fn fixture_model(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("krom-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let raw = "আমি বাড়ির সামনে দাঁড়িয়ে আছি। তিনি বাড়ির সামনে বসে আছেন। \
               আমরা বাড়ির সামনে খেলি। আমি ভাত খাই। আমি ভাত খাই। সে স্কুলে যায়। \
               ছেলেটি গাছের নিচে বসে। ছেলেটি গাছের সাথে কথা বলে। আমি গাছের নিচে ঘুমাই।";
    let text: String = krom::tokenize::sentences(raw)
        .into_iter()
        .map(|s| s.join(" ") + "\n")
        .collect();
    let train = dir.join("train.txt");
    fs::write(&train, text).unwrap();
    let model = dir.join("model");
    let opts = Options {
        min_count: 1,
        prune: [1, 1],
        ..Options::default()
    };
    build::build(&train, &model, &opts).unwrap();
    model
}

fn words(ranked: &[krom::Ranked<'_>]) -> Vec<String> {
    ranked.iter().map(|r| r.word.to_owned()).collect()
}

#[test]
fn ranks_by_context() {
    let ranker = Ranker::open(&fixture_model("context")).unwrap();
    let candidates = ["সাথে", "সামনে", "স্কুলে"];
    assert_eq!(
        words(&ranker.rank(&["আমি", "বাড়ির"], &candidates))[0],
        "সামনে"
    );
    assert_eq!(words(&ranker.rank(&["গাছের"], &["সামনে", "নিচে"]))[0], "নিচে");
    // A sentence terminator resets the context to the sentence start.
    assert_eq!(
        words(&ranker.rank_text("গাছের নিচে বসে। ", &["আমি", "নিচে"]))[0],
        "আমি"
    );
}

#[test]
fn either_nukta_encoding_matches() {
    let ranker = Ranker::open(&fixture_model("nukta")).unwrap();
    let precomposed = ranker.score(&["বা\u{09DC}ির"], "সামনে");
    let decomposed = ranker.score(&["বা\u{09A1}\u{09BC}ির"], "সামনে");
    assert_eq!(precomposed, decomposed);
    assert!(precomposed > ranker.score(&["অজানা"], "সামনে"));
}

#[test]
fn unknown_words_last_in_input_order() {
    let ranker = Ranker::open(&fixture_model("unknown")).unwrap();
    let ranked = ranker.rank(&["আমি"], &["অজানা১", "অজানা২", "ভাত"]);
    assert_eq!(words(&ranked), ["ভাত", "অজানা১", "অজানা২"]);
}

#[test]
fn unigram_baseline_ignores_context() {
    let ranker = Ranker::open(&fixture_model("unigram"))
        .unwrap()
        .with_max_order(1);
    // "আমি" is the most frequent word regardless of context.
    assert_eq!(words(&ranker.rank(&["গাছের"], &["নিচে", "আমি"]))[0], "আমি");
}

#[test]
fn pruning_drops_rare_ngrams() {
    let dir = fixture_model("prune");
    let train = dir.parent().unwrap().join("train.txt");
    let pruned = dir.parent().unwrap().join("pruned");
    let opts = Options {
        min_count: 1,
        prune: [2, 2],
        ..Options::default()
    };
    let stats = build::build(&train, &pruned, &opts).unwrap();
    // "আমি ভাত খাই" occurs twice, so its trigram survives.
    assert!(stats.kept[2] >= 1);
    let ranker = Ranker::open(&pruned).unwrap();
    assert_eq!(
        words(&ranker.rank(&["আমি", "ভাত"], &["খেলি", "খাই"]))[0],
        "খাই"
    );
}
