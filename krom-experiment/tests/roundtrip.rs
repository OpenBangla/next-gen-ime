//! Build tiny models from fixtures and rank with them.
#![cfg(feature = "build")]

use std::fs;
use std::path::{Path, PathBuf};

use krom_experiment::build::{self, SbOptions, StemOptions};
use krom_experiment::format::AnalyzerKind;
use krom_experiment::{NgramModel, Ranker};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("krom-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Tokenized training text in `prepare` output form.
fn train_file(dir: &Path) -> PathBuf {
    let raw = "আমি বাড়ির সামনে দাঁড়িয়ে আছি। তিনি বাড়ির সামনে বসে আছেন। \
               আমরা বাড়ির সামনে খেলি। আমি ভাত খাই। আমি ভাত খাই। সে স্কুলে যায়। \
               ছেলেটি গাছের নিচে বসে। ছেলেটি গাছের সাথে কথা বলে। আমি গাছের নিচে ঘুমাই।";
    let text: String = krom_experiment::tokenize::sentences(raw)
        .into_iter()
        .map(|s| s.join(" ") + "\n")
        .collect();
    let path = dir.join("train.txt");
    fs::write(&path, text).unwrap();
    path
}

fn words(ranked: &[krom_experiment::Ranked<'_>]) -> Vec<String> {
    ranked.iter().map(|r| r.word.to_owned()).collect()
}

#[test]
fn stupid_backoff_ranks_by_context() {
    let dir = temp_dir("sb");
    let train = train_file(&dir);
    let model = dir.join("model");
    let opts = SbOptions {
        min_count: 1,
        prune: [1, 1],
        limit_vocab: None,
        scale: 2048,
        epsilon: 0.0,
    };
    build::build_stupid_backoff(&train, &model, &opts).unwrap();
    let ranker = Ranker::open(&model).unwrap();

    let candidates = ["সাথে", "সামনে", "স্কুলে"];
    assert_eq!(
        words(&ranker.rank(&["আমি", "বাড়ির"], &candidates))[0],
        "সামনে"
    );
    assert_eq!(words(&ranker.rank(&["গাছের"], &["নিচে", "সাথে"]))[0], "নিচে");
    // Candidates in either nukta encoding resolve to the same key.
    let decomposed = "বাড\u{09BC}ির";
    assert_eq!(
        ranker.score(&[decomposed], "সামনে"),
        ranker.score(&["বা\u{09DC}ির"], "সামনে")
    );
    // Unknown words sort last and keep their input order.
    let ranked = ranker.rank(&["আমি"], &["অজানা১", "অজানা২", "ভাত"]);
    assert_eq!(words(&ranked), ["ভাত", "অজানা১", "অজানা২"]);
}

#[test]
fn unigram_baseline_ignores_context() {
    let dir = temp_dir("uni");
    let train = train_file(&dir);
    let model = dir.join("model");
    let opts = SbOptions {
        min_count: 1,
        prune: [1, 1],
        limit_vocab: None,
        scale: 2048,
        epsilon: 0.0,
    };
    build::build_stupid_backoff(&train, &model, &opts).unwrap();
    let uni = Ranker::new(NgramModel::open(&model).unwrap().with_max_order(1));
    // "আমি" is the most frequent word regardless of context.
    assert_eq!(words(&uni.rank(&["গাছের"], &["নিচে", "আমি"]))[0], "আমি");
}

#[test]
fn stem_tables_generalize_history() {
    let dir = temp_dir("stem");
    let train = train_file(&dir);
    let model = dir.join("model");
    let opts = SbOptions {
        min_count: 1,
        prune: [1, 1],
        limit_vocab: None,
        scale: 2048,
        epsilon: 0.0,
    };
    build::build_stupid_backoff(&train, &model, &opts).unwrap();
    let stem_opts = StemOptions {
        analyzer: AnalyzerKind::Rule,
        prune: 1,
        scale: 2048,
    };
    build::build_stems(&train, &model, &stem_opts).unwrap();

    let surface = Ranker::new(NgramModel::open(&model).unwrap());
    let stemmed = Ranker::open(&model).unwrap();
    // "গাছটির" (গাছ + GEN) never occurs; "গাছের" (also গাছ + GEN) does.
    let history = ["গাছটির"];
    let cands = ["সামনে", "নিচে"];
    let s = stemmed.rank(&history, &cands);
    assert_eq!(s[0].word, "নিচে");
    assert!(stemmed.score(&history, "নিচে") > surface.score(&history, "নিচে"));
}

#[test]
fn arpa_backoff_math() {
    let dir = temp_dir("arpa");
    // log10 probabilities and backoffs, hand-made.
    let arpa = "\\data\\\nngram 1=5\nngram 2=2\nngram 3=1\n\n\\1-grams:\n\
        -1.0\t<unk>\t0\n-99\t<s>\t-0.3\n-1.0\t</s>\t0\n-0.5\tক\t-0.2\n-0.7\tখ\t-0.1\n\n\
        \\2-grams:\n-0.2\t<s> ক\t-0.4\n-0.3\tক খ\n\n\\3-grams:\n-0.05\t<s> ক খ\n\n\\end\\\n";
    let path = dir.join("lm.arpa");
    fs::write(&path, arpa).unwrap();
    let model = dir.join("model");
    build::arpa::build_from_arpa(&path, &model, None, 2048).unwrap();
    let ranker = Ranker::open(&model).unwrap();
    let close = |a: f32, b: f32| (a - b).abs() < 2e-3;
    // Empty context is sentence start: bigram <s> ক.
    assert!(close(ranker.score(&[], "ক"), -0.2));
    // Trigram hit.
    let start = ["<s>".to_owned(), "ক".to_owned()];
    assert!(close(ranker.rank_history(&start, &["খ"])[0].score, -0.05));
    let hist = ["ক"]; // history is [ক] (no <s>, since a context word is given)
    assert!(close(ranker.score(&hist, "খ"), -0.3));
    // Bigram miss: bo(ক) + p(ক) = -0.2 + -0.5.
    assert!(close(ranker.score(&hist, "ক"), -0.7));
    // Unknown word gets the <unk> probability.
    assert!(close(ranker.score(&hist, "গ"), -1.0));
}
