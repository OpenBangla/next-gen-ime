//! Offline model building (feature `build`).
//!
//! Counting is sort-based: n-gram keys are packed into integers whose numeric
//! order equals the FST byte order, sorted in parallel and run-length
//! counted, so the FST can be streamed out in one pass.

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use fst::MapBuilder;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::format::{self, MAX_ID, Meta};
use crate::tokenize::{self, BOS, EOS, FOREIGN, NUM, UNK};
use crate::{Error, normalize};

/// Lines per parallel batch when streaming the raw corpus.
const BATCH_LINES: usize = 20_000;

/// Statistics from [`prepare`].
#[derive(Debug, Default)]
pub struct PrepareStats {
    pub documents: u64,
    pub duplicates: u64,
    pub train_tokens: u64,
    pub dev_tokens: u64,
    pub test_tokens: u64,
}

/// Tokenize a raw corpus (one document per line) into `train.txt`,
/// `dev.txt` and `test.txt` under `out`: one sentence per line, folded
/// tokens separated by spaces.
///
/// Documents are deduplicated and assigned to a split by content hash
/// (1% test, 1% dev). Stops once `max_train_tokens` training tokens are
/// written. A truncated final line (partial download) is dropped.
pub fn prepare(corpus: &Path, out: &Path, max_train_tokens: u64) -> Result<PrepareStats, Error> {
    fs::create_dir_all(out)?;
    let mut writers = [
        BufWriter::new(File::create(out.join("train.txt"))?),
        BufWriter::new(File::create(out.join("dev.txt"))?),
        BufWriter::new(File::create(out.join("test.txt"))?),
    ];
    let mut stats = PrepareStats::default();
    let mut seen = FxHashSet::default();
    let mut reader = BufReader::with_capacity(1 << 22, File::open(corpus)?);
    let mut batch: Vec<String> = Vec::with_capacity(BATCH_LINES);
    let mut line = Vec::new();
    let mut done = false;
    while !done {
        batch.clear();
        while batch.len() < BATCH_LINES {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 || line.last() != Some(&b'\n') {
                done = true; // end of file, or a truncated final line
                break;
            }
            let text = String::from_utf8_lossy(&line);
            let text = text.trim();
            if !text.is_empty() {
                batch.push(text.to_owned());
            }
        }
        let docs: Vec<(u64, String, u64)> = batch
            .par_iter()
            .map(|doc| {
                let mut hasher = DefaultHasher::new();
                doc.hash(&mut hasher);
                let mut text = String::new();
                let mut tokens = 0u64;
                for sentence in tokenize::sentences(doc) {
                    // Sentences of only placeholders carry no signal.
                    if sentence.iter().all(|t| t == FOREIGN || t == NUM) {
                        continue;
                    }
                    tokens += sentence.len() as u64;
                    text.push_str(&sentence.join(" "));
                    text.push('\n');
                }
                (hasher.finish(), text, tokens)
            })
            .collect();
        for (hash, text, tokens) in docs {
            stats.documents += 1;
            if !seen.insert(hash) {
                stats.duplicates += 1;
                continue;
            }
            let split = match hash % 100 {
                0 => 2,
                1 => 1,
                _ => 0,
            };
            writers[split].write_all(text.as_bytes())?;
            match split {
                0 => stats.train_tokens += tokens,
                1 => stats.dev_tokens += tokens,
                _ => stats.test_tokens += tokens,
            }
            if stats.train_tokens >= max_train_tokens {
                done = true;
                break;
            }
        }
    }
    for w in &mut writers {
        w.flush()?;
    }
    Ok(stats)
}

/// Options for [`build`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Words seen fewer times map to `<unk>`.
    pub min_count: u32,
    /// Minimum counts to keep a bigram / trigram.
    pub prune: [u32; 2],
    /// When set, only n-grams predicting one of these (folded) words are
    /// kept (unigrams are always kept), e.g. the upodesh dictionary.
    pub limit_vocab: Option<FxHashSet<String>>,
    /// Quantization levels per log10 unit.
    pub scale: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            min_count: 3,
            prune: [3, 3],
            limit_vocab: None,
            scale: format::DEFAULT_SCALE,
        }
    }
}

/// Statistics from [`build`].
#[derive(Debug, Default)]
pub struct BuildStats {
    pub tokens: u64,
    pub vocab: usize,
    /// Kept 1-, 2- and 3-grams.
    pub kept: [u64; 3],
    /// Size of `vocab.fst` + `ngram.fst`.
    pub bytes: u64,
}

/// Read a word list (whitespace separated), folded.
pub fn read_word_list(path: &Path) -> Result<FxHashSet<String>, Error> {
    let text = fs::read_to_string(path)?;
    Ok(text
        .split_whitespace()
        .map(|w| normalize::fold(w).into_owned())
        .collect())
}

fn padded(line: &str) -> impl Iterator<Item = &str> {
    std::iter::once(BOS)
        .chain(line.split(' ').filter(|t| !t.is_empty()))
        .chain([EOS])
}

const FIELD: u32 = 21;
const FIELD_MASK: u64 = (1 << FIELD) - 1;

/// Pack up to three ids (oldest first, `+1` so 0 means absent) left-aligned:
/// numeric order equals the byte order of the 3/6/9-byte FST keys.
fn pack_key(ids: &[u32]) -> u64 {
    let mut key = 0u64;
    for (i, &id) in ids.iter().enumerate() {
        key |= (id as u64 + 1) << (FIELD * (2 - i as u32));
    }
    key
}

/// Write the FST key bytes of a packed key into `buf`; returns its order.
fn unpack_key(key: u64, buf: &mut Vec<u8>) -> usize {
    buf.clear();
    let mut order = 0;
    for i in 0..3 {
        let field = key >> (FIELD * (2 - i)) & FIELD_MASK;
        if field == 0 {
            break;
        }
        format::push_id(buf, field as u32 - 1);
        order += 1;
    }
    order
}

/// Number of ids in a packed key.
fn key_order(key: u64) -> usize {
    (0..3)
        .take_while(|i| key >> (FIELD * (2 - i)) & FIELD_MASK != 0)
        .count()
}

/// Last (predicted) id of a packed key.
fn key_last(key: u64) -> usize {
    (key >> (FIELD * (3 - key_order(key) as u32)) & FIELD_MASK) as usize - 1
}

/// Sort and run-length count keys.
fn count_sorted(mut keys: Vec<u64>) -> Vec<(u64, u32)> {
    keys.par_sort_unstable();
    let mut out: Vec<(u64, u32)> = Vec::with_capacity(keys.len() / 4);
    for key in keys {
        match out.last_mut() {
            Some((last, count)) if *last == key => *count += 1,
            _ => out.push((key, 1)),
        }
    }
    out
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Build a Stupid Backoff trigram model from a prepared `train.txt` into the
/// model directory `out`.
pub fn build(train: &Path, out: &Path, opts: &Options) -> Result<BuildStats, Error> {
    fs::create_dir_all(out)?;
    let text = fs::read_to_string(train)?;
    let lines: Vec<&str> = text.lines().collect();

    // Vocabulary: lexicographic ranks are the ids.
    let unigrams = lines
        .par_iter()
        .fold(FxHashMap::<&str, u32>::default, |mut map, line| {
            for token in padded(line) {
                *map.entry(token).or_default() += 1;
            }
            map
        })
        .reduce(FxHashMap::default, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
            a
        });
    let mut words: Vec<String> = unigrams
        .iter()
        .filter(|&(w, &c)| c >= opts.min_count || [BOS, EOS, FOREIGN, NUM].contains(w))
        .map(|(w, _)| (*w).to_owned())
        .collect();
    for special in [BOS, EOS, UNK] {
        if !words.iter().any(|w| w == special) {
            words.push(special.to_owned());
        }
    }
    words.sort_unstable();
    if words.len() > MAX_ID as usize {
        return Err(Error::Format(format!(
            "vocabulary of {} exceeds {MAX_ID}",
            words.len()
        )));
    }
    drop(unigrams);
    let ids: FxHashMap<&str, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as u32))
        .collect();
    let unk = ids[UNK];
    let mut vocab = MapBuilder::new(BufWriter::new(File::create(out.join("vocab.fst"))?))?;
    for (id, word) in words.iter().enumerate() {
        vocab.insert(word, id as u64)?;
    }
    vocab.finish()?;

    // All 1–3-gram keys.
    let keys: Vec<u64> = lines
        .par_iter()
        .flat_map_iter(|line| {
            let seq: Vec<u32> = padded(line).map(|t| *ids.get(t).unwrap_or(&unk)).collect();
            let mut keys = Vec::with_capacity(seq.len() * 3);
            for i in 0..seq.len() {
                keys.push(pack_key(&seq[i..=i]));
                if i >= 1 {
                    keys.push(pack_key(&seq[i - 1..=i]));
                }
                if i >= 2 {
                    keys.push(pack_key(&seq[i - 2..=i]));
                }
            }
            keys
        })
        .collect();
    drop(lines);
    drop(text);
    let counts = count_sorted(keys);

    // Predicted tokens; `<s>` is only ever a context.
    let bos = ids[BOS] as usize;
    let total: u64 = counts
        .iter()
        .filter(|&&(k, _)| key_order(k) == 1 && key_last(k) != bos)
        .map(|&(_, c)| c as u64)
        .sum();
    let allowed: Option<Vec<bool>> = opts.limit_vocab.as_ref().map(|set| {
        words
            .iter()
            .map(|w| w.starts_with('<') || set.contains(w))
            .collect()
    });

    // In key order every context's own entry precedes its continuations, so
    // the denominators are the most recent unigram / bigram counts.
    let mut builder = MapBuilder::new(BufWriter::new(File::create(out.join("ngram.fst"))?))?;
    let mut stats = BuildStats {
        tokens: total,
        vocab: words.len(),
        ..Default::default()
    };
    let mut key = Vec::with_capacity(3 * format::ID_BYTES);
    let (mut c1, mut c2) = (0u32, 0u32);
    for &(packed, count) in &counts {
        let order = unpack_key(packed, &mut key);
        let (denominator, keep) = match order {
            1 => {
                c1 = count;
                (total as f64, true)
            }
            2 => {
                c2 = count;
                (c1 as f64, count >= opts.prune[0])
            }
            _ => (c2 as f64, count >= opts.prune[1]),
        };
        let keep = keep && (order == 1 || allowed.as_ref().is_none_or(|a| a[key_last(packed)]));
        if keep {
            let logp = (count as f64 / denominator).log10() as f32;
            builder.insert(&key, format::pack(logp, opts.scale))?;
            stats.kept[order - 1] += 1;
        }
    }
    builder.finish()?;

    let meta = Meta {
        tokens: total,
        scale: opts.scale,
        unk_logp: (0.5 / total as f64).log10() as f32,
        ..Meta::default()
    };
    meta.write(out)?;
    stats.bytes = file_size(&out.join("ngram.fst")) + file_size(&out.join("vocab.fst"));
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_keys_sort_like_bytes() {
        let keys = [
            pack_key(&[5]),
            pack_key(&[5, 0]),
            pack_key(&[5, 0, 9]),
            pack_key(&[5, 1]),
            pack_key(&[6]),
        ];
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
        let bytes: Vec<Vec<u8>> = keys
            .iter()
            .map(|&k| {
                let mut b = Vec::new();
                unpack_key(k, &mut b);
                b
            })
            .collect();
        assert!(bytes.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn key_fields() {
        let tri = pack_key(&[7, 8, 9]);
        assert_eq!((key_order(tri), key_last(tri)), (3, 9));
        assert_eq!(
            (key_order(pack_key(&[0])), key_last(pack_key(&[0]))),
            (1, 0)
        );
    }
}
