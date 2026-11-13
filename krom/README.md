# krom

Context-based ranking of Bengali word suggestions.

[`upodesh`](https://crates.io/crates/upodesh) (`=0.4.0`) generates candidate
words for what the user is typing, in arbitrary order. `krom` orders those
candidates by how likely each is to follow the previous words. It uses a
Stupid Backoff trigram model stored in memory-mapped
[`fst`](https://crates.io/crates/fst) maps. It is pure Rust, MIT licensed, and
has no C++ dependencies.

```rust
let ranker = krom::Ranker::open("model".as_ref())?;
let candidates = upodesh::bangla::suggest("বাংলা");
for r in ranker.rank(&["আমি", "তোমাকে"], &candidates).iter().take(5) {
    println!("{} {:.3}", r.word, r.score);
}
// Free text works too; a sentence terminator resets the context.
let ranked = ranker.rank_text("সে বলল। আমি", &candidates);
```

- **Encodings.** Candidates and context can use either encoding of `য়`/`ড়`/`ঢ়`. Both are folded to the same key internally, and the original strings are returned.
- **Empty context.** An empty context means the start of a sentence.
- **Ordering.** Unknown words sort last, keeping their input order.

The design comes from [`krom-experiment`](../krom-experiment). It compared KenLM, modified Kneser-Ney, stem-factored history and SentencePiece. Plain Stupid Backoff in an FST came within 0.7 Hit@1 points of the best variant, at the smallest size and with the fewest dependencies. Stemming and SentencePiece did not help.

## Model

A model is a directory of memory-mapped files:

- **`vocab.fst`** maps a folded word to its id; ids are lexicographic ranks.
- **`ngram.fst`** maps 1–3-gram keys to a quantized log10 probability. Keys are word ids written as 3-byte big-endian.
- **`meta.txt`** holds the format version, the backoff penalty and the quantization scale.

A candidate's score is its trigram score if that trigram is in the model. Otherwise the model falls back to the bigram score, then the unigram score, subtracting log10 0.4 for each step down.

## Building

```sh
# IndicCorp V2 bn (CC-0); the first 2.5 GB holds about 125M tokens.
curl -L -r 0-2684354559 -o data/bn.head.txt \
  https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/bn.txt

cargo build --release --features build
W=$(ls -d ~/.cargo/registry/src/*/upodesh-0.4.0)/data/source-words.txt   # upodesh dictionary

# Tokenize, deduplicate, split by document hash into train / dev / test (98/1/1).
target/release/krom-build prepare --corpus data/bn.head.txt --out data/prep --max-train-tokens 100000000

# Defaults: --min-count 3 --prune 3,3 --scale 32. Only n-grams predicting dictionary words are kept.
target/release/krom-build build --train data/prep/train.txt --out model --limit-vocab $W
```

Building from 100M tokens takes about 7 s and peaks at about 6 GB of RAM. `--prune` trades size for accuracy (the size is `vocab.fst` + `ngram.fst`):

| `--prune` | size | Hit@1 |
|---|---|---|
| `2,2` | 90 MB | 85.4 |
| `3,3` (default) | 49 MB | 85.0 |
| `5,5` | 28 MB | 84.4 |

## Evaluation

```sh
target/release/krom-eval --test data/prep/test.txt --model model
```

The eval simulates typing every held-out word:

- **Candidates.** At each prefix of 2 or more characters, the candidates are `upodesh::bangla::suggest(prefix)`, ranked given the two previous words.
- **Which cases count.** A case is scored only when the target is among at least 2 candidates.
- **KSR@3.** The keystroke saving when the word is picked as soon as it appears in the top 3.

Results on 100k cases with the default model:

| ranking | Hit@1 | Hit@3 | MRR | KSR@3 | p50 µs | p99 µs | size MB |
|---|---|---|---|---|---|---|---|
| upodesh order | 21.45 | 53.91 | 0.419 | 7.96 | – | – | – |
| unigram only | 76.85 | 95.00 | 0.861 | 13.12 | 2.5 | 54 | – |
| **krom** | **85.05** | **96.98** | **0.911** | **13.57** | 3.7 | 66 | 48.9 |

Criterion timings: ranking 20 candidates takes about 10 µs, and 148 candidates about 64 µs.

## Caveats

- **1-character prefixes.** `upodesh::bangla::suggest` (0.4.0) computes `0usize - 1` for a 1-character prefix. That panics in debug builds and runs effectively forever in release. Don't call it with fewer than 2 characters.
- **Ranking ceiling.** upodesh only returns candidates up to a fixed length beyond the prefix, so the target word is among them for just 28% of prefixes. That caps keystroke savings for any ranker.
- **Domain.** The training corpus is mostly news, so expect a domain gap for chat-style text.
