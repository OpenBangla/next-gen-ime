# krom-experiment

The research crate behind [`krom`](../krom): every ranking variant evaluated
for context-based ranking of Bengali word suggestions. The production crate
implements the recommendation at the end of this document.

[`upodesh`](https://crates.io/crates/upodesh) (`=0.4.0`) generates the
candidate words for what the user is typing, but returns them in arbitrary
(`HashSet`) order. `krom` orders those candidates by how likely each one is to
follow the previous words. It uses a trigram backoff model stored in
memory-mapped [`fst`](https://crates.io/crates/fst) maps. `krom` does not
generate words.

```rust
let ranker = krom_experiment::Ranker::open("model".as_ref())?;
let candidates = upodesh::bangla::suggest("বাংলা");
for r in ranker.rank(&["আমি", "তোমাকে"], &candidates).iter().take(5) {
    println!("{} {:.3}", r.word, r.score);
}
// Free text works too; a sentence terminator resets the context.
let ranked = ranker.rank_text("সে বলল। আমি", &candidates);
```

- **Candidate encodings.** Candidates can use either encoding of `য়`/`ড়`/`ঢ়`; upodesh stores precomposed letters, while the corpus mixes both. They are folded to one key internally, and the original strings are returned.
- **Speed.** Ranking 20 candidates takes about 18 µs (criterion, M-series Mac). Each candidate costs at most three `fst` lookups.
- **Empty context.** An empty context means the start of a sentence.

## Layout

| path | purpose |
|---|---|
| `src/normalize.rs` | key folding (NFC, ZWJ/ZWNJ, khanda-ta), plus the upodesh encoding |
| `src/tokenize.rs` | sentence splitting, punctuation trimming, the `<s> </s> <unk> <num> <x>` tokens |
| `src/ngram.rs` | FST backoff model (Stupid Backoff, or ARPA backoff weights) |
| `src/stem.rs`, `src/stem_model.rs` | rule stemmer and stem-factored history tables (experimental, see results) |
| `src/kenlm.rs` | KenLM runtime scorer, feature `kenlm` |
| `src/build/` | offline pipeline, feature `build` |
| `src/bin/krom-build.rs`, `src/bin/krom-eval.rs` | CLIs |

The model is a directory of memory-mapped files:

- **`vocab.fst`** maps a folded word to its id. Ids are lexicographic ranks.
- **`ngram.fst`** maps 3/6/9-byte big-endian id keys to a packed `u64`:
  - a 16-bit quantized log10 probability;
  - a 16-bit backoff;
  - a flag bit.
- **`meta.txt`** holds the format version, the estimation method and the quantization scale.

## Building a model

```sh
# 1. Corpus: IndicCorp V2 bn (CC-0). Taking the first 2.5 GB is enough for 100M tokens.
curl -L -r 0-2684354559 -o data/corpus/bn.head.txt \
  https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/bn.txt

cargo build --release --features build
B=target/release/krom-build
W=$(ls -d ~/.cargo/registry/src/*/upodesh-0.4.0)/data/source-words.txt   # upodesh dictionary

# 2. Tokenize, deduplicate, and split by document hash into train / dev / test (98/1/1).
$B prepare --corpus data/corpus/bn.head.txt --out data/prep --max-train-tokens 100000000

# 3. Stupid Backoff model. Only n-grams that predict dictionary words are kept.
$B sb --train data/prep/train.txt --out model --limit-vocab $W --scale 32 [--prune 3,3]
```

Building with 100M tokens takes about 8 s and about 6 GB of RAM. Counting is sort-based: packed integer keys whose numeric order equals the FST byte order.

### KenLM (modified Kneser-Ney)

```sh
lmplz -o 3 --prune 0 1 1 --limit_vocab_file model/vocab.txt --arpa lm.arpa < data/prep/train.txt
$B arpa --arpa lm.arpa --out model-kn --limit-vocab $W --scale 32      # pure-Rust runtime
build_binary -q 8 -b 8 -a 22 trie lm.arpa lm.trie.bin                  # KenLM runtime (feature kenlm)
```

## Evaluation

`krom-eval` simulates a user typing every word of the held-out test set, using upodesh 0.4.0 candidates:

- **Prefixes.** At each prefix length p ≥ 2, the candidates are `upodesh::bangla::suggest(prefix)`.
- **Scoring.** Each variant ranks the candidates given the two previous words.
- **Which cases count.** A case is scored only when the target is among at least 2 candidates. That held for 28% of the prefixes, because of how upodesh expands prefixes.
- **KSR@3.** The keystroke saving when the user picks the word as soon as it appears in the top 3.

```sh
cargo run --release --features build,kenlm,sp-eval --bin krom-eval -- --test data/prep/test.txt \
  --variant upodesh --variant uni:U=model --variant fst:F-SB=model --variant fst+stem:F-SB+stem=model \
  --variant kenlm:K=lm.trie.bin
```

### Results

Setup:
- **Data.** 100M training tokens from IndicCorp V2 bn.
- **Test set.** 100k cases over 64.7k test words, about 13.7 candidates per case.
- **Vocabulary and pruning.** Words seen at least 3 times; n-grams seen at least 2 times (unless the row says otherwise).

Size covers the files a runtime needs.

| variant | Hit@1 | Hit@3 | MRR | KSR@3 | p50 µs | p99 µs | size MB |
|---|---|---|---|---|---|---|---|
| upodesh order (arbitrary) | 21.30 | 51.73 | 0.414 | 7.48 | – | – | – |
| U: unigram only | 76.87 | 94.99 | 0.861 | 13.11 | 2.5 | 56 | 90 |
| **F-SB**: Stupid Backoff in fst | 85.43 | 97.15 | 0.914 | 13.61 | 3.8 | 68 | 90 |
| F-SB, n-grams ≥ 3 | 85.03 | 96.97 | 0.911 | 13.57 | 3.8 | 73 | 49 |
| F-SB, n-grams ≥ 5 | 84.40 | 96.80 | 0.907 | 13.52 | 3.7 | 70 | 28 |
| F-SB + rule-stem history | 85.42 | 97.15 | 0.914 | 13.61 | 7.6 | 126 | 225 |
| F-SB + SentencePiece-stem history | 85.24 | 97.11 | 0.913 | 13.60 | 10.2 | 131 | 189 |
| F-KN: KenLM MKN → ARPA → fst | 85.63 | 97.12 | 0.915 | 13.59 | 4.1 | 70 | 155 |
| K: KenLM runtime, trie, q8 | 85.74 | 97.22 | 0.916 | 13.62 | 2.6 | 44 | 83 |
| K: KenLM runtime, probing | 85.74 | 97.22 | 0.916 | 13.62 | 2.0 | 39 | 330 |
| K + rule-stem history | 85.74 | 97.22 | 0.916 | 13.62 | 6.4 | 95 | 221 |

On the dev set, a 1M-token training corpus scored 75.7 (U) and 77.7 (F-SB). A 10M-token corpus scored 76.7 and 82.4. In both cases the stem tables changed Hit@1 by less than 0.05.

## Findings

- **Context is what matters.** Trigram context adds about 8.5 Hit@1 points over unigram frequency. The unigram model is in turn far better than upodesh's arbitrary order. The choice of estimator is a second-order effect: modified Kneser-Ney beats Stupid Backoff by only 0.2–0.3 points.

- **Stemming does not help this task.** We tried a factored-LM-style backoff, P(w | stem₋₂, stem₋₁, suffix-class₋₁), where the predicted word stays a surface form. It is measurably neutral: on the eval cases it changes the gold word's rank in 0.2% of cases, half for better and half for worse.
  - **Why.** All candidates share the typed prefix and the same history, so when the history is sparse they all back off together. Their relative order is decided by surface bigram and unigram evidence that 100M tokens already provide for dictionary words.
  - **More data doesn't change it.** The result also holds at 1M and 10M tokens.
  - **Cost.** The stem tables cost more space than the n-grams themselves.

- **SentencePiece is worse than useless as a stemmer.** Using the first piece of the repo's BPE model (`bangla_gru_sp`, 16k vocabulary) yields only 2,182 distinct "stems" for a vocabulary of 407k words. The first BPE piece is a short fragment such as `প্র`, so the history is over-merged and accuracy drops slightly.
  - **As the n-gram unit.** SentencePiece pieces were also rejected by the earlier research: a piece trigram covers only about one word of history, and scores would vary with how many pieces a candidate has.

- **KenLM.**
  - **As an estimator.** Modified Kneser-Ney via `lmplz` is the most accurate option and trains in 30 s (about 3 GB of RAM).
  - **As a runtime.** Its trie binary is the smallest file at full accuracy (83 MB) and the fastest.
  - **Costs.** It is C++ under LGPL-3.0 (via `kenlm-rs`), and `lmplz` needs Boost.
  - **Converted to fst.** Converting its ARPA output to krom's fst keeps MKN quality with a pure-Rust runtime. But the backoff weights stored per context make the fst large (155 MB).

## Recommendation

1. **Ship the pure-Rust FST runtime with Stupid Backoff and count pruning.** `--prune 3,3 --scale 32` gives 49 MB at 85.0 Hit@1, and `--prune 5,5` gives 28 MB at 84.4.
   - It is MIT, has no C++, uses memory-mapped files, builds in seconds, and uses the same `fst` crate as upodesh.
   - It is within 0.7 points of the best variant.
2. **Use KenLM only if the last ~0.5 point matters more than dependencies.** The trie at 83 MB is the best accuracy per byte.
3. **Don't ship stem tables or SentencePiece analysis for this ranker.**

## Caveats

- **upodesh 0.4.0 bugs.**
  - `bangla::suggest` with a 1-character prefix underflows (`0usize - 1`): it panics in debug builds and runs effectively forever in release. The eval starts at prefix length 2.
  - Its candidate order is `HashSet` order, so the "upodesh order" row varies between runs (±1 point).
- **Ranking ceiling.** upodesh limits how long its candidates can be relative to the prefix, so the target is among the candidates for only 28% of prefixes. That caps KSR for any ranker.
- **Domain.** The corpus is news-heavy, so expect a domain gap for chat-style input.

## Possible next steps

- **Upodesh-only vocabulary.** Restrict the unigram and history vocabulary to the upodesh dictionary; most of the 407k vocabulary entries are never candidates.
- **Smaller backoff values.** Quantize the ARPA backoffs more coarsely, or keep only contexts that have children, to bring F-KN down to F-SB size.
- **Personalization.** Add the user's own bigrams in a small mutable overlay.
