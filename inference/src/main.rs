use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use sentencepiece::SentencePieceProcessor;

const CONTEXT_LEN: usize = 8;
const NUM_SPECIAL_TOKENS: usize = 4; // <pad>=0, <unk>=1, <s>=2, </s>=3

struct Predictor {
    session: Session,
    spp: SentencePieceProcessor,
    /// id -> piece string (without the \u{2581} marker)
    vocab: Vec<String>,
}

impl Predictor {
    fn new(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_dir.join("model.onnx"))?;

        let spp = SentencePieceProcessor::open(model_dir.join("sp_bangla.model"))?;
        let vocab = load_vocab(&model_dir.join("sp_bangla.vocab"))?;

        Ok(Self {
            session,
            spp,
            vocab,
        })
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        let pieces = self.spp.encode(text).unwrap_or_default();
        pieces.iter().map(|p| p.id as i64).collect()
    }

    fn prepare_context(&self, tokens: &[i64]) -> Array2<i64> {
        let mut ctx = vec![0i64; CONTEXT_LEN];
        if tokens.len() >= CONTEXT_LEN {
            let start = tokens.len() - CONTEXT_LEN;
            ctx.copy_from_slice(&tokens[start..]);
        } else {
            let offset = CONTEXT_LEN - tokens.len();
            ctx[offset..].copy_from_slice(tokens);
        }
        Array2::from_shape_vec((1, CONTEXT_LEN), ctx).unwrap()
    }

    fn run_model(&mut self, input: Array2<i64>) -> Vec<f32> {
        let tensor = Tensor::from_array(input).unwrap();
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => tensor])
            .unwrap();
        let logits = outputs["logits"].try_extract_array::<f32>().unwrap();
        logits.as_slice().unwrap().to_vec()
    }

    fn predict(&mut self, text: &str, prefix: &str, top_k: usize) -> Vec<(String, f32)> {
        let tokens = self.encode(text);
        if tokens.is_empty() {
            return vec![];
        }

        let input = self.prepare_context(&tokens);
        let logits = self.run_model(input);

        if prefix.is_empty() {
            self.top_k_from_logits(&logits, top_k)
        } else {
            self.top_k_with_prefix(&logits, prefix, top_k)
        }
    }

    /// Standard top-k: softmax over all logits, pick the best k (skipping special tokens).
    fn top_k_from_logits(&self, logits: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let probs = softmax(logits);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .into_iter()
            .filter(|(idx, _)| *idx >= NUM_SPECIAL_TOKENS)
            .filter_map(|(idx, prob)| {
                let piece = self.vocab.get(idx)?;
                if piece.is_empty() {
                    return None;
                }
                Some((piece.clone(), prob))
            })
            .take(top_k)
            .collect()
    }

    /// Prefix-aware top-k: mask non-matching tokens before softmax.
    fn top_k_with_prefix(
        &self,
        logits: &[f32],
        prefix: &str,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let mut masked = vec![f32::NEG_INFINITY; logits.len()];
        for (idx, &logit) in logits.iter().enumerate() {
            if idx >= NUM_SPECIAL_TOKENS {
                if let Some(piece) = self.vocab.get(idx) {
                    if !piece.is_empty() && piece.starts_with(prefix) {
                        masked[idx] = logit;
                    }
                }
            }
        }

        let probs = softmax(&masked);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .into_iter()
            .filter(|(_, prob)| *prob > 0.0 && prob.is_finite())
            .filter_map(|(idx, prob)| {
                let piece = self.vocab.get(idx)?;
                Some((piece.clone(), prob))
            })
            .take(top_k)
            .collect()
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Parse the SentencePiece vocab file (tsv: piece\tscore) into a Vec indexed by id.
/// Strips the \u{2581} word-boundary marker from pieces.
fn load_vocab(path: &Path) -> Result<Vec<String>, io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut vocab = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let piece = line.split('\t').next().unwrap_or("");
        let clean = piece.replace('\u{2581}', "");
        vocab.push(clean);
    }
    Ok(vocab)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bangla_gru_sp");

    println!("Loading model from {:?}...", model_dir);
    let mut predictor = Predictor::new(&model_dir)?;
    println!("Model loaded. Vocabulary size: {}", predictor.vocab.len());

    // Demo predictions
    let examples: &[(&str, &str)] = &[
        ("\u{0986}\u{09AE}\u{09BF} \u{09AC}\u{09BE}\u{0982}\u{09B2}\u{09BE}\u{09AF}\u{09BC}", ""),
        ("\u{0986}\u{09AE}\u{09BF} \u{09AC}\u{09BE}\u{0982}\u{09B2}\u{09BE}\u{09AF}\u{09BC}", "\u{0997}"),
        ("\u{09AC}\u{09BE}\u{0982}\u{09B2}\u{09BE}\u{09A6}\u{09C7}\u{09B6}\u{09C7}\u{09B0} \u{09B0}\u{09BE}\u{099C}\u{09A7}\u{09BE}\u{09A8}\u{09C0}", "\u{09A2}\u{09BE}"),
        ("\u{0986}\u{099C}\u{0995}\u{09C7} \u{0986}\u{09AE}\u{09BF}", "\u{0995}\u{09BE}"),
        ("\u{09B8}\u{09C7} \u{09B8}\u{09CD}\u{0995}\u{09C1}\u{09B2}\u{09C7}", ""),
    ];

    println!("\n{}", "=".repeat(60));
    println!("  Bangla Next Word Prediction (ONNX + Rust)");
    println!("{}", "=".repeat(60));

    for &(text, prefix) in examples {
        println!("\nInput: {text}");
        if !prefix.is_empty() {
            println!("Prefix: '{prefix}'");
        }
        let preds = predictor.predict(text, prefix, 5);
        if preds.is_empty() {
            println!("  (no predictions)");
        } else {
            for (i, (word, prob)) in preds.iter().enumerate() {
                println!("  {}. {:12} {:.1}%", i + 1, word, prob * 100.0);
            }
        }
    }

    // Interactive mode
    println!("\n{}", "=".repeat(60));
    println!("  Interactive mode \u{2014} format: <text> | <prefix>");
    println!("  Type 'quit' to exit");
    println!("{}", "=".repeat(60));

    let stdin = io::stdin();
    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" || line == "q" {
            break;
        }

        let (text, prefix) = if let Some((t, p)) = line.split_once('|') {
            (t.trim(), p.trim())
        } else {
            (line, "")
        };

        let preds = predictor.predict(text, prefix, 10);
        if preds.is_empty() {
            println!(
                "  (no predictions{})",
                if prefix.is_empty() {
                    ""
                } else {
                    " matching prefix"
                }
            );
        } else {
            for (i, (word, prob)) in preds.iter().enumerate() {
                println!("  {:2}. {} ({:.1}%)", i + 1, word, prob * 100.0);
            }
        }
    }

    Ok(())
}
