//! Compare ranking variants on upodesh 0.4.0 candidates.
//!
//! For every word of the test set, the user is simulated typing it: at each
//! prefix length p ≥ 2 the candidates are `upodesh::bangla::suggest(prefix)`
//! and each variant ranks them given the two previous words. Cases where the
//! target is not among ≥ 2 candidates are not scored (coverage is reported).
//!
//! ```text
//! krom-eval --test data/test.txt [--max-cases 50000] [--stem-penalty -0.4]
//!     --variant upodesh
//!     --variant uni:U=model-sb
//!     --variant fst:F-SB=model-sb          # surface tables only
//!     --variant fst+stem:F-SB+stem=model-sb
//!     --variant kenlm:K=lm.binary          # feature `kenlm`
//!     --variant kenlm+stem:K+stem=lm.binary@model-kn
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[cfg(feature = "kenlm")]
use krom_experiment::StemModel;
use krom_experiment::normalize::{fold, to_upodesh_form};
use krom_experiment::tokenize::{BOS, EOS};
use krom_experiment::{NgramModel, Ranker, Scorer};

trait Variant {
    /// Candidate indices, best first.
    fn order(&self, history: &[String], candidates: &[String]) -> Vec<usize>;
}

struct UpodeshOrder;

impl Variant for UpodeshOrder {
    fn order(&self, _: &[String], candidates: &[String]) -> Vec<usize> {
        (0..candidates.len()).collect()
    }
}

impl<S: Scorer> Variant for Ranker<S> {
    fn order(&self, history: &[String], candidates: &[String]) -> Vec<usize> {
        self.rank_history(history, candidates)
            .iter()
            .map(|r| r.index)
            .collect()
    }
}

struct Case {
    token: usize,
    prefix_len: usize,
    history: Vec<String>,
    candidates: std::rc::Rc<Vec<String>>,
    gold: usize,
}

struct Token {
    len: usize,
}

struct CaseSet {
    cases: Vec<Case>,
    tokens: Vec<Token>,
    /// Prefixes whose candidates contained / missed the target.
    covered: u64,
    missed: u64,
}

fn dir_size(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".fst")
        })
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

fn open_variant(spec: &str, penalty: f32) -> Result<(String, Box<dyn Variant>, u64), String> {
    if spec == "upodesh" {
        return Ok(("upodesh order".into(), Box::new(UpodeshOrder), 0));
    }
    let (kind, rest) = spec
        .split_once(':')
        .ok_or_else(|| format!("bad variant {spec}"))?;
    let (name, target) = rest
        .split_once('=')
        .ok_or_else(|| format!("bad variant {spec}"))?;
    let err = |e: krom_experiment::Error| format!("{spec}: {e}");
    let (variant, size): (Box<dyn Variant>, u64) = match kind {
        "uni" => {
            let model = NgramModel::open(target.as_ref())
                .map_err(err)?
                .with_max_order(1);
            let size = ["vocab.fst", "ngram.fst"]
                .iter()
                .map(|f| {
                    fs::metadata(Path::new(target).join(f))
                        .map(|m| m.len())
                        .unwrap_or(0)
                })
                .sum();
            (Box::new(Ranker::new(model)), size)
        }
        "fst" => {
            let model = NgramModel::open(target.as_ref()).map_err(err)?;
            let size = dir_size(target.as_ref())
                - [
                    Path::new(target).join("stems.fst"),
                    Path::new(target).join("stem_ngram.fst"),
                ]
                .iter()
                .map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                .sum::<u64>();
            (Box::new(Ranker::new(model)), size)
        }
        "fst+stem" => {
            let ranker = Ranker::open(target.as_ref())
                .map_err(err)?
                .with_stem_penalty(penalty);
            (Box::new(ranker), dir_size(target.as_ref()))
        }
        #[cfg(feature = "kenlm")]
        "kenlm" | "kenlm+stem" => {
            let (file, stems) = match target.split_once('@') {
                Some((file, stems)) => (file, Some(stems)),
                None => (target, None),
            };
            let model = krom_experiment::kenlm::KenlmModel::open(file.as_ref()).map_err(err)?;
            let mut size = dir_size(file.as_ref());
            let stems = match (kind, stems) {
                ("kenlm+stem", Some(dir)) => {
                    size += fs::metadata(Path::new(dir).join("stem_ngram.fst"))
                        .map(|m| m.len())
                        .unwrap_or(0)
                        + fs::metadata(Path::new(dir).join("stems.fst"))
                            .map(|m| m.len())
                            .unwrap_or(0)
                        + fs::metadata(Path::new(dir).join("vocab.fst"))
                            .map(|m| m.len())
                            .unwrap_or(0);
                    StemModel::open(dir.as_ref()).map_err(err)?
                }
                ("kenlm+stem", None) => return Err(format!("{spec}: needs FILE@STEM_DIR")),
                _ => None,
            };
            (
                Box::new(
                    Ranker::new(model)
                        .with_stems(stems)
                        .with_stem_penalty(penalty),
                ),
                size,
            )
        }
        _ => return Err(format!("unknown variant kind {kind}")),
    };
    Ok((name.to_owned(), variant, size))
}

fn build_cases(test: &Path, max_cases: usize) -> Result<CaseSet, String> {
    let text = fs::read_to_string(test).map_err(|e| format!("{}: {e}", test.display()))?;
    let mut cache: HashMap<String, std::rc::Rc<Vec<String>>> = HashMap::new();
    let mut cases = Vec::new();
    let mut tokens = Vec::new();
    let (mut covered, mut missed) = (0u64, 0u64);
    'outer: for line in text.lines() {
        let toks: Vec<&str> = std::iter::once(BOS)
            .chain(line.split(' '))
            .chain([EOS])
            .collect();
        for i in 1..toks.len() - 1 {
            let target = toks[i];
            if target.starts_with('<') {
                continue;
            }
            let surface: Vec<char> = to_upodesh_form(target).chars().collect();
            let history: Vec<String> = toks[i.saturating_sub(2)..i]
                .iter()
                .map(|s| s.to_string())
                .collect();
            let token = tokens.len();
            let mut any = false;
            for p in 2..surface.len() {
                let prefix: String = surface[..p].iter().collect();
                let candidates = cache
                    .entry(prefix.clone())
                    .or_insert_with(|| std::rc::Rc::new(upodesh::bangla::suggest(&prefix)))
                    .clone();
                let gold = candidates.iter().position(|c| fold(c) == target);
                match gold {
                    Some(gold) if candidates.len() >= 2 => {
                        covered += 1;
                        any = true;
                        cases.push(Case {
                            token,
                            prefix_len: p,
                            history: history.clone(),
                            candidates,
                            gold,
                        });
                    }
                    Some(_) => {}
                    None => missed += 1,
                }
            }
            if any {
                tokens.push(Token { len: surface.len() });
            }
            if cases.len() >= max_cases {
                break 'outer;
            }
        }
    }
    Ok(CaseSet {
        cases,
        tokens,
        covered,
        missed,
    })
}

#[derive(Default)]
struct Metrics {
    n: [u64; 4],
    hit1: [u64; 4],
    hit3: [u64; 4],
    rr: [f64; 4],
    latency_ns: Vec<u64>,
    saved: u64,
    typed_total: u64,
}

fn bucket(p: usize) -> usize {
    match p {
        2..=3 => 1,
        4..=5 => 2,
        _ => 3,
    }
}

fn evaluate(variant: &dyn Variant, cases: &[Case], tokens: &[Token]) -> Metrics {
    let mut m = Metrics {
        latency_ns: Vec::with_capacity(cases.len()),
        ..Default::default()
    };
    // Earliest prefix length at which each token appears in the top 3.
    let mut first_top3 = vec![usize::MAX; tokens.len()];
    for case in cases {
        let start = Instant::now();
        let order = variant.order(&case.history, &case.candidates);
        m.latency_ns.push(start.elapsed().as_nanos() as u64);
        let rank = order
            .iter()
            .position(|&i| i == case.gold)
            .expect("gold is a candidate");
        for b in [0, bucket(case.prefix_len)] {
            m.n[b] += 1;
            m.hit1[b] += (rank == 0) as u64;
            m.hit3[b] += (rank < 3) as u64;
            m.rr[b] += 1.0 / (rank + 1) as f64;
        }
        if rank < 3 {
            let first = &mut first_top3[case.token];
            *first = (*first).min(case.prefix_len);
        }
    }
    for (token, &first) in tokens.iter().zip(&first_top3) {
        m.typed_total += token.len as u64;
        if first != usize::MAX {
            // Type `first` characters, then one tap to pick the suggestion.
            m.saved += (token.len as u64).saturating_sub(first as u64 + 1);
        }
    }
    m.latency_ns.sort_unstable();
    m
}

fn pct(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut test = None;
    let mut max_cases = 50_000usize;
    let mut penalty = krom_experiment::DEFAULT_STEM_PENALTY;
    let mut specs = Vec::new();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--test" => test = Some(PathBuf::from(value()?)),
            "--max-cases" => max_cases = value()?.parse().map_err(|_| "bad --max-cases")?,
            "--stem-penalty" => penalty = value()?.parse().map_err(|_| "bad --stem-penalty")?,
            "--variant" => specs.push(value()?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let test = test.ok_or("--test is required")?;
    let variants: Vec<_> = specs
        .iter()
        .map(|s| open_variant(s, penalty))
        .collect::<Result<_, _>>()?;

    let started = Instant::now();
    let CaseSet {
        cases,
        tokens,
        covered,
        missed,
    } = build_cases(&test, max_cases)?;
    let mean_cands =
        cases.iter().map(|c| c.candidates.len()).sum::<usize>() as f64 / cases.len().max(1) as f64;
    println!(
        "{} cases over {} words; target among candidates for {:.1}% of prefixes; {:.1} candidates/case; built in {:.1}s\n",
        cases.len(),
        tokens.len(),
        pct(covered, covered + missed),
        mean_cands,
        started.elapsed().as_secs_f64()
    );
    println!(
        "| variant | Hit@1 | Hit@3 | MRR | Hit@1 p2-3 | Hit@1 p4-5 | Hit@1 p6+ | KSR@3 | p50 µs | p99 µs | size MB |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for (name, variant, size) in &variants {
        let m = evaluate(variant.as_ref(), &cases, &tokens);
        let q =
            |p: f64| m.latency_ns[((m.latency_ns.len() as f64 - 1.0) * p) as usize] as f64 / 1000.0;
        println!(
            "| {name} | {:.2} | {:.2} | {:.4} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} | {:.1} |",
            pct(m.hit1[0], m.n[0]),
            pct(m.hit3[0], m.n[0]),
            m.rr[0] / m.n[0].max(1) as f64,
            pct(m.hit1[1], m.n[1]),
            pct(m.hit1[2], m.n[2]),
            pct(m.hit1[3], m.n[3]),
            pct(m.saved, m.typed_total),
            q(0.5),
            q(0.99),
            *size as f64 / 1e6,
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("krom-eval: {e}");
            ExitCode::FAILURE
        }
    }
}
