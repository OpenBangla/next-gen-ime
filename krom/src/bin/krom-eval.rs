//! Evaluate ranking on upodesh 0.4.0 candidates.
//!
//! Every word of the test set is typed by a simulated user: at each prefix
//! length p ≥ 2 the candidates are `upodesh::bangla::suggest(prefix)` and the
//! model ranks them given the two previous words. Cases where the target is
//! not among ≥ 2 candidates are not scored (coverage is reported). Each
//! model is compared with upodesh's own order and with its unigram-only
//! ranking.
//!
//! ```text
//! krom-eval --test data/test.txt --model model [--model other] [--max-cases 100000]
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::time::Instant;

use krom::Ranker;
use krom::normalize::{fold, to_upodesh_form};
use krom::tokenize::{BOS, EOS};

struct Case {
    token: usize,
    prefix_len: usize,
    history: Vec<String>,
    candidates: Rc<Vec<String>>,
    gold: usize,
}

struct CaseSet {
    cases: Vec<Case>,
    /// Length in code points of each evaluated word.
    tokens: Vec<usize>,
    /// Prefixes whose candidates contained / missed the target.
    covered: u64,
    missed: u64,
}

fn build_cases(test: &Path, max_cases: usize) -> Result<CaseSet, String> {
    let text = fs::read_to_string(test).map_err(|e| format!("{}: {e}", test.display()))?;
    let mut cache: HashMap<String, Rc<Vec<String>>> = HashMap::new();
    let mut set = CaseSet {
        cases: Vec::new(),
        tokens: Vec::new(),
        covered: 0,
        missed: 0,
    };
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
            let token = set.tokens.len();
            let mut any = false;
            // upodesh 0.4.0 underflows on 1-character prefixes.
            for p in 2..surface.len() {
                let prefix: String = surface[..p].iter().collect();
                let candidates = cache
                    .entry(prefix.clone())
                    .or_insert_with(|| Rc::new(upodesh::bangla::suggest(&prefix)))
                    .clone();
                match candidates.iter().position(|c| fold(c) == target) {
                    Some(gold) if candidates.len() >= 2 => {
                        set.covered += 1;
                        any = true;
                        set.cases.push(Case {
                            token,
                            prefix_len: p,
                            history: history.clone(),
                            candidates,
                            gold,
                        });
                    }
                    Some(_) => {}
                    None => set.missed += 1,
                }
            }
            if any {
                set.tokens.push(surface.len());
            }
            if set.cases.len() >= max_cases {
                break 'outer;
            }
        }
    }
    Ok(set)
}

struct Metrics {
    n: u64,
    hit1: u64,
    hit3: u64,
    rr: f64,
    /// Keystroke saving rate when picking the word once it is in the top 3.
    ksr: f64,
    latency_ns: Vec<u64>,
}

/// `order` returns candidate indices, best first.
fn evaluate(set: &CaseSet, order: impl Fn(&Case) -> Vec<usize>) -> Metrics {
    let mut m = Metrics {
        n: 0,
        hit1: 0,
        hit3: 0,
        rr: 0.0,
        ksr: 0.0,
        latency_ns: Vec::new(),
    };
    let mut first_top3 = vec![usize::MAX; set.tokens.len()];
    for case in &set.cases {
        let start = Instant::now();
        let ranked = order(case);
        m.latency_ns.push(start.elapsed().as_nanos() as u64);
        let rank = ranked
            .iter()
            .position(|&i| i == case.gold)
            .expect("gold is a candidate");
        m.n += 1;
        m.hit1 += (rank == 0) as u64;
        m.hit3 += (rank < 3) as u64;
        m.rr += 1.0 / (rank + 1) as f64;
        if rank < 3 {
            first_top3[case.token] = first_top3[case.token].min(case.prefix_len);
        }
    }
    let (mut saved, mut typed) = (0u64, 0u64);
    for (&len, &first) in set.tokens.iter().zip(&first_top3) {
        typed += len as u64;
        if first != usize::MAX {
            // Type `first` characters, then one tap to pick the suggestion.
            saved += (len as u64).saturating_sub(first as u64 + 1);
        }
    }
    m.ksr = 100.0 * saved as f64 / typed.max(1) as f64;
    m.latency_ns.sort_unstable();
    m
}

fn model_size(dir: &Path) -> f64 {
    ["vocab.fst", "ngram.fst"]
        .iter()
        .map(|f| fs::metadata(dir.join(f)).map(|m| m.len()).unwrap_or(0))
        .sum::<u64>() as f64
        / 1e6
}

fn print_row(name: &str, m: &Metrics, size: Option<f64>) {
    let pct = |a: u64| 100.0 * a as f64 / m.n.max(1) as f64;
    let q = |p: f64| m.latency_ns[((m.latency_ns.len() - 1) as f64 * p) as usize] as f64 / 1e3;
    println!(
        "| {name} | {:.2} | {:.2} | {:.4} | {:.2} | {:.1} | {:.1} | {} |",
        pct(m.hit1),
        pct(m.hit3),
        m.rr / m.n.max(1) as f64,
        m.ksr,
        q(0.5),
        q(0.99),
        size.map_or("–".into(), |s| format!("{s:.1}")),
    );
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut test = None;
    let mut max_cases = 100_000usize;
    let mut models: Vec<PathBuf> = Vec::new();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--test" => test = Some(PathBuf::from(value()?)),
            "--max-cases" => max_cases = value()?.parse().map_err(|_| "bad --max-cases")?,
            "--model" => models.push(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let test = test.ok_or("--test is required")?;
    if models.is_empty() {
        return Err("at least one --model is required".into());
    }

    let set = build_cases(&test, max_cases)?;
    if set.cases.is_empty() {
        return Err("no evaluable cases".into());
    }
    let mean =
        set.cases.iter().map(|c| c.candidates.len()).sum::<usize>() as f64 / set.cases.len() as f64;
    println!(
        "{} cases over {} words; target among candidates for {:.1}% of prefixes; {mean:.1} candidates/case\n",
        set.cases.len(),
        set.tokens.len(),
        100.0 * set.covered as f64 / (set.covered + set.missed).max(1) as f64,
    );
    println!("| ranking | Hit@1 | Hit@3 | MRR | KSR@3 | p50 µs | p99 µs | size MB |");
    println!("|---|---|---|---|---|---|---|---|");
    print_row(
        "upodesh order",
        &evaluate(&set, |c| (0..c.candidates.len()).collect()),
        None,
    );
    for dir in &models {
        let name = dir.display().to_string();
        let open = || Ranker::open(dir).map_err(|e| format!("{name}: {e}"));
        let rank = |ranker: &Ranker, c: &Case| -> Vec<usize> {
            ranker
                .rank_history(&c.history, &c.candidates)
                .iter()
                .map(|r| r.index)
                .collect()
        };
        let unigram = open()?.with_max_order(1);
        print_row(
            &format!("{name} (unigram only)"),
            &evaluate(&set, |c| rank(&unigram, c)),
            None,
        );
        let ranker = open()?;
        print_row(
            &name,
            &evaluate(&set, |c| rank(&ranker, c)),
            Some(model_size(dir)),
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
