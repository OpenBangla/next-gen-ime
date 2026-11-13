//! Build krom models.
//!
//! ```text
//! krom-build prepare --corpus bn.txt --out data/ [--max-train-tokens 100000000]
//! krom-build sb      --train data/train.txt --out model-sb [--min-count 3] [--prune 2,2] [--limit-vocab words.txt] [--scale 2048] [--epsilon 0]
//! krom-build arpa    --arpa lm.arpa --out model-kn [--limit-vocab words.txt] [--scale 2048]
//! krom-build stems   --train data/train.txt --model model-sb [--analyzer rule|sp:PATH] [--prune 2] [--scale 2048]
//!
//! `--scale` is the number of quantization levels per log10 unit.
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use krom_experiment::build::{self, SbOptions, StemOptions};
use krom_experiment::format::{AnalyzerKind, DEFAULT_SCALE};

fn parse_flags(args: impl Iterator<Item = String>) -> Result<HashMap<String, String>, String> {
    let mut flags = HashMap::new();
    let mut args = args.peekable();
    while let Some(flag) = args.next() {
        let name = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {flag}"))?;
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        flags.insert(name.to_owned(), value);
    }
    Ok(flags)
}

fn path(flags: &HashMap<String, String>, name: &str) -> Result<PathBuf, String> {
    flags
        .get(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("--{name} is required"))
}

fn num<T: std::str::FromStr>(
    flags: &HashMap<String, String>,
    name: &str,
    default: T,
) -> Result<T, String> {
    flags.get(name).map_or(Ok(default), |v| {
        v.parse().map_err(|_| format!("bad --{name}"))
    })
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .ok_or("missing command: prepare | sb | arpa | stems")?;
    let flags = parse_flags(args)?;
    let limit = |flags: &HashMap<String, String>| {
        flags
            .get("limit-vocab")
            .map(|p| build::read_word_list(p.as_ref()).map_err(|e| e.to_string()))
            .transpose()
    };
    let started = Instant::now();
    match command.as_str() {
        "prepare" => {
            let stats = build::prepare(
                &path(&flags, "corpus")?,
                &path(&flags, "out")?,
                num(&flags, "max-train-tokens", u64::MAX)?,
            )
            .map_err(|e| e.to_string())?;
            println!("{stats:?}");
        }
        "sb" => {
            let prune: Vec<u32> = flags
                .get("prune")
                .map_or("2,2", String::as_str)
                .split(',')
                .map(|v| v.parse().map_err(|_| "bad --prune".to_owned()))
                .collect::<Result<_, _>>()?;
            let [p2, p3] = prune[..] else {
                return Err("--prune takes two values".into());
            };
            let opts = SbOptions {
                min_count: num(&flags, "min-count", 3)?,
                prune: [p2, p3],
                limit_vocab: limit(&flags)?,
                scale: num(&flags, "scale", DEFAULT_SCALE)?,
                epsilon: num(&flags, "epsilon", 0.0)?,
            };
            let stats =
                build::build_stupid_backoff(&path(&flags, "train")?, &path(&flags, "out")?, &opts)
                    .map_err(|e| e.to_string())?;
            println!("{stats:?}");
        }
        "arpa" => {
            let limit = limit(&flags)?;
            let stats = build::arpa::build_from_arpa(
                &path(&flags, "arpa")?,
                &path(&flags, "out")?,
                limit.as_ref(),
                num(&flags, "scale", DEFAULT_SCALE)?,
            )
            .map_err(|e| e.to_string())?;
            println!("{stats:?}");
        }
        "stems" => {
            let analyzer = match flags.get("analyzer").map(String::as_str) {
                None | Some("rule") => AnalyzerKind::Rule,
                Some(other) => match other.strip_prefix("sp:") {
                    Some(model) => AnalyzerKind::SentencePiece(model.to_owned()),
                    None => return Err(format!("unknown analyzer {other}")),
                },
            };
            let opts = StemOptions {
                analyzer,
                prune: num(&flags, "prune", 2)?,
                scale: num(&flags, "scale", DEFAULT_SCALE)?,
            };
            let stats = build::build_stems(&path(&flags, "train")?, &path(&flags, "model")?, &opts)
                .map_err(|e| e.to_string())?;
            println!("{stats:?}");
        }
        other => return Err(format!("unknown command {other}")),
    }
    eprintln!("done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("krom-build: {e}");
            ExitCode::FAILURE
        }
    }
}
