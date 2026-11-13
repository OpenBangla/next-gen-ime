//! Build krom models.
//!
//! ```text
//! krom-build prepare --corpus bn.txt --out data/ [--max-train-tokens 100000000]
//! krom-build build   --train data/train.txt --out model [--limit-vocab words.txt]
//!                    [--min-count 3] [--prune 3,3] [--scale 32]
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use krom::build::{self, Options};

type Flags = HashMap<String, String>;

fn parse_flags(args: impl Iterator<Item = String>) -> Result<Flags, String> {
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

fn path(flags: &Flags, name: &str) -> Result<PathBuf, String> {
    flags
        .get(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("--{name} is required"))
}

fn num<T: std::str::FromStr>(flags: &Flags, name: &str, default: T) -> Result<T, String> {
    flags.get(name).map_or(Ok(default), |v| {
        v.parse().map_err(|_| format!("bad --{name}"))
    })
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("missing command: prepare | build")?;
    let flags = parse_flags(args)?;
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
        "build" => {
            let defaults = Options::default();
            let prune = match flags.get("prune") {
                None => defaults.prune,
                Some(v) => match v.split_once(',').map(|(a, b)| (a.parse(), b.parse())) {
                    Some((Ok(a), Ok(b))) => [a, b],
                    _ => return Err("--prune takes BIGRAM,TRIGRAM minimum counts".into()),
                },
            };
            let limit_vocab = flags
                .get("limit-vocab")
                .map(|p| build::read_word_list(p.as_ref()).map_err(|e| e.to_string()))
                .transpose()?;
            let opts = Options {
                min_count: num(&flags, "min-count", defaults.min_count)?,
                prune,
                limit_vocab,
                scale: num(&flags, "scale", defaults.scale)?,
            };
            let stats = build::build(&path(&flags, "train")?, &path(&flags, "out")?, &opts)
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
