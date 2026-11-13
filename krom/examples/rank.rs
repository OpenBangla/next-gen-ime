//! Interactive check: type `context words | bengali prefix`.
//!
//! `cargo run --release --example rank -- model`

use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "model".into());
    let ranker = krom::Ranker::open(dir.as_ref())?;
    let prompt = || {
        print!("context | prefix> ");
        io::stdout().flush()
    };
    prompt()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let (context, prefix) = line.rsplit_once('|').unwrap_or(("", line.as_str()));
        let prefix = prefix.trim();
        // upodesh 0.4.0 underflows on 1-character prefixes.
        if prefix.chars().count() >= 2 {
            let candidates = upodesh::bangla::suggest(prefix);
            for r in ranker.rank_text(context, &candidates).iter().take(10) {
                println!("{:>8.3}  {}", r.score, r.word);
            }
        }
        prompt()?;
    }
    Ok(())
}
