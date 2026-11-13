//! Interactive check: type `context words | bengali prefix`.
//!
//! `cargo run --release --example rank -- model-sb`

use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "model-sb".into());
    let ranker = krom_experiment::Ranker::open(dir.as_ref())?;
    let stdin = io::stdin();
    print!("context | prefix> ");
    io::stdout().flush()?;
    for line in stdin.lock().lines() {
        let line = line?;
        let (context, prefix) = line.rsplit_once('|').unwrap_or(("", line.as_str()));
        let candidates = upodesh::bangla::suggest(prefix.trim());
        for r in ranker.rank_text(context, &candidates).iter().take(10) {
            println!("{:>8.3}  {}", r.score, r.word);
        }
        print!("context | prefix> ");
        io::stdout().flush()?;
    }
    Ok(())
}
