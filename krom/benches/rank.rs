//! Ranking latency on real upodesh candidates.
//!
//! `KROM_MODEL=model cargo bench` (defaults to `model`).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench(c: &mut Criterion) {
    let dir = std::env::var("KROM_MODEL").unwrap_or_else(|_| "model".into());
    let Ok(ranker) = krom::Ranker::open(dir.as_ref()) else {
        eprintln!("no model at {dir}; set KROM_MODEL");
        return;
    };
    for prefix in ["প্রতি", "বাংলা"] {
        let candidates = upodesh::bangla::suggest(prefix);
        c.bench_function(
            &format!("rank {} candidates of {prefix}", candidates.len()),
            |b| b.iter(|| ranker.rank(black_box(&["আমি", "বাড়ির"]), black_box(&candidates))),
        );
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
