use std::path::Path;
use std::time::Instant;

#[global_allocator]
static PEAK_ALLOC: peak_alloc::PeakAlloc = peak_alloc::PeakAlloc;

mod predictor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bangla_gru_sp");

    println!("Loading model from {:?}...", model_dir);
    let mut predictor = predictor::Predictor::new(&model_dir)?;
    println!("Model loaded. Vocabulary size: {}", predictor.vocab.len());

    // Demo predictions
    let examples: &[(&str, &str)] = &[
        ("আমি বাংলায়", ""),
        ("আমি বাংলায়", "গ"),
        ("বাংলাদেশের রাজধানী", "ঢা"),
        ("আজকে আমি", "কা"),
        ("সে স্কুলে", ""),
    ];

    println!("\n{}", "=".repeat(60));
    println!("  Bangla Next Word Prediction (ONNX + Rust)");
    println!("{}", "=".repeat(60));

    for &(text, prefix) in examples {
        println!("\nInput: {text}");
        if !prefix.is_empty() {
            println!("Prefix: '{prefix}'");
        }
        
        let start = Instant::now();
        
        let preds = predictor.predict(text, prefix, 5);
        
        let duration = start.elapsed();
        
        if preds.is_empty() {
            println!("  (no predictions)");
        } else {
            for (i, (word, prob)) in preds.iter().enumerate() {
                println!("  {}. {:12} {:.1}%", i + 1, word, prob * 100.0);
            }
        }
        println!("Execution time: {:?}\n", duration);
    }

    // Interactive mode
    // println!("\n{}", "=".repeat(60));
    // println!("  Interactive mode \u{2014} format: <text> | <prefix>");
    // println!("  Type 'quit' to exit");
    // println!("{}", "=".repeat(60));

    // let stdin = io::stdin();
    // loop {
    //     print!("\n> ");
    //     io::stdout().flush()?;

    //     let mut line = String::new();
    //     if stdin.lock().read_line(&mut line)? == 0 {
    //         break;
    //     }
    //     let line = line.trim();
    //     if line.is_empty() {
    //         continue;
    //     }
    //     if line == "quit" || line == "exit" || line == "q" {
    //         break;
    //     }

    //     let (text, prefix) = if let Some((t, p)) = line.split_once('|') {
    //         (t.trim(), p.trim())
    //     } else {
    //         (line, "")
    //     };

    //     let preds = predictor.predict(text, prefix, 10);
    //     if preds.is_empty() {
    //         println!(
    //             "  (no predictions{})",
    //             if prefix.is_empty() {
    //                 ""
    //             } else {
    //                 " matching prefix"
    //             }
    //         );
    //     } else {
    //         for (i, (word, prob)) in preds.iter().enumerate() {
    //             println!("  {:2}. {} ({:.1}%)", i + 1, word, prob * 100.0);
    //         }
    //     }
    // }
    // 
    
    let current_mem = PEAK_ALLOC.current_usage_as_mb();
    println!("This program currently uses {} MB of RAM.", current_mem);
    let peak_mem = PEAK_ALLOC.peak_usage_as_mb();
    println!("The max amount that was used {} MB.", peak_mem);

    Ok(())
}
