# next-gen-ime
Building the Next Generation of Bangla IME!

### export_onnx.py — ONNX Export Script

  - Loads the PyTorch GRU checkpoint from bangla_gru_sp/model.pt
  - Wraps in NextWordGRUInference (strips dropout for deterministic export)
  - Uses the legacy TorchScript exporter (dynamo=False) — the dynamo exporter
  mis-converts GRU layers
  - Verifies the exported ONNX model matches PyTorch output (max diff 4.38e-06)
  - Run: uv run --with onnx --with onnxruntime python export_onnx.py

### inference/ — Rust Inference Project

  Cargo.toml — depends on ort (bundled ONNX Runtime, ndarray feature), ndarray,
  sentencepiece

  src/main.rs — Predictor struct that:
  - Loads the ONNX model via ort::Session
  - Tokenizes input with SentencePieceProcessor
  - Builds the vocab id-to-piece mapping from sp_bangla.vocab
  - predict(text, prefix, top_k) — runs inference and returns top-k results
  - top_k_with_prefix() — masks non-matching logits to -inf before softmax (same
   algorithm as the Python predict_with_prefix_advanced)
  - Includes demo examples and an interactive <text> | <prefix> CLI
  - Run: cd inference && cargo run

  Note: requires brew install sentencepiece so the sentencepiece Rust crate
  links against the system library (avoids protobuf symbol conflicts with ort's
  bundled ONNX Runtime).
