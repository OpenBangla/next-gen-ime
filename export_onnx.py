"""
Export the trained GRU model to ONNX format.

Usage:
    uv run --with onnx --with onnxruntime --with onnxscript python export_onnx.py
"""

import torch
import torch.nn as nn
import numpy as np


class NextWordGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hid_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.drop(self.emb(x))
        output, _ = self.gru(embedded)
        last_hidden = self.drop(output[:, -1, :])
        return self.fc(last_hidden)


class NextWordGRUInference(nn.Module):
    """Dropout-free wrapper for deterministic ONNX export."""

    def __init__(self, model: NextWordGRU):
        super().__init__()
        self.emb = model.emb
        self.gru = model.gru
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.emb(x)
        output, _ = self.gru(embedded)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)


def main():
    model_path = "bangla_gru_sp/model.pt"
    onnx_path = "bangla_gru_sp/model.onnx"

    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    print(f"Config: {config}")

    print("Building model...")
    model = NextWordGRU(
        vocab_size=config["vocab_size"],
        emb_dim=config["emb_dim"],
        hid_dim=config["hid_dim"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Wrap in inference-only model (no dropout nodes in the ONNX graph)
    export_model = NextWordGRUInference(model)
    export_model.eval()

    context_len = config["context_len"]
    dummy_input = torch.zeros(1, context_len, dtype=torch.long)

    print(f"Exporting to ONNX ({onnx_path})...")
    # Use legacy TorchScript exporter — the dynamo exporter mis-converts GRU layers
    torch.onnx.export(
        export_model,
        (dummy_input,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    print(f"Exported to {onnx_path}")

    # Verify the exported model
    import onnxruntime as rt

    sess = rt.InferenceSession(onnx_path)
    out = sess.run(None, {"input_ids": dummy_input.numpy()})
    print(f"Verification - output shape: {out[0].shape}")

    # Compare with the inference model (no dropout)
    with torch.no_grad():
        pt_out = export_model(dummy_input).numpy()

    diff = np.abs(pt_out - out[0]).max()
    print(f"Max absolute difference: {diff:.6e}")
    if diff < 1e-4:
        print("ONNX export verified successfully!")
    else:
        print(f"WARNING: Large difference ({diff:.6e})")


if __name__ == "__main__":
    main()
