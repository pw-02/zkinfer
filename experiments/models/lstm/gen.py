"""Randomly initialized two-layer LSTM adapted from the zkbc benchmark."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

OUTPUT_DIR = Path(__file__).resolve().parent

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(128, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


def export(model, x, filename, output_names):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

    torch.onnx.export(
        model, x, str(OUTPUT_DIR / filename),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        opset_version=18,
        dynamo=False,
    )

    with (OUTPUT_DIR / ("input.json" if filename == "lstm.onnx" else Path(filename).stem + "_input.json")).open("w") as f:
        json.dump({
            "input_data": [x.flatten().tolist()],
            "output_data": [y.flatten().tolist() for y in outputs],
        }, f)

    print(f"Exported {filename}: input={tuple(x.shape)}, "
          f"outputs={[tuple(y.shape) for y in outputs]}, "
          f"parameters={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["original", "medium", "large", "all"],
                        default="all")
    args = parser.parse_args()
    variants = {
        "original": (32, 2, 32, "lstm.onnx"),
        "medium": (64, 2, 64, "lstm_medium.onnx"),
        "large": (128, 2, 128, "lstm_large.onnx"),
    }
    for name, (hidden, layers, sequence, filename) in variants.items():
        if args.size not in ("all", name):
            continue
        torch.manual_seed(42)
        export(SimpleLSTM(hidden, layers), 0.1 * torch.rand(1, sequence, 128),
               filename, ["hidden_state", "cell_state"])
