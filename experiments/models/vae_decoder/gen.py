"""Randomly initialized VAE decoder adapted from the zkbc benchmark."""
import json
from pathlib import Path

import torch
import torch.nn as nn

OUTPUT_DIR = Path(__file__).resolve().parent

class SmallVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_decode = nn.Linear(128, 32 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(self.fc_decode(z).reshape(-1, 32, 16, 16))


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

    with (OUTPUT_DIR / "input.json").open("w") as f:
        json.dump({
            "input_data": [x.flatten().tolist()],
            "output_data": [y.flatten().tolist() for y in outputs],
        }, f)

    print(f"Exported {filename}: input={tuple(x.shape)}, "
          f"outputs={[tuple(y.shape) for y in outputs]}, "
          f"parameters={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    torch.manual_seed(42)
    export(SmallVAE(), 0.1 * torch.rand(1, 128),
           "vae_decoder.onnx", ["output"])
