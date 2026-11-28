"""Randomly initialized VAE decoder adapted from the zkbc benchmark."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

OUTPUT_DIR = Path(__file__).resolve().parent

class SmallVAE(nn.Module):
    def __init__(self, channels=32, spatial_size=16):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.fc_decode = nn.Linear(128, channels * spatial_size**2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels // 2, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(self.fc_decode(z).reshape(-1, self.channels, self.spatial_size, self.spatial_size))


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

    with (OUTPUT_DIR / ("input.json" if filename == "vae_decoder.onnx" else Path(filename).stem + "_input.json")).open("w") as f:
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
        "original": (32, 16, "vae_decoder.onnx"),
        "medium": (64, 16, "vae_decoder_medium.onnx"),
        "large": (64, 32, "vae_decoder_large.onnx"),
    }
    for name, (channels, spatial, filename) in variants.items():
        if args.size not in ("all", name):
            continue
        torch.manual_seed(42)
        export(SmallVAE(channels, spatial), 0.1 * torch.rand(1, 128),
               filename, ["output"])
