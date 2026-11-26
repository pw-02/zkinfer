
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml


SEED = 42
INPUT_DIM = 784
OUTPUT_DIM = 10
OPSET_VERSION = 13

MODEL_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = MODEL_DIR.parent.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent
WORKLOAD_DIR = EXPERIMENTS_DIR / "config" / "workload"

CONFIGURATIONS = [
    (2, 128),
    (4, 128),
    (8, 128),
    (4, 256),
    (4, 512),
    (4, 1024),
]


class MLP(nn.Module):
    def __init__(self, depth, width):
        super().__init__()

        layers = []
        in_features = INPUT_DIM

        for _ in range(depth):
            layers.extend([
                nn.Linear(in_features, width),
                nn.ReLU(),
            ])
            in_features = width

        layers.append(nn.Linear(width, OUTPUT_DIM))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def generate_input():
    rng = np.random.default_rng(SEED)

    x = rng.uniform(
        0.0, 1.0, size=(1, INPUT_DIM)
    ).astype(np.float32)

    input_path = MODEL_DIR / "input.json"

    data = {
        "input_data": [x.flatten().tolist()],
        "input_shapes": [list(x.shape)],
    }

    with input_path.open("w") as f:
        json.dump(data, f, indent=4)

    return torch.from_numpy(x), input_path


def generate_model(depth, width, x, input_path):
    torch.manual_seed(SEED)

    name = f"mlp_{depth}_layers_{width}_width"
    model = MLP(depth, width).eval()

    onnx_path = MODEL_DIR / f"{name}.onnx"

    torch.onnx.export(
        model,
        x,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET_VERSION,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(onnx_path))

    workload = {
        "name": name,
        "onnx_file": onnx_path.relative_to(
            PROJECT_DIR
        ).as_posix(),
        "input_file": input_path.relative_to(
            PROJECT_DIR
        ).as_posix(),
        "input_shapes": {
            "input": [1, INPUT_DIM],
        },
    }

    yaml_path = WORKLOAD_DIR / f"{name}.yaml"

    with yaml_path.open("w") as f:
        yaml.safe_dump(
            workload,
            f,
            sort_keys=False,
        )

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Generated {name} | "
        f"parameters={parameters:,} | "
        f"size={onnx_path.stat().st_size / 1e6:.2f} MB"
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)

    x, input_path = generate_input()

    for depth, width in CONFIGURATIONS:
        generate_model(depth, width, x, input_path)


if __name__ == "__main__":
    main()
