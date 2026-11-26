"""Generate synthetic nanoGPT ONNX benchmarks and zkInfer workload configs.

Architecture adapted from https://github.com/karpathy/nanoGPT
Preserves the finite attention mask and opset used by the supplied generator.
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml


SEED = 42
BLOCK_SIZE = 64
SEQ_LENGTH = 64
VOCAB_SIZE = 65
N_HEAD = 4
OPSET_VERSION = 10

MODEL_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = MODEL_DIR.parent.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent
WORKLOAD_DIR = EXPERIMENTS_DIR / 'config' / 'workload'

CONFIGURATIONS = [
    (2, 32),
    (4, 32),
    (2, 64),
    (4, 64),
    (8, 64),
    (10, 64),
    (20, 64),
    (2, 128),
    (4, 128),
    (8, 128),
]


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    ))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(
            config.block_size, config.block_size
        )).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float(-10))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 4
    n_head: int = N_HEAD
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        count = sum(p.numel() for p in self.parameters())
        if non_embedding:
            count -= self.transformer.wpe.weight.numel()
        return count

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


def generate_input():
    """Reuse existing tokens; never silently replace an existing input file."""
    path = MODEL_DIR / 'input.json'
    shape = [1, SEQ_LENGTH]
    if path.exists():
        data = json.loads(path.read_text())
        if data.get('input_shapes') != [shape] or len(data['input_data']) != 1:
            raise ValueError(f'Expected one input of shape {shape} in {path}')
        tokens = data['input_data'][0]
        if len(tokens) != SEQ_LENGTH or any(
            not isinstance(v, (int, float)) or not 0 <= v < VOCAB_SIZE or int(v) != v
            for v in tokens
        ):
            raise ValueError(f'Invalid token IDs in {path}')
        x = torch.tensor(tokens, dtype=torch.long).reshape(shape)
        if 'output_data' in data:
            # The original output is model-specific. Keep the original file intact.
            path = MODEL_DIR / 'input_shared.json'
            shared = {'input_shapes': [shape], 'input_data': [x.flatten().tolist()]}
            if path.exists() and json.loads(path.read_text()) != shared:
                raise ValueError(f'Existing {path} differs from original tokens')
            if not path.exists():
                path.write_text(json.dumps(shared, indent=4) + '\n')
            print(f'Original input.json contains model-specific outputs; using {path.name} for new configs.')
    else:
        rng = torch.Generator().manual_seed(SEED)
        x = torch.randint(VOCAB_SIZE, tuple(shape), generator=rng, dtype=torch.long)
        path.write_text(json.dumps({'input_shapes': [shape],
                                    'input_data': [x.flatten().tolist()]}, indent=4) + '\n')
    return x, path


def generate_model(n_layer, n_embd, x, input_path, overwrite=False):
    if n_layer <= 0 or n_embd <= 0 or n_embd % N_HEAD:
        raise ValueError('Layers must be positive; embedding must be a positive multiple of N_HEAD')
    name = f'nano_gpt_{n_layer}_layers_{n_embd}_embd'
    onnx_path = MODEL_DIR / f'{name}.onnx'
    yaml_path = WORKLOAD_DIR / f'{name}.yaml'
    if onnx_path.exists() and not overwrite:
        print(f'Skipping existing model: {name}')
    else:
        torch.manual_seed(SEED)
        model = GPT(GPTConfig(n_layer=n_layer, n_embd=n_embd)).cpu().eval()
        with torch.no_grad():
            torch.onnx.export(
                model, x, str(onnx_path),
                export_params=True, opset_version=OPSET_VERSION,
                do_constant_folding=True, input_names=['input'],
                output_names=['output'], dynamo=False,
            )
        print(f'Generated {name} | parameters={model.get_num_params(False):,} | '
              f'size={onnx_path.stat().st_size / 1e6:.2f} MB')
    onnx.checker.check_model(str(onnx_path))
    workload = {
        'name': name,
        'onnx_file': onnx_path.relative_to(PROJECT_DIR).as_posix(),
        'input_file': input_path.relative_to(PROJECT_DIR).as_posix(),
        'input_shapes': {'input': list(x.shape)},
    }
    if not yaml_path.exists() or overwrite:
        yaml_path.write_text(yaml.safe_dump(workload, sort_keys=False))
        print(f'Workload: {yaml_path.name}')
    else:
        print(f'Preserving existing workload: {yaml_path.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--embd', type=int)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing models and workload configs')
    args = parser.parse_args()
    if (args.layers is None) != (args.embd is None):
        parser.error('--layers and --embd must be supplied together')
    configs = CONFIGURATIONS if args.layers is None else [(args.layers, args.embd)]
    if any(l <= 0 or e <= 0 or e % N_HEAD for l, e in configs):
        parser.error(f'Layers must be positive and embedding must be a positive multiple of {N_HEAD}')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    x, input_path = generate_input()
    for n_layer, n_embd in configs:
        generate_model(n_layer, n_embd, x, input_path, args.overwrite)


if __name__ == '__main__':
    main()
