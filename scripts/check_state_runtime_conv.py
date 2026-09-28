#!/usr/bin/env python3
"""CPU-only checks of the runtime's actual native fallback convolution helper.

The helper is extracted from state_quant.py with Python's AST so this check
does not import the CUDA/Triton state codec or allocate model weights. GPU
visibility is disabled before importing torch. Only the CPU fallback branches
are exercised; the receipt makes no whole-model or GPU-kernel parity claim.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper():
    path = ROOT / "mamba2_recall" / "state_quant.py"
    module = ast.parse(path.read_text())
    controller = next(node for node in module.body
                      if isinstance(node, ast.ClassDef) and node.name == "StateQuant")
    helper = next(node for node in controller.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_convolution")
    if len(helper.decorator_list) != 1 or not isinstance(helper.decorator_list[0], ast.Name) \
            or helper.decorator_list[0].id != "staticmethod":
        raise RuntimeError("Expected the runtime's static convolution helper")
    helper.decorator_list = []
    definition = ast.Module(body=[helper], type_ignores=[])
    namespace = {"torch": torch, "F": F}
    exec(compile(definition, str(path), "exec"), namespace)
    return namespace["_convolution"], path


@torch.no_grad()
def check_geometry(convolution, channels):
    torch.manual_seed(527 + channels)
    mx = types.SimpleNamespace(
        conv1d=torch.nn.Conv1d(channels, channels, 4, groups=channels,
                              padding=3, dtype=torch.float16),
        act=torch.nn.SiLU(), activation="silu")
    x = torch.randn(2, 13, channels, dtype=torch.float16)

    def cache():
        value = torch.zeros(2, 4, channels, dtype=torch.float16).transpose(1, 2)
        return types.SimpleNamespace(conv=value, tokens=0)

    def native_prefill(value):
        return mx.act(mx.conv1d(value.transpose(1, 2)).transpose(1, 2)[:, :-3])

    rows = []
    for length in (1, 2, 13):
        entry = cache()
        got = convolution(mx, x[:, :length], entry)
        expected = native_prefill(x[:, :length])
        expected_state = F.pad(x[:, :length].transpose(1, 2), (4-length, 0))
        if not torch.equal(got, expected) or not torch.equal(entry.conv, expected_state):
            raise RuntimeError(f"Native prefill mismatch, channels={channels}, tokens={length}")
        rows.append({"check": "native_prefill", "tokens": length,
                     "output_exact": True, "cache_exact": True})

    entry = cache()
    convolution(mx, x[:, :7], entry)
    entry.tokens = 7
    expected_state = torch.roll(entry.conv, shifts=-1, dims=-1)
    expected_state[:, :, -1] = x[:, 7]
    expected = mx.act((expected_state * mx.conv1d.weight[:, 0]).sum(-1)
                      + mx.conv1d.bias)[:, None]
    got = convolution(mx, x[:, 7:8], entry)
    if not torch.equal(got, expected) or not torch.equal(entry.conv, expected_state):
        raise RuntimeError(f"Native decode mismatch, channels={channels}")
    rows.append({"check": "native_decode", "output_exact": True, "cache_exact": True})

    entry = cache()
    first = convolution(mx, x[:, :5], entry)
    entry.tokens = 5
    second = convolution(mx, x[:, 5:], entry)
    difference = (torch.cat((first, second), dim=1) - native_prefill(x)).abs().max().item()
    if difference >= 0.003 or not torch.equal(entry.conv, x[:, -4:].transpose(1, 2)):
        raise RuntimeError(f"Chunk continuation mismatch, channels={channels}, max_abs={difference}")
    rows.append({"check": "chunk_continuation", "split": [5, 8],
                 "max_abs_error": difference, "max_abs_limit": 0.003,
                 "cache_exact": True})

    original = convolution(mx, x, cache())
    changed = x.clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 10
    altered = convolution(mx, changed, cache())
    if not torch.equal(original[:, :8], altered[:, :8]):
        raise RuntimeError(f"Future inputs changed an earlier output, channels={channels}")
    rows.append({"check": "future_token_causality", "unchanged_prefix_tokens": 8,
                 "prefix_exact": True})
    return {"batch": 2, "channels": channels, "conv_width": 4, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Preserve the existing receipt; choose another output")
    torch.set_num_threads(1)
    convolution, source = load_helper()
    from mamba_ssm.modules import mamba2 as native

    original_prefill = native.causal_conv1d_fn
    original_decode = native.causal_conv1d_update
    native.causal_conv1d_fn = None
    native.causal_conv1d_update = None
    try:
        geometries = [check_geometry(convolution, channels) for channels in (7, 10240)]
    finally:
        native.causal_conv1d_fn = original_prefill
        native.causal_conv1d_update = original_decode

    result = {
        "complete": True, "device": "cpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": platform.python_version(), "torch": torch.__version__,
        "native_conv_functions_initially_absent": {
            "prefill": original_prefill is None, "decode": original_decode is None},
        "source_sha256": {str(source.relative_to(ROOT)): sha(source),
                          str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)},
        "geometries": geometries,
        "scope": "Actual AST-extracted runtime helper, native CPU fallback arithmetic and cache updates; no state codec, model weights, GPU allocation, compiled causal-conv kernel or whole-model parity test",
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
