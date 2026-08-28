# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark full-window and KV-cache NeRD Transformer inference."""

import argparse
import json
import time

import torch
from isaaclab_neural.models.models import ModelMixedInput


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _create_model(args):
    input_sample = {"observation": torch.zeros(args.batch_size, 1, args.input_dim, device=args.device)}
    input_cfg = {"low_dim": ["observation"]}
    network_cfg = {
        "normalize_input": False,
        "normalize_output": False,
        "encoder": {
            "low_dim": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            }
        },
        "transformer": {
            "block_size": max(32, args.history),
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": 0.0,
            "bias": False,
            "position_encoding": "rope",
        },
        "model": {
            "mlp": {
                "layer_sizes": [64],
                "activation": "relu",
                "layernorm": False,
            }
        },
    }
    return ModelMixedInput(
        input_sample,
        args.output_dim,
        input_cfg,
        network_cfg,
        device=args.device,
    ).eval()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(callable_, warmup, iterations, device):
    for _ in range(warmup):
        callable_()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        callable_()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main():
    """Run the model-only inference benchmark."""
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but CUDA is unavailable.")
    if args.n_embd % args.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head.")

    torch.manual_seed(args.seed)
    model = _create_model(args)
    sequence = torch.randn(args.batch_size, args.history, args.input_dim, device=device)

    with torch.inference_mode():
        full_output = model({"observation": sequence})
        cache = None
        cached_outputs = []
        for step in range(args.history):
            output, cache = model.forward_cached(
                {"observation": sequence[:, step : step + 1]},
                cache,
                max_cache_length=args.history,
            )
            cached_outputs.append(output)
        cached_output = torch.cat(cached_outputs, dim=1)
        max_abs_error = (full_output - cached_output).abs().max().item()

        full_input = {"observation": sequence}

        def full_forward():
            model(full_input, single_step=True)

        cached_input = {"observation": sequence[:, -1:]}
        cache = model.init_transformer_cache(args.batch_size, args.history)
        for step in range(args.history):
            _, cache = model.forward_cached(
                {"observation": sequence[:, step : step + 1]},
                cache,
            )

        def cached_forward():
            model.forward_cached(cached_input, cache)

        full_ms = _measure(full_forward, args.warmup, args.iterations, device)
        cached_ms = _measure(cached_forward, args.warmup, args.iterations, device)

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "history": args.history,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "prefix_max_abs_error": max_abs_error,
        "full_window_ms": full_ms,
        "kv_cache_ms": cached_ms,
        "speedup": full_ms / cached_ms,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
