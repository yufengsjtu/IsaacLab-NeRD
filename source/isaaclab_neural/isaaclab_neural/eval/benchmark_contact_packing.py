# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark contact-token packing independently of the simulator."""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch
from isaaclab_neural.contacts.contact_set_encoder import (
    ContactSetEncoder,
    _DirectedContactRows,
    mask_invalid_contact_tokens,
    transform_contact_tokens_to_body_frame,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_envs", nargs="+", type=int, default=[1, 64, 512, 1024])
    parser.add_argument("--contacts_per_env", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--bodies_per_env", type=int, default=4)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_encoder(
    num_envs: int,
    bodies_per_env: int,
    capacity: int,
    device: torch.device,
) -> ContactSetEncoder:
    body_count = num_envs * bodies_per_env
    body_world = torch.arange(num_envs).repeat_interleave(bodies_per_env)
    model = SimpleNamespace(
        body_count=body_count,
        body_world=body_world,
        body_com=torch.zeros(body_count, 3),
    )
    return ContactSetEncoder(
        model=model,
        primary_body_mask=torch.ones(body_count, dtype=torch.bool, device=device),
        shape_body_torch=torch.arange(body_count, device=device),
        bodies_per_env=bodies_per_env,
        num_envs=num_envs,
        max_contact_tokens=capacity,
        device=device,
    )


def _build_rows(
    num_envs: int,
    contacts_per_env: int,
    bodies_per_env: int,
    device: torch.device,
) -> _DirectedContactRows:
    pairs = num_envs * contacts_per_env
    pair_world = torch.arange(num_envs, device=device).repeat_interleave(contacts_per_env)
    contact_index = torch.arange(pairs, device=device)
    body0 = pair_world * bodies_per_env + contact_index.remainder(bodies_per_env)
    body1 = pair_world * bodies_per_env + (contact_index + 1).remainder(bodies_per_env)
    gap = -(contact_index.remainder(17).to(torch.float32) + 1.0) * 1.0e-3
    zeros = torch.zeros(pairs, 3, device=device)
    normal = torch.zeros_like(zeros)
    normal[:, 2] = 1.0
    return _DirectedContactRows(
        valid=torch.ones(2 * pairs, dtype=torch.bool, device=device),
        world_id=torch.cat((pair_world, pair_world)),
        body_id=torch.cat((body0, body1)),
        other_body_id=torch.cat((body1, body0)),
        other_dynamic=torch.ones(2 * pairs, dtype=torch.bool, device=device),
        point_world=torch.cat((zeros, zeros)),
        normal_world=torch.cat((normal, -normal)),
        lever_world=torch.cat((zeros, zeros)),
        gap=torch.cat((gap, gap)),
        velocity_world=torch.cat((zeros, zeros)),
    )


def _measure(
    encoder: ContactSetEncoder,
    rows: _DirectedContactRows,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        encoder._pack_rows(rows)
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        encoder._pack_rows(rows)
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def _measure_downstream(
    contact_tokens: torch.Tensor,
    history_length: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    tokens = contact_tokens.unsqueeze(1).expand(-1, history_length, -1, -1)
    root_body_q = torch.zeros(contact_tokens.shape[0], history_length, 7, device=device)
    root_body_q[..., 6] = 1.0

    def run() -> None:
        transformed = transform_contact_tokens_to_body_frame(tokens, root_body_q)
        mask_invalid_contact_tokens(transformed)

    for _ in range(warmup):
        run()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        run()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    """Run the contact-packing scaling sweep."""
    args = _parse_args()
    device = torch.device(args.device)
    results = []
    for num_envs in args.num_envs:
        for contacts_per_env in args.contacts_per_env:
            encoder = _build_encoder(num_envs, args.bodies_per_env, args.capacity, device)
            rows = _build_rows(num_envs, contacts_per_env, args.bodies_per_env, device)
            milliseconds = _measure(encoder, rows, args.warmup, args.iterations, device)
            packed, _, _ = encoder._pack_rows(rows)
            downstream_ms = _measure_downstream(
                packed,
                args.history_length,
                args.warmup,
                args.iterations,
                device,
            )
            result = {
                "device": str(device),
                "num_envs": num_envs,
                "contacts_per_env": contacts_per_env,
                "capacity": args.capacity,
                "history_length": args.history_length,
                "pack_milliseconds": milliseconds,
                "downstream_milliseconds": downstream_ms,
                "total_milliseconds": milliseconds + downstream_ms,
                "microseconds_per_env": milliseconds * 1000.0 / num_envs,
                "contacts_per_second": num_envs * contacts_per_env * 1000.0 / milliseconds,
            }
            results.append(result)
            print(json.dumps(result))

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as output:
            json.dump(results, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
