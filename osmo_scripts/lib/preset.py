#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preset loading helpers for OSMO submit and runtime scripts."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

try:
    from .simple_yaml import load_yaml
except ImportError:
    from simple_yaml import load_yaml


PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


def load_preset(name: str, preset_file: str | None = None) -> dict:
    path = Path(preset_file) if preset_file else PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Preset {name!r} not found at {path}")

    preset = load_yaml(path)
    if not isinstance(preset, dict):
        raise ValueError(f"Preset file {path} did not contain a mapping.")
    return preset


def _value(default, override: str | None):
    return default if override in (None, "") else override


def _print_shell_assignment(name: str, value):
    print(f"{name}={shlex.quote(str(value))}")


def resolve_submit(args: argparse.Namespace):
    preset = load_preset(args.preset, args.preset_file)
    workflow = preset.get("workflow", {})
    resources = preset.get("resources", {})

    workflow_base_name = _value(workflow.get("base_name", args.preset), args.workflow_name)
    dataset_subdir = _value(workflow.get("dataset_subdir", workflow_base_name), args.dataset_subdir)
    workflow_name = f"{workflow_base_name}-{args.run_id}"

    _print_shell_assignment("WORKFLOW_BASE_NAME", workflow_base_name)
    _print_shell_assignment("WORKFLOW_NAME", workflow_name)
    _print_shell_assignment("DATASET_SUBDIR", dataset_subdir)
    _print_shell_assignment("OSMO_NUM_GPU", _value(resources.get("num_gpu", 1), args.num_gpu))
    _print_shell_assignment("OSMO_NUM_CPU", _value(resources.get("num_cpu", 8), args.num_cpu))
    _print_shell_assignment("OSMO_MEMORY", _value(resources.get("memory", "32Gi"), args.memory))
    _print_shell_assignment("OSMO_STORAGE", _value(resources.get("storage", "64Gi"), args.storage))
    _print_shell_assignment("OSMO_PLATFORM", _value(resources.get("platform", "ovx-l40s"), args.platform))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-submit", help="Resolve submit-time preset defaults.")
    resolve_parser.add_argument("--preset", required=True)
    resolve_parser.add_argument("--preset-file")
    resolve_parser.add_argument("--run-id", required=True)
    resolve_parser.add_argument("--workflow-name")
    resolve_parser.add_argument("--dataset-subdir")
    resolve_parser.add_argument("--num-gpu")
    resolve_parser.add_argument("--num-cpu")
    resolve_parser.add_argument("--memory")
    resolve_parser.add_argument("--storage")
    resolve_parser.add_argument("--platform")
    resolve_parser.set_defaults(func=resolve_submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
