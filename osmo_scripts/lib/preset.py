#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preset loading helpers for OSMO submit and runtime scripts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simple_yaml import load_yaml
else:
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
    if preset.get("experiment", {}).get("dataset_profile") == "so101_keyboard":
        if TYPE_CHECKING:
            from .package_code import collect_packaged_files
        else:
            try:
                from .package_code import collect_packaged_files
            except ImportError:
                from package_code import collect_packaged_files

        root = Path(__file__).resolve().parents[2]
        checkpoint = Path(preset["experiment"]["collection"]["checkpoint"])
        checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else (root / checkpoint).resolve()
        if (
            not checkpoint.is_file()
            or not checkpoint.is_relative_to(root)
            or checkpoint.relative_to(root) not in collect_packaged_files(root)
        ):
            raise ValueError("Stage the SO101 policy inside the packaged repository before upload/submission.")

    workflow_base_name = _value(workflow.get("base_name", args.preset), args.workflow_name)
    dataset_subdir = _value(workflow.get("dataset_subdir", workflow_base_name), args.dataset_subdir)
    workflow_name = f"{workflow_base_name}-{args.run_id}"

    _print_shell_assignment("WORKFLOW_BASE_NAME", workflow_base_name)
    _print_shell_assignment("WORKFLOW_NAME", workflow_name)
    _print_shell_assignment("DATASET_SUBDIR", dataset_subdir)
    _print_shell_assignment("OSMO_IMAGE", _value(workflow.get("image", ""), getattr(args, "image", None)))
    _print_shell_assignment(
        "OSMO_NUCLEUS_CREDENTIAL",
        _value(workflow.get("nucleus_credential", ""), getattr(args, "nucleus_credential", None)),
    )
    _print_shell_assignment("OSMO_NUM_GPU", _value(resources.get("num_gpu", 1), args.num_gpu))
    _print_shell_assignment("OSMO_NUM_CPU", _value(resources.get("num_cpu", 8), args.num_cpu))
    _print_shell_assignment("OSMO_MEMORY", _value(resources.get("memory", "32Gi"), args.memory))
    _print_shell_assignment("OSMO_STORAGE", _value(resources.get("storage", "64Gi"), args.storage))
    _print_shell_assignment("OSMO_PLATFORM", _value(resources.get("platform", "ovx-l40s"), args.platform))
    _print_shell_assignment("OSMO_TRAINING_ONLY", int(bool(preset.get("experiment", {}).get("training_only", False))))


def render_workflow(args: argparse.Namespace) -> Path:
    """Render a credential-free local preview without uploads, profile changes or OSMO calls."""
    import yaml

    preset = load_preset(args.preset, args.preset_file)
    workflow, resources = preset["workflow"], preset["resources"]
    template = Path(args.workflow_file).read_text()
    values = yaml.safe_load(template.split("\ndefault-values:", 1)[1])
    base = _value(workflow.get("base_name", args.preset), args.workflow_name)
    values.update(
        workflow_name=f"{base}-{args.run_id}",
        workflow_base_name=base,
        dataset_subdir=_value(workflow.get("dataset_subdir", base), args.dataset_subdir),
        osmo_experiment_preset=args.preset,
        image=_value(workflow.get("image", ""), args.image),
        num_gpu=int(_value(resources.get("num_gpu", 1), args.num_gpu)),
        num_cpu=int(_value(resources.get("num_cpu", 8), args.num_cpu)),
        memory=_value(resources.get("memory", "32Gi"), args.memory),
        storage=_value(resources.get("storage", "64Gi"), args.storage),
        resource_platform=_value(resources.get("platform", "ovx-l40s"), args.platform),
        dataset_cache_mode="require" if preset.get("experiment", {}).get("training_only") else "auto",
        train_cfg_overrides=args.train_cfg_overrides or "",
    )
    if not values["image"] or values["num_gpu"] <= 0 or values["num_cpu"] <= 0:
        raise ValueError("Workflow requires an image and positive resource counts.")
    root = Path(__file__).resolve().parents[1]
    for key, file in (
        ("init_sh_localpath", "init.sh"),
        ("entrypoint_localpath", "entrypoint.sh"),
        ("nvdataset_io_localpath", "lib/nvdataset_io.py"),
        ("swift_io_localpath", "lib/swift_io.py"),
    ):
        values[key] = str(root / file)

    def replace(match):
        name = match.group(1) or match.group(2)
        if name == "workflow_id":
            return json.dumps("{{workflow_id}}")
        if name not in values:
            raise ValueError(f"Unresolved workflow variable: {name}.")
        return json.dumps(str(values[name]) if match.group(1) else values[name])

    rendered = re.sub(r'"{{\s*(\w+)\s*}}"|{{\s*(\w+)\s*}}', replace, template.split("\ndefault-values:", 1)[0])
    parsed = yaml.safe_load(rendered)
    nucleus = _value(workflow.get("nucleus_credential", ""), getattr(args, "nucleus_credential", None))
    if nucleus:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", nucleus):
            raise ValueError("Invalid Nucleus credential name.")
        credentials = parsed["workflow"]["tasks"][0].setdefault("credentials", {})
        if nucleus in credentials:
            raise ValueError("Nucleus and storage credentials must have distinct names.")
        credentials[nucleus] = {"OMNI_USER": "omni_user", "OMNI_PASS": "omni_pass"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        yaml.safe_dump(parsed, stream, sort_keys=False)
    print(f"Rendered local preview: {output}. No upload or submission was performed.")
    return output


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
    resolve_parser.add_argument("--image")
    resolve_parser.add_argument("--nucleus-credential")
    resolve_parser.set_defaults(func=resolve_submit)

    render_parser = subparsers.add_parser("render-workflow", help="Render a local preview without remote operations.")
    render_parser.add_argument("--preset", required=True)
    render_parser.add_argument("--run-id", default="preview")
    render_parser.add_argument("--output", required=True)
    render_parser.add_argument("--workflow-file", default=str(PRESET_DIR.parent / "osmo_workflow.yaml"))
    for name in (
        "preset-file",
        "workflow-name",
        "dataset-subdir",
        "num-gpu",
        "num-cpu",
        "memory",
        "storage",
        "platform",
        "image",
        "train-cfg-overrides",
        "nucleus-credential",
    ):
        render_parser.add_argument("--" + name)
    render_parser.set_defaults(func=render_workflow)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
