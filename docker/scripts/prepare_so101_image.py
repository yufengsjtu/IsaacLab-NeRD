# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export exact dependency closures from validated environments into a minimal Docker context."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def dependency_closure(roots: list[str], overrides: list[str]) -> dict[str, str]:
    """Resolve installed distribution metadata without installing packages or reading credentials."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    allowed = {canonicalize_name(item.name): item for item in map(Requirement, overrides)}
    pending = list(map(Requirement, roots))
    seen: dict[str, set[str]] = {}
    versions = {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        distribution = metadata.distribution(name)
        version = distribution.version
        if requirement.specifier and version not in requirement.specifier:
            if name not in allowed or version not in allowed[name].specifier:
                raise ValueError(f"Installed {name}=={version} does not satisfy {requirement}.")
        extras = set(requirement.extras)
        if name in seen and extras <= seen[name]:
            continue
        seen.setdefault(name, set()).update(extras)
        versions[name] = version
        for text in distribution.requires or []:
            dependency = Requirement(text)
            if dependency.url:
                raise ValueError(f"Direct dependency URLs require an explicit build recipe: {name}.")
            if dependency.marker is None or any(
                dependency.marker.evaluate({**default_environment(), "extra": extra}) for extra in ("", *seen[name])
            ):
                pending.append(dependency)
    return dict(sorted(versions.items()))


def export_context(lab_python: Path, sdk_python: Path, output: Path) -> None:
    """Write pinned requirements and a Dockerfile into a fresh, credential-free build directory."""
    sdk_requirements = Path(__file__).resolve().parents[2] / "osmo_scripts/nvdataset_requirements.in"
    roots = {
        "lab": [
            "torch==2.12.0+cu130",
            "torchvision==0.27.0+cu130",
            "torchaudio==2.11.0+cu130",
            "newton[sim]==1.6.0",
            "mujoco==3.12.0",
            "mujoco-warp==3.12.0",
            "warp-lang==1.17.0",
            "newton-usd-schemas==0.4.1",
            "rsl-rl-lib==5.5.1",
        ],
        "sdk": sdk_requirements.read_text().splitlines(),
    }
    overrides = [
        "torch==2.12.0",
        "torchvision==0.27.0",
        "newton==1.6.0",
        "mujoco==3.12.0",
        "mujoco-warp==3.12.0",
        "newton-usd-schemas==0.4.1",
    ]
    resolved = {}
    for name, python in (("lab", lab_python), ("sdk", sdk_python)):
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "inspect",
            "--roots",
            json.dumps(roots[name]),
            "--overrides",
            json.dumps(overrides if name == "lab" else []),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        resolved[name] = json.loads(result.stdout)
    output.mkdir(parents=True, exist_ok=False)
    for name, packages in resolved.items():
        (output / (name + ".txt")).write_text(
            "".join(f"{package}=={version}\n" for package, version in packages.items())
        )
    shutil.copy2(Path(__file__).resolve().parents[1] / "Dockerfile.so101", output / "Dockerfile")
    (output / "provenance.json").write_text(
        json.dumps({"roots": roots, "lab_overrides": overrides, "resolved": resolved}, indent=2) + "\n"
    )
    print(f"Prepared {output}: {len(resolved['lab'])} lab packages, {len(resolved['sdk'])} isolated SDK packages.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--roots", required=True)
    inspect.add_argument("--overrides", default="[]")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--lab-python", type=Path, required=True)
    prepare.add_argument("--sdk-python", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use the validated Python 3.12 environments for this image.")
    if args.command == "inspect":
        print(json.dumps(dependency_closure(json.loads(args.roots), json.loads(args.overrides))))
    else:
        export_context(args.lab_python, args.sdk_python, args.output)


if __name__ == "__main__":
    main()
