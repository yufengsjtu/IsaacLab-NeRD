# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard Isaac Lab 3 launcher import paths used by NeRD scripts."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NEURAL_SRC = PACKAGE_ROOT / "isaaclab_neural"


def test_launch_simulation_is_imported_from_isaaclab_app() -> None:
    stale: list[str] = []
    for path in sorted(NEURAL_SRC.rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "launch_simulation" in stripped and "isaaclab_tasks.utils" in stripped:
                stale.append(f"{path.relative_to(PACKAGE_ROOT)}: {stripped}")
                break
    assert stale == [], "launch_simulation must be imported from isaaclab.app:\n" + "\n".join(stale)
