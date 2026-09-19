# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard Isaac Lab 3 launcher import paths used by NeRD scripts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

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


def test_reach_comparison_preserves_policy_configuration_helpers(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "policy.pt"
    params = tmp_path / "params"
    params.mkdir()
    (params / "agent.yaml").write_text("class_name: OnPolicyRunner\nclip_actions: 0.75\nseed: 1\n")
    monkeypatch.setattr(sys, "argv", ["compare", "--actuator", "implicit", "--policy-checkpoint", str(checkpoint)])
    comparison = importlib.import_module("isaaclab_neural.eval.compare_franka_reach_actuators")
    loaded = comparison.load_saved_agent_cfg_for_policy(str(checkpoint), {}, device="cpu", seed=7)
    runner_cfg, class_name, clip_actions, device = comparison.normalize_runner_cfg(loaded, device="cuda:0")
    assert loaded["device"] == "cpu"
    assert runner_cfg == {"class_name": "OnPolicyRunner", "clip_actions": 0.75, "seed": 7, "device": "cuda:0"}
    assert (class_name, clip_actions, device) == ("OnPolicyRunner", 0.75, "cuda:0")
    calls = []
    dones = torch.tensor([True, False])
    comparison.reset_policy(SimpleNamespace(reset=calls.append), None, dones)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], dones)
    assert callable(comparison.build_policy_env)
