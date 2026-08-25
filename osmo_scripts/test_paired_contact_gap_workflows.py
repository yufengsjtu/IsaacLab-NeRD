# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for standalone paired contact-gap OSMO workflows."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "osmo_scripts" / "paired_contact_gap"


def test_workflows_are_standalone_one_gpu_jobs() -> None:
    generation = (WORKFLOW_DIR / "generate_workflow.yaml").read_text()
    evaluation = (WORKFLOW_DIR / "eval_workflow.yaml").read_text()

    for workflow in (generation, evaluation):
        assert "cpu: 12" in workflow
        assert "gpu: 1" in workflow
        assert "memory: 96Gi" in workflow
        assert "storage: 64Gi" in workflow
        assert "platform: ovx-l40" in workflow
        assert "/tmp/paired-eval-entrypoint.sh" in workflow
        assert "osmo_scripts/start.sh" not in workflow
        assert "osmo_scripts/run_experiment.py" not in workflow
    assert generation.count("    - url:") == 1
    assert evaluation.count("    - url:") == 3


def test_entrypoint_freezes_policy_suite_and_fixed_checkpoint_contract() -> None:
    entrypoint = (WORKFLOW_DIR / "entrypoint.sh").read_text()

    required_generation_args = (
        "--task Isaac-Velocity-Rough-Anymal-C-v0",
        "--sample-mode policy",
        "--initial-states-source env",
        "--contact-packing-policy body_round_robin_pair_atomic",
        "--contact-representation raw15_tokens",
        "--max-contact-tokens 64",
        "--num-envs 1024",
        "--num-transitions 1000000",
        "--trajectory-length 400",
        "--write-chunk-transitions 409600",
        "--seed 40",
        "--states-frame body",
        "--prediction-type relative",
    )
    for argument in required_generation_args:
        assert argument in entrypoint
    assert "--randomize-pd-gains" not in entrypoint
    assert "model_1499.pt" in entrypoint
    assert "--expected-seeds 0,1,2" in entrypoint
    assert "--batch-size 512" in entrypoint
    assert "isaaclab_neural.eval.generate_paired_contact_dataset" in entrypoint
    assert "isaaclab_neural.eval.run_paired_checkpoint_eval" in entrypoint
    assert "isaaclab_neural.train.train" not in entrypoint
    assert "run_experiment.py" not in entrypoint
    assert "${#" not in entrypoint
