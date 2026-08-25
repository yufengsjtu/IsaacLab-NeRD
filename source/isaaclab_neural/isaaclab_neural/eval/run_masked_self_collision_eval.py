# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run paired C-active evaluation after masking robot self-collision tokens."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import torch
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX,
    CONTACT_TOKEN_OTHER_DYNAMIC_INDEX,
    CONTACT_TOKEN_SOLVER_ACTIVE_FIELD,
    CONTACT_TOKEN_VALID_INDEX,
)


def mask_solver_active_self_collisions(
    data: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return a shallow batch copy whose self-collision solver flags are false."""
    if "contact_tokens" not in data or CONTACT_TOKEN_SOLVER_ACTIVE_FIELD not in data:
        raise ValueError("Self-collision masking requires contact tokens and solver-active flags.")
    tokens = data["contact_tokens"]
    solver_active = data[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD]
    if tokens.shape[-1] != CONTACT_TOKEN_DIM:
        raise ValueError(f"Contact tokens must end in {CONTACT_TOKEN_DIM} channels.")
    if solver_active.dtype != torch.bool or solver_active.shape != tokens.shape[:-1]:
        raise ValueError("Solver-active flags must be boolean and match the contact-token prefix shape.")

    self_collision = (
        solver_active
        & (tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5)
        & (tokens[..., CONTACT_TOKEN_OTHER_DYNAMIC_INDEX] > 0.5)
        & (tokens[..., CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX] >= 0)
    )
    masked = dict(data)
    masked[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD] = solver_active & ~self_collision
    return masked, self_collision


def annotate_masked_result(output_dir: str | Path) -> Path:
    """Label base evaluator output as the C self-collision intervention."""
    result_path = Path(output_dir).expanduser().resolve() / "metrics.json"
    metadata = json.loads(result_path.read_text())
    if metadata.get("design") != "c_active":
        raise ValueError("Masked self-collision evaluation must wrap the c_active design.")
    metadata["source_design"] = "c_active"
    metadata["design"] = "c_active_mask_self"
    metadata["intervention"] = {
        "name": "mask_solver_active_robot_self_collisions",
        "stage": "raw_batch_before_solver_active_filter_and_normalization",
        "selection": ("valid and solver_active and other_is_dynamic and other_body_slot >= 0"),
        "training_changed": False,
    }
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(result_path)
    return result_path


def _argument_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
        return sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{flag} is required.") from exc


def main() -> None:
    """Patch only the standalone evaluator's raw-batch preprocessing and run it."""
    output_dir = _argument_value("--output-dir")
    if _argument_value("--design") != "c_active":
        raise ValueError("Self-collision masking supports only --design c_active.")

    from isaaclab_neural.train import SequenceModelTrainer

    original = SequenceModelTrainer.preprocess_transferred_data_batch

    @torch.no_grad()
    def preprocess_without_self_collisions(
        trainer: SequenceModelTrainer,
        data: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        masked, _self_collision = mask_solver_active_self_collisions(data)
        return original(trainer, masked)

    SequenceModelTrainer.preprocess_transferred_data_batch = preprocess_without_self_collisions
    try:
        runpy.run_module(
            "isaaclab_neural.eval.run_paired_checkpoint_eval",
            run_name="__main__",
        )
    finally:
        SequenceModelTrainer.preprocess_transferred_data_batch = original
    result_path = annotate_masked_result(output_dir)
    print(f"[masked-self-eval] annotated {result_path}")


if __name__ == "__main__":
    main()
