# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import sys

import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_SOLVER_ACTIVE_FIELD
from isaaclab_neural.eval import run_masked_self_collision_eval as masked_eval
from isaaclab_neural.eval.run_masked_self_collision_eval import (
    annotate_masked_result,
    mask_solver_active_self_collisions,
)


def test_mask_only_valid_solver_active_dynamic_known_body_tokens() -> None:
    tokens = torch.zeros(1, 1, 5, 17)
    tokens[..., 0] = 1.0
    solver_active = torch.tensor([[[True, True, True, False, True]]])
    tokens[0, 0, 0, 2:4] = torch.tensor([3.0, 1.0])
    tokens[0, 0, 1, 2:4] = torch.tensor([-1.0, 1.0])
    tokens[0, 0, 2, 2:4] = torch.tensor([4.0, 0.0])
    tokens[0, 0, 3, 2:4] = torch.tensor([5.0, 1.0])
    tokens[0, 0, 4, 0] = 0.0
    tokens[0, 0, 4, 2:4] = torch.tensor([6.0, 1.0])
    batch = {
        "contact_tokens": tokens,
        CONTACT_TOKEN_SOLVER_ACTIVE_FIELD: solver_active,
    }

    masked, self_collision = mask_solver_active_self_collisions(batch)

    assert self_collision.tolist() == [[[True, False, False, False, False]]]
    assert masked[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD].tolist() == [[[False, True, True, False, True]]]
    assert batch[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD] is solver_active
    assert bool(batch[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD][0, 0, 0])


def test_mask_rejects_missing_or_malformed_solver_flags() -> None:
    tokens = torch.zeros(2, 3, 4, 17)
    with pytest.raises(ValueError, match="requires contact tokens"):
        mask_solver_active_self_collisions({"contact_tokens": tokens})
    with pytest.raises(ValueError, match="boolean"):
        mask_solver_active_self_collisions(
            {
                "contact_tokens": tokens,
                CONTACT_TOKEN_SOLVER_ACTIVE_FIELD: torch.ones(2, 3, 4),
            }
        )


def test_annotate_masked_result_preserves_base_metadata(tmp_path) -> None:
    result_path = tmp_path / "metrics.json"
    result_path.write_text(json.dumps({"schema_version": 1, "design": "c_active"}))

    assert annotate_masked_result(tmp_path) == result_path

    metadata = json.loads(result_path.read_text())
    assert metadata["schema_version"] == 1
    assert metadata["source_design"] == "c_active"
    assert metadata["design"] == "c_active_mask_self"
    assert metadata["intervention"]["training_changed"] is False
    assert metadata["intervention"]["stage"].startswith("raw_batch")


def test_main_applies_mask_during_base_eval_and_restores_trainer_method(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab_neural.train import SequenceModelTrainer

    observed = {}

    def passthrough(_trainer, data):
        observed["flags"] = data[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD].clone()
        return data

    monkeypatch.setattr(SequenceModelTrainer, "preprocess_transferred_data_batch", passthrough)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def fake_run_module(module_name, *, run_name):
        assert module_name == "isaaclab_neural.eval.run_paired_checkpoint_eval"
        assert run_name == "__main__"
        tokens = torch.zeros(1, 1, 2, 17)
        tokens[..., 0] = 1.0
        tokens[0, 0, 0, 2:4] = torch.tensor([2.0, 1.0])
        data = {
            "contact_tokens": tokens,
            CONTACT_TOKEN_SOLVER_ACTIVE_FIELD: torch.ones(1, 1, 2, dtype=torch.bool),
        }
        SequenceModelTrainer.preprocess_transferred_data_batch(object(), data)
        (output_dir / "metrics.json").write_text(json.dumps({"design": "c_active"}))
        return {}

    monkeypatch.setattr(masked_eval.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["masked-eval", "--design", "c_active", "--output-dir", str(output_dir)],
    )

    masked_eval.main()

    assert observed["flags"].tolist() == [[[False, True]]]
    assert SequenceModelTrainer.preprocess_transferred_data_batch is passthrough
    assert json.loads((output_dir / "metrics.json").read_text())["design"] == "c_active_mask_self"
