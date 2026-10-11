# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher comparisons do not reward early death or count auto-reset episodes as survivors."""

import torch
from isaaclab_neural.eval.policy_benchmark import EpisodeBenchmark


def test_first_episode_metrics_stop_on_failure_and_keep_timeouts_separate():
    metrics = EpisodeBenchmark(2, 0.5, "cpu")
    command = torch.zeros(2, 3)
    metrics.record(
        torch.ones(2, 2),
        torch.zeros(2),
        command,
        torch.ones(2),
        torch.tensor([True, False]),
        torch.zeros(2, dtype=torch.bool),
    )
    metrics.record(
        torch.full((2, 2), 100.0),
        torch.zeros(2),
        command,
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([False, True]),
    )
    result = metrics.report()
    assert result["failure_rate"] == 0.5
    assert result["timeout_rate"] == 0.5
    assert result["episode_lengths_s"] == [0.5, 1.0]
    assert result["episode_returns"] == [1.0, 2.0]
    assert result["tracking_samples_per_env"] == [0, 1]
    assert result["per_env_xy_error_mps"][0] is None
    assert abs(result["per_env_xy_error_mps"][1] - 2**0.5) < 1e-6


def test_tracking_uses_yaw_frame_not_tilted_body_frame():
    from types import SimpleNamespace

    from isaaclab_neural.eval.policy_benchmark import task_tracking_state

    half = 2**-0.5
    robot = SimpleNamespace(
        root_quat_w=SimpleNamespace(torch=torch.tensor([[half, 0.0, 0.0, half]])),
        root_lin_vel_w=SimpleNamespace(torch=torch.tensor([[0.0, 1.0, 0.0]])),
        root_ang_vel_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.5]])),
    )
    velocity, yaw = task_tracking_state(robot)
    torch.testing.assert_close(velocity, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(yaw, torch.tensor([0.5]))


def test_archive_loader_accepts_data_tags_not_arbitrary_objects():
    import pytest
    import yaml
    from isaaclab_neural.eval.policy_benchmark import ArchiveLoader

    data = yaml.load(
        "x: !!python/tuple [1, 2]\ny: !!python/object/apply:builtins.slice [null, null, null]", Loader=ArchiveLoader
    )
    assert data == {"x": [1, 2], "y": "slice(None, None, None)"}
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load("x: !!python/object/apply:builtins.eval ['1+1']", Loader=ArchiveLoader)
