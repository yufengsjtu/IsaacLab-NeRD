# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for NeRD eval video configuration."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from isaaclab_neural.eval.eval import configure_eval_video
from isaaclab_neural.physics import NerdNewtonCfg


def _video_args(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "video": True,
        "video_length": 200,
        "video_interval": 0,
        "video_dir": str(tmp_path),
        "visualizer": None,
        "visualizer_explicit": False,
        "enable_cameras": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_configure_eval_video_is_noop_without_flag(tmp_path) -> None:
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=NerdNewtonCfg(), visualizer_cfgs=[]), video_recorders=[])
    args = _video_args(tmp_path, video=False)

    configure_eval_video(env_cfg, args)

    assert env_cfg.video_recorders == []
    assert env_cfg.sim.visualizer_cfgs == []
    assert args.enable_cameras is False


def test_nerd_video_records_newton_gl_without_kit(tmp_path) -> None:
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=NerdNewtonCfg(), visualizer_cfgs=[], default_visualizer_cfg=None),
        video_recorders=[],
    )
    args = _video_args(tmp_path)

    configure_eval_video(env_cfg, args)

    assert args.enable_cameras is False
    assert args.visualizer == ["newton_gl"]
    assert args.visualizer_explicit is True
    types = [getattr(cfg, "visualizer_type", None) for cfg in env_cfg.sim.visualizer_cfgs]
    assert types == ["newton_gl"]
    assert env_cfg.sim.visualizer_cfgs[0].headless is True
    assert len(env_cfg.video_recorders) == 1
    recorder = env_cfg.video_recorders[0]
    assert recorder.source == "visualizer:newton_gl"
    assert recorder.output_dir == str(tmp_path)
    assert recorder.video_length == 200
    assert recorder.video_interval == 0


def test_nerd_video_opens_window_when_viz_newton_is_explicit(tmp_path) -> None:
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=NerdNewtonCfg(), visualizer_cfgs=[], default_visualizer_cfg=None),
        video_recorders=[],
    )
    args = _video_args(tmp_path, visualizer=["newton"], visualizer_explicit=True)

    configure_eval_video(env_cfg, args)

    assert env_cfg.sim.visualizer_cfgs[0].headless is False
    assert args.visualizer == ["newton_gl"]


@pytest.mark.parametrize("allow_abbrev", [True, False])
def test_alias_vis_is_accepted_as_viz(allow_abbrev) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=allow_abbrev)
    from isaaclab_neural.eval.eval import _alias_vis_as_viz

    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    _alias_vis_as_viz(parser)
    args = parser.parse_args(["--vis", "newton"])
    assert args.visualizer == ["newton_gl"]
    assert args.visualizer_explicit is True


def test_nerd_video_rejects_explicit_viz_none(tmp_path) -> None:
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=NerdNewtonCfg(), visualizer_cfgs=[]), video_recorders=[])
    args = _video_args(tmp_path, visualizer=None, visualizer_explicit=True)

    with pytest.raises(ValueError, match="--viz none"):
        configure_eval_video(env_cfg, args)


def test_policy_evaluation_leaves_mutable_environment_history(monkeypatch) -> None:
    import torch
    from isaaclab_neural.eval import eval as evaluation

    history = []
    policy_modes = []

    def policy(obs):
        policy_modes.append(torch.is_inference_mode_enabled())
        return obs.clone()

    def step(action):
        history.append(action.clone())
        return torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

    env = SimpleNamespace(get_observations=lambda: torch.zeros(1, 1), step=step, close=lambda: None)
    monkeypatch.setattr(evaluation, "load_saved_agent_cfg_for_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation, "build_policy_env", lambda *args: (env, policy, None))
    monkeypatch.setattr(evaluation, "reset_policy", lambda *args: None)
    monkeypatch.setattr(evaluation, "save_reward_plot", lambda *args: None)
    args = argparse.Namespace(policy_checkpoint="unused", device="cpu", seed=0, task="test", num_envs=1, num_steps=2)

    evaluation.run_policy(env, None, args)

    assert policy_modes == [True, True]
    for entry in history:
        entry.index_copy_(0, torch.tensor([0]), torch.ones(1, 1))
        torch.testing.assert_close(entry, torch.ones(1, 1))
