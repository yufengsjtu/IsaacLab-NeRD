# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for NeRD eval video configuration."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml
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
def test_alias_vis_is_accepted_as_viz(allow_abbrev, monkeypatch) -> None:
    from isaaclab_neural.eval.eval import parse_args

    parser = argparse.ArgumentParser(allow_abbrev=allow_abbrev)
    monkeypatch.setattr(argparse, "ArgumentParser", lambda *args, **kwargs: parser)
    monkeypatch.setattr(sys, "argv", ["eval", "--vis", "newton"])

    args, remaining = parse_args()

    assert remaining == []
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


@pytest.fixture
def policy_playback(monkeypatch, tmp_path):
    import gymnasium as gym
    from rsl_rl import runners

    import isaaclab.app

    monkeypatch.setattr(sys, "argv", ["play"])
    from isaaclab_neural.rl.rsl_rl import play

    checkpoint = tmp_path / "model.pt"
    captured = {}
    env = SimpleNamespace(get_observations=lambda: None, close=lambda: None)
    env.unwrapped = SimpleNamespace(device="cpu", step_dt=0.02)

    def make_env(task, **kwargs):
        captured["env_cfg"] = kwargs["cfg"]
        return env

    def wrap_env(env, *, clip_actions):
        captured["clip_actions"] = clip_actions
        return env

    def make_runner(env, cfg, **kwargs):
        captured["runner_cfg"] = cfg
        return SimpleNamespace(
            load=lambda path, **kwargs: captured.update(checkpoint=path),
            get_inference_policy=lambda **kwargs: lambda obs: obs,
        )

    monkeypatch.setattr(isaaclab.app, "launch_simulation", lambda *args: nullcontext())
    monkeypatch.setattr(play, "newton_material_binding_api_autofix", nullcontext)
    monkeypatch.setattr(play, "retrieve_file_path", lambda path: path)
    monkeypatch.setattr(play.cli_args, "apply_neural_model_to_env_cfg", lambda *args: None)
    monkeypatch.setattr(play.cli_args, "build_launch_cfg", lambda cfg: cfg)
    monkeypatch.setattr(gym, "make", make_env)
    monkeypatch.setattr(play, "RslRlVecEnvWrapper", wrap_env)
    monkeypatch.setattr(runners, "OnPolicyRunner", make_runner)

    def run(agent_cfg, extra_args=()):
        from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg

        monkeypatch.setattr(
            sys,
            "argv",
            ["play", "--checkpoint", str(checkpoint), "--device", "cpu", "--num_steps", "0", *extra_args],
        )
        args, _ = play.parse_args()
        monkeypatch.setattr(play, "args_cli", args)
        play.main.__wrapped__(NerdG1FlatEnvCfg(), agent_cfg)
        return captured

    return checkpoint, run, captured


@pytest.mark.parametrize("saved_config, explicit_overrides", ((False, False), (True, False), (True, True)))
def test_policy_playback_restores_training_configuration(policy_playback, saved_config, explicit_overrides):
    from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import G1FlatNeRDPPORunnerCfg

    from isaaclab.utils.dict import class_to_dict

    checkpoint, run, _ = policy_playback
    fallback = G1FlatNeRDPPORunnerCfg()
    expected = class_to_dict(fallback)
    if saved_config:
        expected["actor"]["hidden_dims"] = [64, 32]
        expected["clip_actions"] = 1.0
        expected["seed"] = 7
        expected["device"] = "cuda:7"
        params = checkpoint.parent / "params"
        params.mkdir()
        (params / "agent.yaml").write_text(yaml.safe_dump(expected), encoding="utf-8")
    extra_args = ["--clip-actions", "2.0", "--seed", "11"] if explicit_overrides else []

    captured = run(fallback, extra_args)

    assert captured["clip_actions"] == (2.0 if explicit_overrides else expected["clip_actions"])
    assert captured["runner_cfg"]["actor"]["hidden_dims"] == expected["actor"]["hidden_dims"]
    assert captured["runner_cfg"]["device"] == "cpu"
    assert captured["runner_cfg"]["seed"] == (11 if explicit_overrides else expected["seed"])
    assert captured["env_cfg"].seed == captured["runner_cfg"]["seed"]
    assert captured["checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("contents", ("- not-a-mapping\n", "actor: [\n"))
def test_policy_playback_rejects_invalid_saved_configuration(policy_playback, contents):
    from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import G1FlatNeRDPPORunnerCfg

    checkpoint, run, captured = policy_playback
    params = checkpoint.parent / "params"
    params.mkdir()
    (params / "agent.yaml").write_text(contents, encoding="utf-8")

    with pytest.raises((ValueError, yaml.YAMLError)):
        run(G1FlatNeRDPPORunnerCfg())

    assert "env_cfg" not in captured
