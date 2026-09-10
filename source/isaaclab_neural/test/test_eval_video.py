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


def test_alias_vis_is_accepted_as_viz() -> None:
    parser = argparse.ArgumentParser()
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
