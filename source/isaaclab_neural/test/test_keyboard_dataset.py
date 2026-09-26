# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for fixed-keyboard collection and reset-safe replay contracts."""

from __future__ import annotations

from unittest.mock import create_autospec

import h5py
import pytest
import torch
from isaaclab_neural.data.keyboard import (
    KEYBOARD_SCHEMA,
    keyboard_clip_indices,
    keyboard_replay_batches,
    read_keyboard_metadata,
    write_keyboard_dataset,
)
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.generate.keyboard_dataset import configure_fixed_keyboard
from native_hdf5_samples import make_native_rollouts

from isaaclab_tasks.contrib.keyboard.so101_env_cfg import SO101KeyboardEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_backend_exposes_current_mjwarp_buffers_and_rejects_other_solvers(monkeypatch):
    from types import SimpleNamespace

    import isaaclab_newton.physics as physics
    from isaaclab_neural.generate.adapter import NewtonDataGenerationBackend

    data = SimpleNamespace()
    manager = SimpleNamespace(
        backend=SimpleNamespace(model=object(), state_0=object(), control=object()),
        _solver=SimpleNamespace(mjw_data=data),
    )
    monkeypatch.setattr(physics, "NewtonManager", manager)
    backend = NewtonDataGenerationBackend("cpu")
    assert backend.mjwarp_data is data
    replacement = SimpleNamespace()
    manager._solver = SimpleNamespace(mjw_data=replacement)
    assert backend.mjwarp_data is replacement
    manager._solver = SimpleNamespace()
    with pytest.raises(RuntimeError, match="MJWarp replay requires"):
        _ = backend.mjwarp_data
    manager.backend = None
    with pytest.raises(RuntimeError, match="not initialized"):
        NewtonDataGenerationBackend("cpu")


@pytest.mark.parametrize("empty", [False, True])
def test_indexed_keyboard_contacts_match_reference_clip_order(empty):
    from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_FRAME_COUNTS, NATIVE_CONTACT_VALUE_FIELDS
    from isaaclab_neural.generate.keyboard_dataset import pack_keyboard_contact_clips
    from isaaclab_neural.generate.sampler import build_native_contact_rollout

    frames = []
    for step in range(4):
        ids = torch.tensor([] if empty else [2, 0, 2, -1, 1, 0], dtype=torch.int32)
        frame = {
            name: torch.arange(ids.numel() * 2).reshape(-1, 2).float() + step * 100
            for name in NATIVE_CONTACT_VALUE_FIELDS
        }
        frame["world_ids"] = ids
        frames.append(frame)
    selected, horizon, worlds = [(2, 0), (0, 2), (2, 1), (3, 0), (1, 1)], 2, 4
    batches = [
        build_native_contact_rollout(
            frames[start : start + horizon], torch.arange(worlds) == world, torch.arange(worlds)
        )
        for world, start in selected
    ]
    expected = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    actual = pack_keyboard_contact_clips(frames, selected, horizon=horizon, num_worlds=worlds)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    assert actual[NATIVE_CONTACT_FRAME_COUNTS].shape == (len(selected), horizon)


def test_hessian_record_bound_covers_strided_triangular_writes():
    import numpy as np
    from isaaclab_neural.generate.keyboard_determinism import hessian_record_bound

    widths = np.array([[1, 7, 3, -999], [5, 2, 0, -999]])
    indices = np.array([[2, 0, 1, 999], [1, 0, 999, 999]])
    counts = np.array([3, 2])
    for stride in (1, 2, 4):
        for lanes in (1, 4, 32):
            actual = max(
                sum(
                    len(
                        range(
                            lane,
                            int(widths[world, indices[world, group]])
                            * (int(widths[world, indices[world, group]]) + 1)
                            // 2,
                            lanes,
                        )
                    )
                    for group in range(slot, int(count), stride)
                )
                for world, count in enumerate(counts)
                for slot in range(stride)
                for lane in range(lanes)
            )
            assert hessian_record_bound(widths, indices, counts, stride, lanes, 8) >= actual
    assert hessian_record_bound(widths, indices, np.zeros(2, dtype=int), 1, 32, 8) == 1
    with pytest.raises(ValueError):
        hessian_record_bound(widths, indices, np.array([4, 2]), 1, 32, 8)
    with pytest.raises(ValueError):
        hessian_record_bound(widths, indices, counts, 1, 32, 6)
    with pytest.raises(ValueError):
        hessian_record_bound(widths, indices, np.array([5, 2]), 1, 32, 8)


def test_workspace_scope_restores_launcher_and_preserves_compiled_metadata(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from isaaclab_neural.generate import keyboard_determinism as workspace
    from warp._src import context, deterministic

    original = Mock()
    monkeypatch.setattr(deterministic, "launch_deterministic", original)
    monkeypatch.setattr(context, "runtime", SimpleNamespace(captures={}))
    monkeypatch.setattr(workspace, "_record_bound", lambda *args: 3)
    metadata = deterministic.DeterministicMeta(max_records=100)
    with pytest.raises(RuntimeError, match="test failure"):
        with workspace.keyboard_deterministic_workspace(True) as contract:
            assert contract == workspace.workspace_contract()
            deterministic.launch_deterministic(*([None] * 8), metadata, [])
            assert original.call_args.args[8].max_records == 3
            assert metadata.max_records == 100
            with pytest.raises(RuntimeError, match="overlap"):
                with workspace.keyboard_deterministic_workspace(True):
                    pass
            raise RuntimeError("test failure")
    assert deterministic.launch_deterministic is original
    with workspace.keyboard_deterministic_workspace(True):
        pass


def test_workspace_rejects_unaudited_sources(monkeypatch):
    from isaaclab_neural.generate import keyboard_determinism as workspace

    monkeypatch.setattr(workspace, "_SOURCE_HASHES", {})
    with workspace.keyboard_deterministic_workspace(False) as contract:
        assert contract is None
    with pytest.raises(RuntimeError, match="Unsupported"):
        with workspace.keyboard_deterministic_workspace(True):
            pass


def test_derived_model_restore_flushes_updates_and_preserves_ongoing_worlds(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import warp as wp
    from isaaclab_neural.generate import keyboard_dataset as module

    arrays = {"mjwarp_model_" + name.replace(".", "_"): torch.zeros(2, 3) for name in module.MJW_MODEL_FIELDS}
    context = {name: torch.full_like(value, 2.0) for name, value in arrays.items()}
    solver = SimpleNamespace(
        notify_model_changed=Mock(side_effect=lambda flag: [value.fill_(-1) for value in arrays.values()]),
        _invalidate_contact_fast_path=Mock(),
    )
    manager = SimpleNamespace(_solver=solver, _model_changes={1})
    adapter = SimpleNamespace(backend=SimpleNamespace(_manager=manager))
    env = SimpleNamespace(device="cpu", num_envs=2)
    monkeypatch.setattr(module, "_mjwarp_model_arrays", lambda _: arrays)
    monkeypatch.setattr(wp, "to_torch", lambda value: value)
    module._restore_mjwarp_model_context(env, adapter, context, None)
    assert not manager._model_changes
    assert all(torch.equal(value, context[name]) for name, value in arrays.items())
    solver.notify_model_changed.assert_called_once_with(1)
    changed = {name: torch.full_like(value, 9.0) for name, value in context.items()}
    module._restore_mjwarp_model_context(env, adapter, changed, torch.tensor([1]))
    assert all(torch.equal(value[0], context[name][0]) for name, value in arrays.items())
    assert all(torch.equal(value[1], changed[name][1]) for name, value in arrays.items())
    assert solver._invalidate_contact_fast_path.call_count == 2
    with pytest.raises(ValueError, match="Incomplete"):
        module._restore_mjwarp_model_context(env, adapter, {next(iter(context)): torch.zeros(2, 3)}, None)


def test_fixed_keyboard_preserves_task_and_does_not_mutate_source():
    original = resolve_presets(SO101KeyboardEnvCfg(), {"newton_mjwarp"})
    fixed = configure_fixed_keyboard(original, variant=3)
    assert len(original.scene.keyboard.spawn.assets_cfg) == 32
    assert len(fixed.scene.keyboard.spawn.assets_cfg) == 1
    assert fixed.scene.keyboard.spawn.assets_cfg[0].seed == 3
    for term in (fixed.events.reset_keyboard, fixed.commands.typing.reset.pre_solve_reset):
        assert all(tuple(value) == (0.0, 0.0) for value in term.params["pose_range"].values())
    assert original.events.reset_keyboard.params["pose_range"]["z"] == [0.015, 0.05]
    assert original.commands.typing.reset.pre_solve_reset.params["pose_range"]["roll"] == [0.0, 0.75]
    for name in ("actions", "observations", "rewards", "terminations", "sim"):
        assert getattr(fixed, name).to_dict() == getattr(original, name).to_dict()
    assert fixed.decimation == original.decimation == 4
    assert fixed.commands.typing.reset.buffer_size == 8192
    assert fixed.commands.typing.reset.enabled
    assert fixed.commands.typing.typeable_slots == original.commands.typing.typeable_slots
    with pytest.raises(ValueError, match="variant"):
        configure_fixed_keyboard(original, variant=32)


def test_collection_scene_is_fixed_and_replay_rejects_other_variants():
    from argparse import Namespace

    from isaaclab_neural.generate.keyboard_dataset import _resolve_scene

    args = Namespace(num_envs=32, keyboard_variant=None, seed=42, device="cpu", deterministic_physics=None)
    cfg, _, variant, play_mode = _resolve_scene(args, None)
    assert variant == 0 and play_mode
    assert [item.seed for item in cfg.scene.keyboard.spawn.assets_cfg] == [0]
    for term in (cfg.events.reset_keyboard, cfg.commands.typing.reset.pre_solve_reset):
        assert all(tuple(value) == (0.0, 0.0) for value in term.params["pose_range"].values())
    assert cfg.scene.num_envs == 32 and cfg.seed == 42 and cfg.commands.typing.reset.enabled
    assert not cfg.sim.physics.deterministic
    args.deterministic_physics = True
    deterministic_cfg, *_ = _resolve_scene(args, None)
    assert deterministic_cfg.sim.physics.deterministic and not deterministic_cfg.sim.physics.use_cuda_graph
    saved = {"source_num_envs": 32, "source_seed": 42, "keyboard_variant": 0, "deterministic": True}
    args.keyboard_variant = 1
    with pytest.raises(ValueError, match="variant"):
        _resolve_scene(args, saved)


def test_replay_video_preserves_physics_and_records_complete_control_timeline(tmp_path):
    from argparse import Namespace

    from isaaclab_neural.data.keyboard import KEYBOARD_REPLAY_LAYOUT
    from isaaclab_neural.generate.keyboard_dataset import _configure_replay_video, _resolve_scene

    args = Namespace(
        num_envs=8,
        keyboard_variant=None,
        seed=42,
        device="cpu",
        deterministic_physics=True,
        mode="replay",
        dataset=tmp_path / "data.hdf5",
        video_dir=tmp_path / "video",
        video_env_index=3,
    )
    cfg, *_ = _resolve_scene(args, None)
    before = cfg.sim.physics.to_dict()
    reset = cfg.commands.typing.reset.to_dict()
    with h5py.File(args.dataset, "w") as file:
        file.create_dataset("replay/frames/states", shape=(8, 560, 228), dtype="f4")
        origins = file.create_dataset("replay/contexts/env_origin", shape=(8, 140, 3), dtype="f4")
        origins[3, 0] = (2.0, -3.0, 0.5)
    _configure_replay_video(cfg, args, {"replay_layout": KEYBOARD_REPLAY_LAYOUT})
    assert cfg.sim.physics.to_dict() == before
    assert cfg.commands.typing.reset.to_dict() == reset
    assert cfg.scene.num_envs == 8
    assert len(cfg.scene.keyboard.spawn.assets_cfg) == 1
    viewer = cfg.sim.visualizer_cfgs[0]
    assert viewer.headless and viewer.visible_env_indices == [3]
    for key in ("eye", "lookat"):
        expected = tuple(x + y for x, y in zip(getattr(cfg.sim.default_visualizer_cfg, key), (2, -3, 0.5)))
        assert getattr(viewer, key) == pytest.approx(expected)
    assert not viewer.randomly_sample_visible_envs
    assert cfg.video_recorders[0].video_length == 560 // cfg.decimation
    assert cfg.video_recorders[0].fps is None
    assert args.visualizer == ["newton_gl"] and args.visualizer_explicit
    args.video_dir.mkdir()
    with pytest.raises(FileExistsError):
        _configure_replay_video(cfg, args, {"replay_layout": KEYBOARD_REPLAY_LAYOUT})


def test_replay_video_requires_replay_and_valid_world(tmp_path):
    from argparse import Namespace
    from types import SimpleNamespace

    from isaaclab_neural.data.keyboard import KEYBOARD_REPLAY_LAYOUT
    from isaaclab_neural.generate.keyboard_dataset import _configure_replay_video

    args = Namespace(video_dir=None)
    _configure_replay_video(None, args, None)
    args = Namespace(video_dir=tmp_path / "video", mode="collect", video_env_index=0)
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=8))
    with pytest.raises(ValueError, match="synchronous replay"):
        _configure_replay_video(cfg, args, None)
    args.mode = "replay"
    args.video_env_index = 8
    with pytest.raises(ValueError, match="video-env-index"):
        _configure_replay_video(cfg, args, {"replay_layout": KEYBOARD_REPLAY_LAYOUT})


@pytest.mark.parametrize("failure", [False, True])
def test_resident_requests_reuse_env_and_reset_policy_with_independent_seeds(tmp_path, monkeypatch, failure):
    import io
    import json
    from argparse import Namespace
    from types import SimpleNamespace

    import isaaclab_neural.generate.keyboard_dataset as module

    status = tmp_path / "worker" / "status.json"
    requests = [
        {
            "sequence": i,
            "seed": seed,
            "dataset": str(tmp_path / f"part{i}.hdf5"),
            "report": str(tmp_path / f"part{i}.json"),
            "response": str(status.parent / f"response{i}.json"),
        }
        for i, seed in enumerate((42, 43))
    ]
    resets, collected, policy_resets = [], [], []
    env = SimpleNamespace(
        num_envs=2, device="cpu", cfg=SimpleNamespace(seed=42), reset=lambda seed: resets.append(seed)
    )
    args = Namespace(worker_status=status, seed=42, dataset=tmp_path / "first.hdf5", report=tmp_path / "first.json")
    runner, policy = object(), object()
    monkeypatch.setattr(module.sys, "stdin", io.StringIO("\n".join(json.dumps(r) for r in requests)))
    monkeypatch.setattr(module, "_reset_policy", lambda r, done: policy_resets.append((r, done.tolist())))

    def collect(actual_env, adapter, request, metadata, agent, wrapped, actual_runner, actual_policy):
        assert actual_env is env and actual_runner is runner and actual_policy is policy
        if failure and request.seed == 43:
            raise RuntimeError("collection failed")
        collected.append((request.seed, metadata["source_seed"], metadata["resident_worker"]["sequence"]))

    monkeypatch.setattr(module, "collect_keyboard_chunk", collect)
    if failure:
        with pytest.raises(RuntimeError, match="collection failed"):
            module.serve_collection_requests(env, None, args, {}, None, None, runner, policy)
    else:
        module.serve_collection_requests(env, None, args, {}, None, None, runner, policy)
    assert resets == [42, 43]
    assert policy_resets == [(runner, [True, True]), (runner, [True, True])]
    summary = json.loads(status.read_text())
    assert summary["environment_creations"] == summary["policy_loads"] == 1
    assert summary["chunks_completed"] == (1 if failure else 2)
    assert json.loads((status.parent / "response1.json").read_text())["ok"] is (not failure)
    assert collected == ([(42, 42, 0)] if failure else [(42, 42, 0), (43, 43, 1)])


def test_clips_never_cross_resets_and_keep_terminal_frame():
    done = torch.zeros((2, 16), dtype=torch.bool)
    done[0, 3] = True
    done[1, 7] = True
    assert keyboard_clip_indices(done, horizon=8, decimation=4) == [(0, 4), (1, 0), (1, 8)]
    with pytest.raises(ValueError, match="multiple"):
        keyboard_clip_indices(done, horizon=6, decimation=4)
    done[0, 1] = True
    with pytest.raises(ValueError, match="control"):
        keyboard_clip_indices(done, horizon=8, decimation=4)


def test_terminal_tail_is_retained_even_when_it_overlaps():
    done = torch.zeros((1, 12), dtype=torch.bool)
    done[0, 11] = True
    assert keyboard_clip_indices(done, horizon=8, decimation=4) == [(0, 0), (0, 4)]


@pytest.mark.parametrize("fail", [False, True])
def test_recorder_captures_before_reset_and_restores_scene_hook(monkeypatch, fail):
    from types import SimpleNamespace

    import isaaclab_neural.generate.keyboard_dataset as module

    scene = SimpleNamespace(value=0)

    def original(**kwargs):
        return None

    scene.update = original
    command = SimpleNamespace(typed=torch.zeros(1, 5, dtype=torch.long))
    env = SimpleNamespace(
        _physics_handles_decimation=False,
        _sim_step_counter=0,
        num_envs=1,
        cfg=SimpleNamespace(decimation=4),
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    recorder = module.KeyboardFrameRecorder(env, create_autospec(DataGenerationAdapter, instance=True))
    monkeypatch.setattr(module, "_context", lambda *args: {})
    monkeypatch.setattr(recorder, "_snapshot", lambda: {})
    monkeypatch.setattr(recorder, "_after_frame", lambda action: recorder.frames.append({"value": scene.value}))

    def step(action):
        for index in range(4):
            env._sim_step_counter += 1
            scene.value = index + 1
            scene.update(dt=0.01)
            if fail:
                raise RuntimeError("step failed")
        scene.value = -999
        scene.update(dt=0.0)
        return {}, torch.zeros(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

    env.step = step
    if fail:
        with pytest.raises(RuntimeError, match="step failed"):
            recorder.step(torch.zeros(1, 6))
    else:
        recorder.step(torch.zeros(1, 6))
        assert [frame["value"] for frame in recorder.frames] == [1, 2, 3, 4]
        assert bool(recorder.frames[-1]["terminated"].all())
    assert scene.update is original


def test_replay_batches_preserve_world_slots_and_exclude_fillers():
    worlds = torch.tensor([0, 0, 1, 1])
    batches = keyboard_replay_batches(worlds, [0, 1, 2], 2)
    assert batches == [([0, 2], [0, 1]), ([1, 2], [0])]
    for indices, active in batches:
        assert worlds[indices].tolist() == [0, 1]
    with pytest.raises(ValueError, match="source world"):
        keyboard_replay_batches(worlds, [0], 1)
    with pytest.raises(ValueError, match="unique"):
        keyboard_replay_batches(worlds, [0, 0], 2)


@pytest.mark.parametrize("fail", [False, True])
def test_force_labels_precede_contact_refresh_and_pd_overlay_is_restored(monkeypatch, fail):
    from types import SimpleNamespace

    import isaaclab_neural.generate.keyboard_dataset as module
    import warp as wp

    class Scene(dict[str, SimpleNamespace]):
        sensors: dict[str, SimpleNamespace]

        def update(self, dt: float) -> None:
            pass

    force = torch.ones(1, 1, 7, 3)
    control = torch.ones(1, 114)
    scene = Scene(
        keyboard=SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros(1, 108)),
                joint_pos_limits=SimpleNamespace(torch=torch.tensor([-0.002, 0.0]).expand(1, 108, 2)),
            )
        )
    )
    scene.sensors = {
        "robot_contact": SimpleNamespace(data=SimpleNamespace(net_normal_forces_w_history=SimpleNamespace(torch=force)))
    }
    command = SimpleNamespace(
        _inst_idx=torch.zeros(1, 108, dtype=torch.long),
        _col_idx=torch.arange(108).unsqueeze(0),
        cfg=SimpleNamespace(actuation_fraction=0.5),
        typed=torch.zeros(1, 5, dtype=torch.long),
    )
    env = SimpleNamespace(
        _physics_handles_decimation=False,
        _sim_step_counter=0,
        cfg=SimpleNamespace(decimation=1),
        scene=scene,
        num_envs=1,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )

    def sync(*, overlay_applied_effort=False):
        if overlay_applied_effort:
            control.fill_(7.0)
            force.fill_(9.0)

    adapter = create_autospec(DataGenerationAdapter, instance=True)
    adapter.control = SimpleNamespace(joint_f=control)
    adapter.sync.side_effect = sync
    adapter.raw_neural_inputs.return_value = {"joint_f": control.unsqueeze(1)}
    recorder = module.KeyboardFrameRecorder(env, adapter)
    monkeypatch.setattr(module, "_context", lambda *args: {})
    monkeypatch.setattr(wp, "to_torch", lambda value: value)
    snapshot_count = 0

    def snapshot():
        nonlocal snapshot_count
        snapshot_count += 1
        if fail and snapshot_count == 2:
            raise RuntimeError("snapshot failed")
        return {"states": torch.full((1, 228), float(snapshot_count - 1))}

    def step(action):
        env._sim_step_counter += 1
        scene.update(dt=0.01)
        return {}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

    env.step = step
    original_update = scene.update
    monkeypatch.setattr(recorder, "_snapshot", snapshot)
    if fail:
        with pytest.raises(RuntimeError, match="snapshot failed"):
            recorder.step(torch.zeros(1, 6))
    else:
        recorder.step(torch.zeros(1, 6))
        assert torch.equal(recorder.frames[0]["robot_net_normal_forces_w"], torch.ones(1, 7, 3))
        assert torch.equal(recorder.frames[0]["joint_f"], torch.full((1, 114), 7.0))
    assert torch.equal(control, torch.ones(1, 114))
    assert scene.update == original_update


def sample():
    rollout = make_native_rollouts(num_trajectories=1, num_steps=8)
    rollout.update(
        states=torch.zeros(1, 8, 228),
        next_states=torch.zeros(1, 8, 228),
        joint_f=torch.zeros(1, 8, 114),
        actions=torch.zeros(1, 8, 6),
        robot_net_normal_forces_w=torch.zeros(1, 8, 7, 3),
        key_pressed=torch.zeros(1, 8, 108, dtype=torch.bool),
        terminated=torch.zeros(1, 8, dtype=torch.bool),
        truncated=torch.zeros(1, 8, dtype=torch.bool),
    )
    metadata = {
        "schema": KEYBOARD_SCHEMA,
        "task": "IsaacContrib-Keyboard-SO101",
        "keyboard_variant": 0,
        "physics_dt": 0.01,
        "decimation": 4,
        "force_frame": "world",
        "force_units": "N",
        "force_timing": "post_solve_before_reset",
        "force_body_names": [f"body_{i}" for i in range(7)],
        "num_q": 114,
        "num_qd": 114,
        "key_q_indices": list(range(6, 114)),
    }
    return rollout, metadata


@pytest.mark.parametrize("fault", [None, "nonfinite", "reset", "overflow", "replay_claim", "variant"])
def test_basic_keyboard_validation_keeps_data_checks_without_replay(tmp_path, monkeypatch, fault):
    import json
    from pathlib import Path

    from isaaclab_neural.contacts.keyboard_normal_force import BODY_NAMES

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "osmo_scripts"))
    from lib import keyboard_data

    rollout, metadata = sample()
    metadata.update(
        source_num_envs=1,
        source_seed=42,
        deterministic=False,
        collection_mode="policy",
        force_body_names=list(BODY_NAMES),
    )
    rollout["trajectory_context"].update(
        source_env_id=torch.zeros(1, dtype=torch.long), source_start_frame=torch.zeros(1, dtype=torch.long)
    )
    if fault == "variant":
        metadata["keyboard_variant"] = 1
    path = tmp_path / "part.hdf5"
    write_keyboard_dataset(path, rollout, metadata)
    with h5py.File(path, "a") as file:
        file.create_dataset("replay/frames/states", data=rollout["states"].numpy())
        if fault == "nonfinite":
            file["data/next_states"][0, 0, 0] = float("nan")
        elif fault == "reset":
            file["data/terminated"][0, 3] = True
    report = {
        "mode": "collect",
        "basic_validation_passed": True,
        "trajectory_count": 1,
        "total_recorded_frames_per_env": 8,
        "native_contact_packing": {
            "dropped_contacts": 0,
            "truncated_frames": 0,
            "overflow_tokens_total": int(fault == "overflow"),
            "overflow_frames": 0,
        },
    }
    receipt = tmp_path / "part.hdf5.collect.json"
    receipt.write_text(json.dumps(report))
    records = {
        path.name: {
            "validation_mode": "basic",
            "replay_performed": fault == "replay_claim",
            "sha256": keyboard_data.file_sha256(path),
            "collect_sha256": keyboard_data.file_sha256(receipt),
        }
    }
    exp = {
        "require_keyboard_replay": False,
        "collection": {"validation_mode": "basic", "deterministic_physics": False},
        "data_gen_num_envs": 1,
        "trajectory_length": 8,
        "dataset_specs": [{"filename": path.name, "split": "train", "seed": 42}],
    }
    if fault:
        with pytest.raises(ValueError):
            keyboard_data.validate_recordings(tmp_path, exp, records, complete=False)
    else:
        assert keyboard_data.validate_recordings(tmp_path, exp, records, complete=False) == records
        assert not (tmp_path / "part.hdf5.replay.json").exists()
    (tmp_path / keyboard_data.MANIFEST).write_text(json.dumps({"schema": keyboard_data.BASIC_SCHEMA}))
    with pytest.raises(ValueError, match="validation mode"):
        keyboard_data.validate_bundle(tmp_path, {"require_keyboard_replay": True, "dataset_specs": []})


def test_synchronous_timeline_roundtrip_and_clip_alignment(tmp_path):
    from isaaclab_neural.contacts.keyboard_normal_force import BODY_NAMES

    rollout, metadata = sample()
    metadata.update(source_num_envs=1, replay_layout="synchronous_world_timeline_v1", force_body_names=list(BODY_NAMES))
    rollout["trajectory_context"].update(
        source_env_id=torch.zeros(1, dtype=torch.long), source_start_frame=torch.zeros(1, dtype=torch.long)
    )
    rollout["control_boundary"] = torch.tensor([[False, False, False, True] * 2])
    rollout["typed_after"] = torch.zeros(1, 8, 5, dtype=torch.long)
    frames = {name: value.clone() for name, value in rollout.items() if isinstance(value, torch.Tensor)}
    contexts = {
        "source_env_id": torch.zeros(1, 2, dtype=torch.long),
        "reset_state": torch.zeros(1, 2, 254),
        "episode_length": torch.zeros(1, 2, dtype=torch.long),
        "last_action": torch.zeros(1, 2, 6),
        "previous_action": torch.zeros(1, 2, 6),
        "env_origin": torch.zeros(1, 2, 3),
    }
    timeline = {"frames": frames, "contexts": contexts}
    path = tmp_path / "timeline.hdf5"
    write_keyboard_dataset(path, rollout, metadata, replay=timeline)
    with h5py.File(path, "r") as file:
        assert file["replay/frames/states"].shape == (1, 8, 228)
        assert file["replay/contexts/reset_state"].shape == (1, 2, 254)
    from isaaclab_neural.data import TrajectoryDataset

    dataset = TrajectoryDataset(path, sample_sequence_length=4)
    assert dataset[0]["states"].shape == (4, 228)
    frames["actions"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="timeline"):
        write_keyboard_dataset(tmp_path / "misaligned.hdf5", rollout, metadata, replay=timeline)
    with pytest.raises(ValueError, match="timeline"):
        write_keyboard_dataset(tmp_path / "missing.hdf5", rollout, metadata)


@pytest.mark.parametrize("record_video", [False, True])
def test_synchronous_replay_restores_only_matching_terminal_rows(monkeypatch, record_video):
    from types import SimpleNamespace

    import isaaclab_neural.generate.keyboard_dataset as module

    base, metadata = sample()
    expected = {
        name: value[:, :4].repeat(2, 3, *([1] * (value.ndim - 2)))
        for name, value in base.items()
        if isinstance(value, torch.Tensor)
    }
    expected["terminated"][1, 3] = True
    expected["terminated"][0, 7] = True
    expected["terminated"][1, 11] = True
    expected["control_boundary"] = torch.tensor([[False, False, False, True] * 3] * 2)
    expected["typed_after"] = torch.zeros(2, 12, 5, dtype=torch.long)
    expected["primary_body_q"] = torch.zeros(2, 12, 7, 7)
    actual = {name: value.clone() for name, value in expected.items()}
    actual["terminated"][:, 3] = torch.tensor([True, False])
    restored = []

    def restore(env, adapter, context, env_ids=None):
        restored.append(None if env_ids is None else env_ids.tolist())

    class Recorder:
        def __init__(self, env, adapter):
            self.control = 0
            self.env = env

        def step(self, action):
            frame = self.control * 4 + 3
            self.control += 1
            for video in self.env.video_recorders:
                video.step()
            return None, None, actual["terminated"][:, frame], actual["truncated"][:, frame], {}

        def tensors(self):
            return actual

    monkeypatch.setattr(module, "restore_context", restore)
    monkeypatch.setattr(module, "KeyboardFrameRecorder", Recorder)
    captures, renders = [], []
    videos = [SimpleNamespace(step=lambda: captures.append(list(restored)))] if record_video else []
    env = SimpleNamespace(
        device="cpu",
        cfg=SimpleNamespace(decimation=4),
        video_recorders=videos,
        sim=SimpleNamespace(render=lambda: renders.append(True)),
    )
    report = module.replay_timeline(env, None, expected, {"source_env_id": torch.zeros(2, 3)}, metadata)
    assert restored == [None, [0]]
    assert report["reset_rows_restored"] == 1
    assert report["termination_mismatches"] == 2
    assert env.video_recorders is videos
    assert captures == ([[None], [None, [0]], [None, [0]]] if record_video else [])
    assert len(renders) == (3 if record_video else 0)


def test_training_contract_requires_one_fixed_keyboard_and_independent_seeds():
    from copy import deepcopy

    from isaaclab_neural.data.keyboard import validate_keyboard_training_contracts

    _, first = sample()
    first.update(source_num_envs=1, source_seed=42, physics_parameters={"body_mass": [1.0]})
    valid = deepcopy(first)
    valid["source_seed"] = 43
    validate_keyboard_training_contracts({"a": first}, {"valid": valid})
    first["collection_seeds"] = [42, 1042]
    valid["collection_seeds"] = [43, 1042]
    with pytest.raises(ValueError, match="independently"):
        validate_keyboard_training_contracts({"a": first}, {"valid": valid})
    valid.pop("collection_seeds")
    with pytest.raises(ValueError, match="independently"):
        validate_keyboard_training_contracts({"a": first}, {"leaked": first})
    for key, value in (("keyboard_variant", 1), ("physics_parameters", {"body_mass": [2.0]})):
        changed = deepcopy(valid)
        changed[key] = value
        with pytest.raises(ValueError, match="matching variants"):
            validate_keyboard_training_contracts({"a": first}, {"valid": changed})
    valid["physics_dt"] = 0.02
    with pytest.raises(ValueError, match="physics_dt"):
        validate_keyboard_training_contracts({"a": first}, {"valid": valid})
    original = deepcopy(first)
    original["schema"] = "so101_keyboard_original_v1"
    with pytest.raises(ValueError, match="schema"):
        validate_keyboard_training_contracts({"a": original}, {"valid": valid})


def test_keyboard_hdf5_roundtrip_and_exclusive_output(tmp_path):
    path = tmp_path / "keyboard.hdf5"
    rollout, metadata = sample()
    write_keyboard_dataset(path, rollout, metadata)
    assert read_keyboard_metadata(path) == metadata
    with h5py.File(path, "r") as file:
        states = file["data/states"]
        forces = file["data/robot_net_normal_forces_w"]
        assert isinstance(states, h5py.Dataset)
        assert isinstance(forces, h5py.Dataset)
        assert states.shape == (1, 8, 228)
        assert forces.shape == (1, 8, 7, 3)
        assert "contacts/native_pairs/frame_offsets" in file
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_keyboard_dataset(path, rollout, metadata)
    assert path.read_bytes() == original


@pytest.mark.parametrize("encoded", [7, [1, 2], "[]", "null"])
def test_keyboard_reader_rejects_non_object_metadata(tmp_path, encoded):
    path = tmp_path / "invalid_metadata.hdf5"
    with h5py.File(path, "w") as file:
        file.create_group("data").attrs["keyboard_contract"] = encoded
    with pytest.raises(ValueError, match="JSON"):
        read_keyboard_metadata(path)


def test_keyboard_reader_requires_data_group(tmp_path):
    path = tmp_path / "invalid_group.hdf5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=[0]).attrs["keyboard_contract"] = "{}"
    with pytest.raises(ValueError, match="no keyboard replay contract"):
        read_keyboard_metadata(path)


@pytest.mark.parametrize("failure", ["force", "nan", "reset", "state"])
def test_keyboard_writer_rejects_invalid_contract(tmp_path, failure):
    rollout, metadata = sample()
    if failure == "force":
        rollout["robot_net_normal_forces_w"] = torch.zeros(1, 8, 2, 3)
    elif failure == "nan":
        rollout["next_states"][0, 0, 0] = torch.nan
    elif failure == "reset":
        rollout["terminated"][0, 3] = True
    else:
        metadata["num_q"] = 113
    path = tmp_path / "invalid.hdf5"
    with pytest.raises(ValueError):
        write_keyboard_dataset(path, rollout, metadata)
    assert not path.exists()
