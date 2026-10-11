# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The analytic baseline changes physics without silently reverting the NeRD MDP."""

from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg
from isaaclab_neural.rl.rsl_rl.matched_mj import configure_matched_backend, mdp_contract


def test_g1_worker_task_selection_keeps_flat_default_and_selects_rough_pair():
    from isaaclab_neural.generate.opd import g1_worker_tasks

    assert g1_worker_tasks("flat", "nerd", True) == ("Isaac-Velocity-Flat-G1-NeRD-v0", "Isaac-Velocity-Flat-G1-NeRD-v0")
    assert g1_worker_tasks("rough", "mjwarp", True) == ("Isaac-Velocity-Rough-G1-NeRD-v0", "Isaac-Velocity-Rough-G1")
    assert g1_worker_tasks("rough", "nerd", True) == (
        "Isaac-Velocity-Rough-G1-NeRD-v0",
        "Isaac-Velocity-Rough-G1-NeRD-v0",
    )


def test_mbpo_branch_scene_restores_tile_metadata_and_refreshes_height_scan():
    from types import SimpleNamespace

    import pytest
    import torch
    from isaaclab_neural.rl.rsl_rl.mbpo import capture_branch_scene, restore_branch_scene

    context = {"seed": 123, "mesh_sha256": "same", "origins_sha256": "origins"}
    calls = []
    terrain = SimpleNamespace(
        env_origins=torch.tensor([[1.0, 2.0, 0.3], [4.0, 5.0, 0.7]]),
        terrain_levels=torch.tensor([0, 1]),
        terrain_types=torch.tensor([2, 3]),
    )

    class Scene(dict):
        pass

    scene = Scene(
        height_scanner=SimpleNamespace(
            reset=lambda ids: calls.append(("reset", ids.tolist())),
            update=lambda dt, force_recompute: calls.append((dt, force_recompute)),
        )
    )
    scene.terrain, scene.env_origins = terrain, terrain.env_origins.clone()
    base = SimpleNamespace(scene=scene, mbpo_terrain_context=context)
    snapshot = capture_branch_scene(base, torch.tensor([1]))
    restore_branch_scene(base, torch.tensor([0]), snapshot)
    torch.testing.assert_close(scene.env_origins[0], torch.tensor([4.0, 5.0, 0.7]))
    assert terrain.terrain_levels.tolist() == [1, 1] and terrain.terrain_types.tolist() == [3, 3]
    assert calls == [("reset", [0]), (0.0, True)]
    with pytest.raises(ValueError, match="terrain"):
        restore_branch_scene(base, torch.tensor([0]), {**snapshot, "context": {**context, "seed": 124}})


def test_mbpo_physics_context_restores_selected_worlds_and_keeps_isolated_model_private():
    from types import SimpleNamespace

    import pytest
    import torch
    import warp as wp
    from isaaclab_neural.data.physics_context import WORLD_PHYSICS_FIELDS, WorldPhysicsContext

    def make_model(worlds):
        body_world = torch.tensor([-1, *torch.arange(worlds).repeat_interleave(2).tolist()], dtype=torch.int32)
        shape_world = torch.tensor([-1, *torch.arange(worlds).repeat_interleave(3).tolist()], dtype=torch.int32)
        model = SimpleNamespace(body_world=wp.from_torch(body_world), shape_world=wp.from_torch(shape_world))
        for name in WORLD_PHYSICS_FIELDS:
            count = len(body_world) if name.startswith("body_") else len(shape_world)
            value = torch.ones(count)
            if "inertia" in name:
                value = torch.eye(3).repeat(count, 1, 1)
            elif name == "body_com":
                value = torch.zeros(count, 3)
            setattr(model, name, wp.from_torch(value))
        return model

    source, target = make_model(2), make_model(3)
    wp.to_torch(source.body_mass)[3:] = 2.0
    wp.to_torch(source.body_inv_mass)[3:] = 0.5
    source_context, target_context = WorldPhysicsContext(source, 2), WorldPhysicsContext(target, 3)
    captured = source_context.capture(torch.tensor([1, 0]))
    target_context.restore(torch.tensor([2, 0]), captured)
    torch.testing.assert_close(wp.to_torch(target.body_mass), torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]))
    isolated = WorldPhysicsContext.isolated(target, 3)
    changed = {key: value.clone() for key, value in captured.items()}
    changed["body_mass"][:] = 3.0
    changed["body_inv_mass"][:] = 1.0 / 3.0
    isolated.restore(torch.tensor([0, 1]), changed)
    assert wp.to_torch(target.body_mass)[1].item() == 1.0
    assert wp.to_torch(isolated.model.body_mass)[1].item() == 3.0
    with pytest.raises(ValueError, match="fields"):
        target_context.restore(torch.tensor([0, 1]), {"body_mass": captured["body_mass"]})


def test_mbpo_online_window_sampling_keeps_holdout_worlds_out_of_training(tmp_path):
    import numpy as np
    import torch
    from isaaclab_neural.rl.rsl_rl.mbpo_model import ModelData, select_windows

    folder = tmp_path / "model_data"
    folder.mkdir()
    count, history = 4, 10
    windows = {
        "states": torch.zeros(count, history, 87),
        "env_ids": torch.arange(count),
        "native_counts": torch.ones(count * history, dtype=torch.long),
        "native": {"token": torch.arange(count * history)},
    }
    torch.save(windows, folder / "data_00000.pt")
    data = ModelData(tmp_path, holdout_from=2, max_cached_files=1)
    training = data.sample(0, 100, np.random.default_rng(1), holdout=False)
    holdout = data.sample(0, 100, np.random.default_rng(1), holdout=True)
    assert training == [(0, 0), (0, 1)] and holdout == [(0, 2), (0, 3)]
    stratified = ModelData(tmp_path, holdout_ids=[1, 3])
    assert stratified.sample(0, 100, np.random.default_rng(1), holdout=False) == [(0, 0), (0, 2)]
    assert stratified.sample(0, 100, np.random.default_rng(1), holdout=True) == [(0, 1), (0, 3)]
    selected = select_windows(windows, torch.tensor([3, 1]))
    torch.testing.assert_close(selected["native"]["token"], torch.cat((torch.arange(30, 40), torch.arange(10, 20))))


def test_mbpo_sampling_matches_flat_file_budget_and_recent_holdout_window(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from isaaclab_neural.rl.rsl_rl.mbpo_model import ModelData

    data = ModelData(tmp_path, holdout_ids=[1], max_sample_files=512, recent_eval_iterations=50)
    monkeypatch.setattr(data, "iterations", lambda: list(range(600)))
    monkeypatch.setattr(data, "load", lambda iteration: {"env_ids": torch.tensor([0, 1])})
    training = data.sample(599, 10000, np.random.default_rng(1), holdout=False)
    heldout = data.sample(599, 10000, np.random.default_rng(1), holdout=True)
    assert len(training) == len({iteration for iteration, _ in training}) == 512
    assert all(row == 0 for _, row in training)
    assert heldout == [(iteration, 1) for iteration in range(550, 600)]


def test_mbpo_push_probe_reads_generalized_velocity_after_all_position_coordinates():
    from types import SimpleNamespace

    import pytest
    import torch
    from isaaclab_neural.rl.rsl_rl.mbpo_worker import _verify_stock_push

    state = torch.zeros(1, 87)

    def acquire(source, target):
        target.copy_(state)

    def push(mode, dt):
        state[0, 44] += 0.25

    solver = SimpleNamespace(states=state.clone(), dof_q_per_env=44, _acquire_states_to_torch=acquire)
    base = SimpleNamespace(
        cfg=SimpleNamespace(mbpo_randomized_physics=True),
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        sim=SimpleNamespace(physics_manager=SimpleNamespace(_state_0=None)),
        event_manager=SimpleNamespace(
            _mode_term_names={"interval": ["push_robot"]}, _interval_term_time_left=[torch.ones(1)], apply=push
        ),
    )
    result = _verify_stock_push(base, SimpleNamespace(solver=solver))
    assert result["root_velocity_change_mps"] == pytest.approx(0.25)
    assert not state[0, 7:44].any()


def test_mbpo_event_clocks_restore_at_the_correct_control_boundary():
    from types import SimpleNamespace

    import torch
    from isaaclab_neural.rl.rsl_rl.mbpo import capture_manager_state, restore_manager_state

    clock = torch.tensor([3.0, 7.0])
    command = SimpleNamespace(time_left=torch.tensor([2.0, 6.0]))
    base = SimpleNamespace(
        cfg=SimpleNamespace(mbpo_randomized_physics=True),
        num_envs=2,
        step_dt=0.02,
        episode_length_buf=torch.zeros(2),
        action_manager=SimpleNamespace(_terms={}),
        command_manager=SimpleNamespace(_terms={"velocity": command}),
        reward_manager=SimpleNamespace(_term_names=[], _term_cfgs=[]),
        termination_manager=SimpleNamespace(_term_names=[], _term_cfgs=[]),
        event_manager=SimpleNamespace(
            _mode_term_names={"interval": ["push_robot"]},
            _mode_term_cfgs={"interval": [SimpleNamespace(is_global_time=False)]},
            _interval_term_time_left=[clock],
        ),
    )
    snapshot = capture_manager_state(base, torch.tensor([1]))
    restore_manager_state(base, snapshot, torch.tensor([0]), torch.tensor([0]))
    assert clock[0].item() == 7.0 and command.time_left[0].item() == 6.0
    base._mbpo_inside_step = True
    restore_manager_state(base, snapshot, torch.tensor([0]), torch.tensor([0]))
    clock[0] -= base.step_dt
    command.time_left[0] -= base.step_dt
    assert clock[0].item() == 7.0 and command.time_left[0].item() == 6.0


def test_mbpo_bootstrap_keeps_collecting_until_first_model_without_spending_ppo_updates():
    from isaaclab_neural.rl.rsl_rl.mbpo_loop import bootstrap_action

    config = {"warmup_iterations": 32, "max_warmup_iterations": 128, "require_model_before_ppo": True}
    assert bootstrap_action(config, 31, 0) == "collect"
    assert bootstrap_action(config, 32, -1) == "collect"
    assert bootstrap_action(config, 127, -1) == "collect"
    assert bootstrap_action(config, 128, -1) == "wait_model"
    assert bootstrap_action(config, 32, 0) == "ppo"
    assert bootstrap_action(config, 128, 0) == "ppo"


def test_mbpo_collector_stops_at_actual_adaptive_training_boundary(tmp_path):
    from isaaclab_neural.rl.rsl_rl.mbpo_loop import atomic_json, wait_for_policy

    (tmp_path / "policies").mkdir()
    policy = tmp_path / "policies" / "policy_00004.pt"
    policy.write_bytes(b"ready")
    atomic_json({"iterations": 5, "ppo_updates": 2}, tmp_path / "TRAINING_DONE.json")
    assert wait_for_policy(tmp_path, 4, 1) == policy
    assert wait_for_policy(tmp_path, 5, 1) is None


def test_matched_configuration_keeps_mdp_and_collision_contract():
    cfg = NerdG1FlatEnvCfg()
    before = mdp_contract(cfg)
    margin = cfg.sim.physics.default_shape_cfg.margin
    audit = configure_matched_backend(cfg, "mjwarp")
    assert mdp_contract(cfg) == before
    assert audit["mdp_unchanged"]
    assert cfg.sim.physics.solver_cfg.solver_type == "mujoco_warp"
    assert cfg.sim.physics.default_shape_cfg.margin == margin
    assert cfg.scene.contact_forces is None
    assert cfg.rewards.feet_air_time.func.__name__ == "GeometryFeetAirTimePositiveBiped"
    assert cfg.terminations.base_contact.func.__name__ == "geometry_illegal_contact"


def test_mdp_fingerprint_ignores_batch_size_but_detects_reward_changes():
    cfg = NerdG1FlatEnvCfg()
    first = mdp_contract(cfg)
    cfg.scene.num_envs = 128
    assert mdp_contract(cfg) == first
    cfg.rewards.track_ang_vel_z_exp.weight += 1.0
    assert mdp_contract(cfg) != first


def test_extended_policy_study_has_explicit_budget_and_cannot_warm_start_training():
    import pytest
    from isaaclab_neural.rl.rsl_rl.matched_mj import validate_policy_study

    config = {
        "mode": "train",
        "num_envs": 4096,
        "updates": 1500,
        "initial_learning_rate": 1e-3,
        "physics_scope": "matched_geometry_nominal",
        "init_at_random_ep_len": True,
    }
    validate_policy_study(config, num_envs=4096, policy_updates=0)
    for changed in ({"updates": 1501}, {"num_envs": 8192}, {"physics_scope": "stock"}, {"mode": "unknown"}):
        with pytest.raises(ValueError):
            validate_policy_study({**config, **changed}, num_envs=4096, policy_updates=0)
    with pytest.raises(ValueError):
        validate_policy_study(config, num_envs=4096, policy_updates=1)
    validate_policy_study(
        {**config, "mode": "eval", "updates": 0, "eval_steps": 1000}, num_envs=4096, policy_updates=500
    )


def test_policy_evaluation_counts_actual_controls_before_all_episodes_finish(monkeypatch):
    from types import SimpleNamespace

    import torch
    from isaaclab_neural.eval import policy_benchmark
    from isaaclab_neural.rl.rsl_rl.matched_mj import _evaluate_study_policy

    terminated = torch.zeros(2, dtype=torch.bool)
    base = SimpleNamespace(
        num_envs=2,
        step_dt=0.02,
        device="cpu",
        cfg=SimpleNamespace(decimation=4),
        command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(2, 3)),
        scene={"robot": SimpleNamespace(data=None)},
        termination_manager=SimpleNamespace(terminated=terminated, time_outs=torch.zeros_like(terminated)),
    )
    calls = []

    class Policy:
        def __call__(self, obs, stochastic_output):
            assert not stochastic_output
            return torch.zeros(2, 1)

        def reset(self, done):
            pass

    def step(action):
        calls.append(action)
        terminated[: len(calls)] = True
        return {}, torch.ones(2), terminated.clone(), {}

    monkeypatch.setattr(policy_benchmark, "task_tracking_state", lambda data: (torch.zeros(2, 2), torch.zeros(2)))
    runner = SimpleNamespace(env=SimpleNamespace(step=step), get_inference_policy=lambda device: Policy())
    result = _evaluate_study_policy(base, runner, {}, 1000)
    assert len(calls) == result["control_steps"] == 2
    assert result["physical_transitions"] == len(calls) * base.num_envs * base.cfg.decimation
    assert result["failure_rate"] == 1 and result["right_censored_rate"] == 0


def test_study_feature_cache_is_opt_in_and_audits_identical_inputs():
    from types import SimpleNamespace

    import pytest
    import torch
    from isaaclab_neural.rl.rsl_rl.matched_mj import _configure_study_feature_cache

    def original(inputs):
        return inputs["x"] * 2

    solver = SimpleNamespace(
        anchor_frame_step="every",
        states_frame="body",
        use_kv_cache=False,
        use_feature_cache=False,
        predicted_body_net_contact_forces_b=None,
        _model_forward=original,
        neural_model=lambda inputs, single_step: inputs["x"] * 2,
    )
    assert not _configure_study_feature_cache(solver, False)["enabled"]
    assert solver._model_forward is original and not solver.use_feature_cache
    audit = _configure_study_feature_cache(solver, True, audit_frames=2)
    for _ in range(2):
        torch.testing.assert_close(
            solver._model_forward({"x": torch.ones(2, 1, 3), "optional": None}), torch.full((2, 1, 3), 2.0)
        )
    assert solver.use_feature_cache and solver._model_forward is original
    assert audit["audited_frames"] == 2 and audit["max_abs_error"] == 0
    solver._model_forward = lambda inputs: inputs["x"] * 3
    _configure_study_feature_cache(solver, True, audit_frames=1)
    with pytest.raises(AssertionError):
        solver._model_forward({"x": torch.ones(2, 1, 3), "optional": None})
    for field, value in (
        ("anchor_frame_step", "first"),
        ("use_kv_cache", True),
        ("predicted_body_net_contact_forces_b", torch.ones(1)),
    ):
        changed = SimpleNamespace(**{**vars(solver), field: value})
        with pytest.raises(ValueError):
            _configure_study_feature_cache(changed, True)


def test_study_argument_contract_does_not_expand_legacy_matched_pilot():
    import argparse
    from types import SimpleNamespace

    from isaaclab_neural.generate.opd import _validate_probe_arguments

    parser = argparse.ArgumentParser()
    args = SimpleNamespace(
        stage="study-rl",
        backend="nerd",
        num_envs=4096,
        horizon=1,
        policy_snapshot="initial.pt",
        rl_study_config="study.json",
        sequence_fit_physics_steps=4,
        active_updates=1,
        active_eval_steps=64,
        active_reward_weight=0.2,
        active_error_cap=10.0,
        active_record_envs=32,
        active_capacity=128,
        active_eval_seed=21,
        active_mode="train",
        sequence_deterministic=False,
        sequence_controls=4,
        probe_updates=1,
        probe_samples=32,
        probe_lr=1e-4,
    )
    _validate_probe_arguments(parser, args)
    args.stage = "matched-mj-train"
    args.backend = "mjwarp"
    import pytest

    with pytest.raises(SystemExit):
        _validate_probe_arguments(parser, args)
