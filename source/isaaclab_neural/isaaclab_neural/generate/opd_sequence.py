# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only PPO sequence recording and reusable MJWarp action-replay oracle."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import warp as wp

from isaaclab_neural.data.opd import CONTEXT_FIELDS, file_sha256
from isaaclab_neural.data.opd_sequence import SequenceWindowBuffer, read_sequence_shard, write_sequence_shard
from isaaclab_neural.generate.opd import ContextRecorder, simulation_protocol
from isaaclab_neural.physics.applied_joint_f import backend_applied_effort


def _require_stateless_pd(env) -> None:
    for cfg in env.scene["robot"].cfg.actuators.values():
        name = cfg.class_type if isinstance(cfg.class_type, str) else cfg.class_type.__name__
        if name.rsplit(":", 1)[-1] != "ImplicitActuator":
            raise ValueError("Sequence v1 only supports stateless G1 ImplicitActuator control.")


class SequenceRecorder(ContextRecorder):
    """Record selected worlds' actual control sequences and pre-reset physical successor states.

    GPU-to-CPU copies gather selected environments per tensor rather than per
    environment. Historical frames are raw SI snapshots, not cached NN outputs.
    """

    def __init__(
        self,
        env,
        adapter,
        *,
        env_ids: list[int],
        history_steps: int = 10,
        sequence_controls: int = 4,
        verify_frames: int = 32,
    ):
        _require_stateless_pd(env)
        if history_steps != 10:
            raise ValueError("The initial sequence recorder preserves the existing H10 runtime history contract.")
        super().__init__(env, adapter, env_ids=env_ids, verify_frames=verify_frames)
        self.selected = torch.tensor(self.ids, device=env.device, dtype=torch.long)
        self.windows = SequenceWindowBuffer(
            self.ids, history_steps=history_steps, sequence_controls=sequence_controls, decimation=env.cfg.decimation
        )
        self.failed_env_ids = []
        self.verified_histories = 0

    def begin_iteration(self, iteration_id: int) -> None:
        self.windows.begin_iteration(iteration_id)
        self.policy_update_id = iteration_id

    def before(self):
        if self.frame % self.env.cfg.decimation == 0:
            self.native_before = {}
            for key in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
                raw = wp.to_torch(getattr(self.manager._state_0, key))
                shaped = raw.reshape(self.env.num_envs, -1, *raw.shape[1:])
                self.native_before[key] = shaped.index_select(0, self.selected).detach().cpu().clone()
        if self.neural is not None:
            return
        self.adapter.sync()
        self.before_data = self._batch_snapshot(self.adapter.solver, self.adapter.native_contact_pairs())
        efforts = backend_applied_effort(self.env.scene["robot"]).index_select(0, self.selected).detach().cpu()
        warmstart = (
            wp.to_torch(self.manager._solver.mjw_data.qacc_warmstart).index_select(0, self.selected).cpu().clone()
        )
        for j, env_id in enumerate(self.ids):
            self.before_data[env_id]["joint_f"][6:] = efforts[j]
            self.before_data[env_id]["solver_state"] = {"qacc_warmstart": warmstart[j]}

    def after(self):
        if self.policy_update_id != self.windows.iteration_id:
            raise RuntimeError("A new PPO iteration requires an explicit sequence-buffer boundary.")
        current = self.before_data
        if self.neural is not None:
            solver = self.neural.solver
            current = self._batch_snapshot(solver, solver.contact_adapter.native_contact_pairs())
        next_states = torch.empty_like(self.adapter.solver.states)
        self.adapter.solver._acquire_states_to_torch(self.manager._state_0, next_states)
        self.adapter.solver.wrap2PI(next_states)
        finite = torch.isfinite(next_states).all(dim=-1)
        if not finite.all():
            self.failed_env_ids = (~finite).nonzero().flatten().cpu().tolist()
            raise FloatingPointError(f"Non-finite output in global worlds {self.failed_env_ids}; no synthetic labels.")
        selected_next = next_states.index_select(0, self.selected).detach().cpu().clone()
        for j, env_id in enumerate(self.ids):
            if self.frame % self.env.cfg.decimation == 0:
                current[env_id]["newton_state"] = {key: value[j] for key, value in self.native_before.items()}
            if any(not torch.isfinite(current[env_id][key]).all() for key in CONTEXT_FIELDS):
                raise FloatingPointError("Non-finite sequence input.")
            self.history[env_id].append(current[env_id])
        if self.neural is not None and self.frame < self.verify_frames:
            actual = self.neural.solver.states_history
            for key in ("states", "joint_f", "root_body_q"):
                expected = torch.stack([entry[key].index_select(0, self.selected) for entry in actual], dim=1).cpu()
                observed = torch.stack([torch.stack([entry[key] for entry in self.history[i]]) for i in self.ids])
                torch.testing.assert_close(observed, expected, rtol=0, atol=1e-5)
            self.verified_histories += 1
        self.windows.append_frame(
            self.frame,
            self.history,
            {i: selected_next[j] for j, i in enumerate(self.ids)},
            {i: int(self.episodes[i]) for i in self.ids},
        )
        self.frame += 1

    def finish_control_step(self, done: torch.Tensor):
        self.windows.finish_control_step(set(done.nonzero().flatten().cpu().tolist()))
        super().finish_control_step(done)

    def end_iteration(self) -> list[dict]:
        return self.windows.end_iteration()


class SequenceOracle:
    """Replay real targets in MJWarp, reusing one isolated environment across requests."""

    def __init__(self, env, adapter):
        _require_stateless_pd(env)
        self.env, self.adapter = env, adapter
        self.protocol = simulation_protocol(env)
        self.requests = 0

    @torch.no_grad()
    def label(self, source: str | Path, target: str | Path, *, preserve_world_ids: bool = False) -> dict:
        shard = read_sequence_shard(source)
        metadata = shard["metadata"]
        if metadata["protocol"] != self.protocol:
            raise ValueError("Oracle/source physics, ordering or actuator protocol mismatch.")
        if (
            metadata["decimation"] != self.env.cfg.decimation
            or abs(metadata["physics_dt"] - self.env.physics_dt) > 1e-10
        ):
            raise ValueError("Oracle/source action timing mismatch.")
        if Path(target).exists():
            raise FileExistsError(target)
        records = copy.deepcopy(shard["records"])
        groups = {}
        for i, row in enumerate(records):
            if preserve_world_ids and row["env_id"] >= self.env.num_envs:
                raise ValueError("World-preserving replay needs all source world slots.")
            key = (len(row["actions"]), row["anchor_frame"] if preserve_world_ids else -1)
            groups.setdefault(key, []).append(i)
        actual_steps = 0
        max_pd_error = 0.0
        for (controls, _), indices in sorted(groups.items()):
            for start in range(0, len(indices), self.env.num_envs):
                selected = indices[start : start + self.env.num_envs]
                destinations = (
                    [records[i]["env_id"] for i in selected] if preserve_world_ids else list(range(len(selected)))
                )
                padded = [selected[-1]] * self.env.num_envs
                for destination, index in zip(destinations, selected):
                    padded[destination] = index
                rows = [records[i] for i in padded]
                state = torch.stack([r["history"][-1]["states"] for r in rows]).to(self.env.device)
                self.env.scene["robot"].actuators.reset()
                self.env.action_manager.reset()
                self.adapter.backend.state = self.env.sim.physics_manager._state_0
                self.adapter.control.joint_f.zero_()
                self.adapter.reset(state)
                if metadata.get("initial_physical_state_protocol") == "native_newton_state_v1":
                    manager = self.env.sim.physics_manager
                    for key in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
                        raw = wp.to_torch(getattr(manager._state_0, key))
                        restored = torch.stack([r["history"][-1]["newton_state"][key] for r in rows]).to(
                            self.env.device
                        )
                        if restored.numel() != raw.numel():
                            raise ValueError(f"Native state layout differs for {key}.")
                        raw.copy_(restored.reshape_as(raw))
                    manager._solver.reset(manager._state_0, flags=0)
                    self.env.scene.update(dt=self.env.physics_dt)
                    self.adapter.sync()
                if metadata.get("initial_solver_state_protocol") == "recorded_mj_qacc_warmstart":
                    warmstart = torch.stack([r["history"][-1]["solver_state"]["qacc_warmstart"] for r in rows]).to(
                        self.env.device
                    )
                    destination = wp.to_torch(self.env.sim.physics_manager._solver.mjw_data.qacc_warmstart)
                    if destination.shape != warmstart.shape:
                        raise ValueError("Oracle/source warm-start layout differs.")
                    destination.copy_(warmstart)
                torch.testing.assert_close(self.adapter.solver.states, state, rtol=0, atol=1e-5)
                for key in ("root_body_q", "root_body_qd"):
                    expected = torch.stack([r["history"][-1][key] for r in rows]).to(self.env.device)
                    torch.testing.assert_close(getattr(self.adapter.solver, key), expected, rtol=0, atol=1e-4)
                trajectory = []
                for control in range(controls):
                    action = torch.stack([r["actions"][control] for r in rows]).to(self.env.device)
                    targets = torch.stack([r["action_targets"][control] for r in rows]).to(self.env.device)
                    self.env.action_manager.process_action(action)
                    torch.testing.assert_close(
                        self.env.action_manager.get_term("joint_pos").processed_actions, targets, rtol=0, atol=1e-6
                    )
                    for substep in range(metadata["decimation"]):
                        self.adapter.step_action_frame(action, process_action=False, overlay_applied_effort=False)
                        if bool(wp.to_torch(self.adapter.control.joint_f).any()):
                            raise RuntimeError("Sequence replay injected feedforward effort into implicit PD.")
                        values = self.adapter.solver.states[destinations].detach().cpu().clone()
                        if not torch.isfinite(values).all():
                            raise FloatingPointError("Non-finite oracle sequence; stopping the worker.")
                        trajectory.append(values)
                        if control == 0 and substep == 0:
                            expected_pd = torch.stack([r["history"][-1]["joint_f"][6:] for r in rows]).to(
                                self.env.device
                            )
                            pd_error = float(
                                (backend_applied_effort(self.env.scene["robot"]) - expected_pd).abs().max()
                            )
                            max_pd_error = max(max_pd_error, pd_error)
                            if pd_error > 1e-3:
                                raise RuntimeError(f"Oracle/source initial PD mismatch: {pd_error}.")
                predictions = torch.stack(trajectory, dim=1)
                for j, i in enumerate(selected):
                    records[i]["oracle_states"] = predictions[j].clone()
                actual_steps += controls * metadata["decimation"] * self.env.num_envs
        self.requests += 1
        write_sequence_shard(
            target,
            records,
            metadata={
                **metadata,
                "label_origin": "mjwarp",
                "source_shard_sha256": file_sha256(source),
                "oracle_execution": {
                    "deterministic_mode": self.env.cfg.sim.physics.deterministic_mode,
                    "num_substeps": self.env.cfg.sim.physics.num_substeps,
                    "disable_sensors": self.env.cfg.sim.physics.solver_cfg.disable_sensors,
                    "preserve_world_ids": preserve_world_ids,
                },
                "oracle_initialization": metadata.get("initial_solver_state_protocol", "canonical_cold_reset"),
            },
        )
        errors = [float((r["oracle_states"] - r["student_states"]).abs().max()) for r in records]
        return {
            "source": str(source),
            "target": str(target),
            "records": len(records),
            "request_number": self.requests,
            "actual_oracle_physics_transitions": actual_steps,
            "valid_label_transitions": sum(len(r["oracle_states"]) for r in records),
            "initial_pd_abs_max": max_pd_error,
            "source_successor_state_abs_max": max(errors),
            "self_replay": metadata["source_kind"] == "mjwarp",
            "control_lengths": sorted({key[0] for key in groups}),
            "action_protocol": metadata["action_protocol"],
            "preserve_world_ids": preserve_world_ids,
            "oracle_deterministic_mode": self.env.cfg.sim.physics.deterministic_mode,
            "source_execution": metadata.get("execution", {}),
        }


def run_sequence_worker(env, adapter, args) -> None:
    """Run sequence work with an optional force-free geometry MDP for deterministic diagnostics."""
    if args.stage == "sequence-fit-probe":
        from isaaclab_neural.train.opd_sequence import run_sequence_fit_probe

        run_sequence_fit_probe(env, adapter, args)
        return
    from isaaclab_neural.rl.rsl_rl.matched_mj import MatchedGeometryBridge

    bridge = (
        MatchedGeometryBridge(env.unwrapped, adapter)
        if args.sequence_geometry_mdp and args.backend == "mjwarp"
        else None
    )
    try:
        _run_sequence_worker(env, adapter, args)
    finally:
        if bridge:
            bridge.close()


def _run_sequence_worker(env, adapter, args) -> None:
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json

    base = env.unwrapped
    if args.stage == "sequence-oracle":
        oracle = SequenceOracle(base, adapter)
        reports = []
        if args.sequence_serve:
            for line in sys.stdin:
                request = json.loads(line)
                result = oracle.label(
                    request["source"], request["target"], preserve_world_ids=request.get("preserve_world_ids", False)
                )
                reports.append(result)
                print(
                    "OPD_SEQUENCE_RESPONSE " + json.dumps({"request_id": request["request_id"], **result}), flush=True
                )
        else:
            for source in args.queries:
                reports.append(oracle.label(source, args.output / Path(source).name))
        _write_json(args.output / "report.json", {"requests": reports, "environment_initializations": 1})
        return
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.eval.policy_benchmark import load_benchmark_config
    from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics
    from isaaclab_neural.utils.python_utils import set_random_seed

    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    cfg, _ = load_benchmark_config(source, args.policy_snapshot)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(base.device))
    runner.alg.load(
        source, {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}, strict=True
    )
    set_random_seed(args.seed)
    obs, _ = wrapped.reset()
    ids = args.sequence_env_ids if args.sequence_env_ids is not None else list(range(min(base.num_envs, 8)))
    recorder = SequenceRecorder(base, adapter, env_ids=ids, sequence_controls=args.sequence_controls)
    recorder.begin_iteration(0)
    set_random_seed(777)
    runner.alg.train_mode()
    try:
        with torch.inference_mode():
            for _ in range(cfg["num_steps_per_env"]):
                action = runner.alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(action)
                recorder.finish_control_step(dones)
                runner.alg.process_env_step(obs, rewards, dones, extras)
            runner.alg.compute_returns(obs)
        consistency = PpoDiagnostics().check_consistency(runner.alg)
        if not consistency["passed"]:
            raise RuntimeError("PPO collection consistency failed.")
        records = recorder.end_iteration()
        metadata = {
            "physics_dt": base.physics_dt,
            "decimation": base.cfg.decimation,
            "history_steps": 10,
            "sequence_controls": args.sequence_controls,
            "iteration_id": 0,
            "collection_id": str(args.output.resolve()),
            "split": args.split,
            "policy_hash": file_sha256(args.policy_snapshot),
            "dynamics_hash": file_sha256(args.dynamics),
            "source_kind": args.backend,
            "protocol": simulation_protocol(base),
            "label_origin": "unlabelled",
            "initial_solver_state_protocol": "canonical_cold_reset"
            if recorder.neural
            else "recorded_mj_qacc_warmstart",
            "initial_physical_state_protocol": "native_newton_state_v1",
            "execution": {
                "deterministic_mode": base.cfg.sim.physics.deterministic_mode,
                "num_substeps": base.cfg.sim.physics.num_substeps,
                "disable_sensors": getattr(base.cfg.sim.physics.solver_cfg, "disable_sensors", None),
                "geometry_mdp": args.sequence_geometry_mdp,
            },
            "history_origin": "actual_neural_runtime_including_reset_reconstruction"
            if recorder.neural
            else "physical_frames",
        }
        metadata["source_kind"] = "mjwarp" if args.backend == "mjwarp" else "nerd"
        write_sequence_shard(args.output / "sequences.pt", records, metadata=metadata)
        changed = sum(bool((row["action_targets"][1:] - row["action_targets"][:-1]).abs().any()) for row in records)
        _write_json(
            args.output / "report.json",
            {
                "selected_env_ids": ids,
                "records": len(records),
                "changing_target_records": changed,
                "control_lengths": sorted({len(row["actions"]) for row in records}),
                "verified_callbacks": recorder.verified_callbacks,
                "verified_histories": recorder.verified_histories,
                "ppo_consistency": consistency,
                "ppo_optimizer_updates": 0,
                "dynamics_updates": 0,
            },
        )
    finally:
        recorder.close()
