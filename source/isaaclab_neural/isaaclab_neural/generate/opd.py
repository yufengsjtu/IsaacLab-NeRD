# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded G1 cold-start collection and MJWarp oracle workers, one backend per process."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import uuid
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_VALUE_FIELDS
from isaaclab_neural.data.opd import CONTEXT_FIELDS, QUERY_MODE, file_sha256, write_context_shard
from isaaclab_neural.generate.adapter import DataGenerationAdapter, NewtonDataGenerationBackend
from isaaclab_neural.physics.applied_joint_f import backend_applied_effort
from isaaclab_neural.utils.checkpoint import load_opd_policy_snapshot, save_opd_policy_snapshot


class ShadowBackend(NewtonDataGenerationBackend):
    """Use a private collision buffer, without changing analytic solver contacts."""

    def __init__(self, env):
        super().__init__(env.device)
        import newton

        from isaaclab.utils.string import string_to_callable

        manager = env.sim.physics_manager
        self._manager = string_to_callable(manager) if isinstance(manager, str) else manager
        self.pipeline = newton.CollisionPipeline(self.model, **self._manager._collision_cfg.to_pipeline_args())
        self.contacts = self.pipeline.contacts()

    @property
    def native_contacts(self):
        return self.contacts

    def collide_native_contacts(self):
        self.state = self._manager._state_0
        self.pipeline.collide(self.state, self.contacts)
        return self.contacts


def simulation_protocol(env) -> dict:
    """Capture the G1 ordering, actuation and time contract used by a query."""
    robot = env.scene["robot"]
    if robot.data.has_joint_ordering or robot.num_joints != 37:
        raise ValueError("This initial OPD worker requires the current 37-joint G1 backend ordering.")
    if type(env.action_manager.get_term("joint_pos")).__name__ != "JointPositionAction":
        raise ValueError("OPD requires the stock absolute joint-position action contract.")
    if abs(env.physics_dt - 0.005) > 1e-10 or env.cfg.decimation != 4:
        raise ValueError("G1 OPD requires 5 ms physics and action hold=4.")
    model = env.sim.physics_manager._model
    bodies = wp.to_torch(model.body_world) == 0
    shapes = wp.to_torch(model.shape_world) <= 0
    physical = {
        name: wp.to_torch(getattr(model, name))[bodies].cpu().tolist()
        for name in ("body_mass", "body_com", "body_inertia")
    }
    physical.update(
        {
            name: wp.to_torch(getattr(model, name))[shapes].cpu().tolist()
            for name in ("shape_material_mu", "shape_material_ke", "shape_material_kd")
        }
    )
    physical["joint_armature"] = wp.to_torch(model.joint_armature).reshape(env.num_envs, -1)[0].cpu().tolist()
    randomized = getattr(env.cfg, "mbpo_randomized_physics", False)
    if randomized:
        for name in ("body_mass", "body_inertia"):
            physical.pop(name)
    protocol = {
        "physical_parameters": physical,
        "physics_dt": env.physics_dt,
        "control_decimation": env.cfg.decimation,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "stiffness": robot.data.joint_stiffness.torch[0].tolist(),
        "damping": robot.data.joint_damping.torch[0].tolist(),
        "effort_limits": robot.data.joint_effort_limits.torch[0].tolist(),
        "margin": env.cfg.sim.physics.default_shape_cfg.margin,
        "nominal_physics": not randomized,
    }
    if randomized:
        from isaaclab_neural.data.physics_context import WORLD_PHYSICS_FIELDS

        protocol.update(
            physics_parameter_scope="per_world_recorded",
            world_physics_fields=list(WORLD_PHYSICS_FIELDS),
            asset_contract=env.cfg.mbpo_asset_contract,
            event_contract=env.cfg.mbpo_event_contract,
        )
    return protocol


class ContextRecorder:
    """Observe physical frames, including real NeRD reset-history reconstruction.

    The caller must call finish_control_step after each env.step. Selection uses a
    private RNG. Only selected environments are copied to CPU. This class does not
    compute observations or write simulation state/control. Explicit ``env_ids``
    enables subset recording; finite-state checks remain global, but unselected
    failure histories are not claimed to have been retained.
    """

    def __init__(
        self,
        env,
        adapter,
        *,
        capacity: int = 128,
        selected_envs: int = 8,
        seed: int = 0,
        verify_frames: int = 0,
        env_ids: list[int] | None = None,
    ):
        self.env, self.adapter = env, adapter
        self.manager = env.sim.physics_manager
        if self.manager._decimation != 1 or env.cfg.sim.physics.use_cuda_graph:
            raise ValueError("Recording requires eager, one-physics-frame backend stepping.")
        self.neural = getattr(env, "neural_adapter", None)
        if env_ids is None and selected_envs < env.num_envs:
            raise ValueError("The bounded cold-start recorder must retain every environment's failure context.")
        self.ids = list(range(env.num_envs)) if env_ids is None else list(env_ids)
        if not self.ids or len(set(self.ids)) != len(self.ids) or min(self.ids) < 0 or max(self.ids) >= env.num_envs:
            raise ValueError("Selected environment IDs must be distinct valid world indices.")
        self.selected = torch.tensor(self.ids, device=env.device, dtype=torch.long)
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.history = {i: deque(maxlen=10) for i in self.ids}
        self.episodes = np.zeros(env.num_envs, dtype=np.int64)
        self.frame = 0
        self.rows = defaultdict(list)
        self.seen = defaultdict(int)
        self.failures = []
        self.before_data = None
        self.policy_update_id = 0
        self.verify_frames = verify_frames
        self.verified_callbacks = 0
        self._before_callback = partial(self._guarded, self.before)
        self._after_callback = partial(self._guarded, self.after)
        self.manager.register_post_actuator_callback(self._before_callback)
        self.manager.register_post_step_callback(self._after_callback)

    def _guarded(self, callback):
        if self.frame >= self.verify_frames:
            return callback()
        watched = []
        objects = (
            (self.manager._state_0, ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")),
            (
                self.manager._control,
                ("joint_f", "joint_target_q", "joint_target_qd", "joint_target_pos", "joint_target_vel"),
            ),
            (
                self.manager._contacts,
                (
                    "rigid_contact_count",
                    "rigid_contact_shape0",
                    "rigid_contact_shape1",
                    "rigid_contact_point0",
                    "rigid_contact_point1",
                    "rigid_contact_normal",
                ),
            ),
        )
        for obj, names in objects:
            for name in names:
                array = getattr(obj, name, None)
                if array is not None:
                    view = wp.to_torch(array).view(torch.uint8)
                    watched.append((name, view, view.clone()))
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.env.device) if str(self.env.device).startswith("cuda") else None
        try:
            return callback()
        finally:
            for name, view, previous in watched:
                if not torch.equal(view, previous):
                    raise RuntimeError(f"Recorder mutated physics buffer {name}.")
            if not torch.equal(rng, torch.get_rng_state()):
                raise RuntimeError("Recorder advanced the global torch RNG.")
            if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(self.env.device)):
                raise RuntimeError("Recorder advanced the policy CUDA RNG.")
            self.verified_callbacks += 1

    def _snapshot(self, solver, pairs, env_id):
        inputs = solver.get_raw_neural_model_inputs()
        frame = {key: inputs[key][env_id, 0].detach().cpu().clone() for key in CONTEXT_FIELDS}
        selected = pairs["world_ids"] == env_id
        frame["native"] = {key: pairs[key][selected].detach().cpu().clone() for key in NATIVE_CONTACT_VALUE_FIELDS}
        term = self.env.action_manager.get_term("joint_pos")
        frame["actions"] = term.raw_actions[env_id].detach().cpu().clone()
        frame["action_targets"] = term.processed_actions[env_id].detach().cpu().clone()
        return frame

    def _batch_snapshot(self, solver, pairs):
        inputs = solver.get_raw_neural_model_inputs()
        values = {
            key: inputs[key][:, 0].index_select(0, self.selected).detach().cpu().clone() for key in CONTEXT_FIELDS
        }
        term = self.env.action_manager.get_term("joint_pos")
        values["actions"] = term.raw_actions.index_select(0, self.selected).detach().cpu().clone()
        values["action_targets"] = term.processed_actions.index_select(0, self.selected).detach().cpu().clone()
        mask = torch.isin(pairs["world_ids"], self.selected)
        native = {key: pairs[key][mask].detach().cpu().clone() for key in NATIVE_CONTACT_VALUE_FIELDS}
        return {
            env_id: {
                **{key: value[j] for key, value in values.items()},
                "native": {key: value[native["world_ids"] == env_id] for key, value in native.items()},
            }
            for j, env_id in enumerate(self.ids)
        }

    def before(self):
        if self.neural is not None:
            return
        self.adapter.sync()
        self.before_data = self._batch_snapshot(self.adapter.solver, self.adapter.native_contact_pairs())
        efforts = backend_applied_effort(self.env.scene["robot"]).index_select(0, self.selected).detach().cpu()
        for j, env_id in enumerate(self.ids):
            self.before_data[env_id]["joint_f"][6:] = efforts[j]

    def after(self):
        if self.neural is not None:
            solver = self.neural.solver
            current = {i: self._snapshot(solver, solver.contact_adapter.native_contact_pairs(), i) for i in self.ids}
        else:
            current = self.before_data
        next_states = torch.empty_like(self.adapter.solver.states)
        self.adapter.solver._acquire_states_to_torch(self.manager._state_0, next_states)
        self.adapter.solver.wrap2PI(next_states)
        finite = torch.isfinite(next_states).all(dim=-1)
        selected_next = next_states.index_select(0, self.selected).detach().cpu().clone()
        selected_finite = torch.isfinite(selected_next).all(dim=-1).tolist()
        for j, env_id in enumerate(self.ids):
            frame = current[env_id]
            if not all(torch.isfinite(frame[key]).all() for key in CONTEXT_FIELDS):
                raise FloatingPointError("Non-finite OPD input context.")
            if self.neural is None:
                frame["observed_next_state"] = selected_next[j].clone()
                frame["physical_frame_id"] = self.frame
                frame["physical_episode_id"] = int(self.episodes[env_id])
            self.history[env_id].append(frame)
            history = list(self.history[env_id])
            if self.neural is not None:
                actual = self.neural.solver.states_history
                if len(actual) != len(history):
                    raise RuntimeError("OPD history length differs from the model's actual context.")
                for expected, recorded in zip(actual, history):
                    for key in ("states", "joint_f", "root_body_q"):
                        torch.testing.assert_close(
                            expected[key][env_id].detach().cpu(), recorded[key], rtol=0, atol=1e-5
                        )
            length = len(history)
            self.seen[length] += 1
            capacity = self.capacity if length == 10 else min(self.capacity, 16)
            slot = (
                len(self.rows[length])
                if len(self.rows[length]) < capacity
                else int(self.rng.integers(self.seen[length]))
            )
            record = {
                "history": history,
                "next": selected_next[j],
                "env_id": env_id,
                "episode_id": int(self.episodes[env_id]),
                "frame_id": self.frame,
                "policy_update_id": self.policy_update_id,
            }
            if not selected_finite[j]:
                self.failures.append(record)
            elif slot < capacity:
                if slot == len(self.rows[length]):
                    self.rows[length].append(record)
                else:
                    self.rows[length][slot] = record
        self.frame += 1
        if not bool(finite.all()):
            raise FloatingPointError(
                "Non-finite NeRD output; last valid selected contexts retained for oracle queries."
            )
        return selected_next

    def finish_control_step(self, done: torch.Tensor):
        ids = torch.nonzero(done).flatten().tolist()
        self.episodes[ids] += 1
        if not ids:
            return
        if self.neural is None or not self.neural.solver.states_history:
            for i in ids:
                if i in self.history:
                    self.history[i].clear()
            return
        solver = self.neural.solver
        for i in ids:
            if i in self.history:
                frame = self._snapshot(solver, solver.contact_adapter.native_contact_pairs(), i)
                self.history[i] = deque([frame] * len(solver.states_history), maxlen=10)

    def close(self):
        self.manager.unregister_post_step_callback(self._after_callback)
        self.manager._post_actuator_callbacks.remove(self._before_callback)

    def export(
        self, output: Path, metadata: dict, *, env_ids: set[int] | None = None, lengths: set[int] | None = None
    ) -> list[str]:
        """Write labelled MJ contexts or unlabelled student contexts, grouped by true length."""
        paths = []
        grouped = dict(self.rows)
        if self.neural is not None:
            for failure in self.failures:
                length = len(failure["history"])
                grouped[length] = [*grouped.get(length, []), failure]
        for length, records in sorted(grouped.items()):
            if lengths is not None and length not in lengths:
                continue
            if env_ids is not None:
                records = [row for row in records if row["env_id"] in env_ids]
            if not records:
                continue
            payload = {
                key: torch.stack([torch.stack([frame[key] for frame in row["history"]]) for row in records])
                for key in (*CONTEXT_FIELDS, "actions", "action_targets")
            }
            pairs = [frame["native"] for row in records for frame in row["history"]]
            payload["native_contacts"] = {
                key: torch.cat([frame[key] for frame in pairs]) for key in NATIVE_CONTACT_VALUE_FIELDS
            }
            payload["native_contacts"]["frame_counts"] = torch.tensor(
                [len(frame["world_ids"]) for frame in pairs], dtype=torch.int64
            ).reshape(len(records), length)
            world_ids = torch.tensor([r["env_id"] for r in records], dtype=torch.int64)
            payload["trajectory_context"] = {
                key: world_ids for key in ("state_world_id", "root_world_id", "contact_world_id")
            }
            identities = torch.tensor(
                [[r[k] for k in ("env_id", "episode_id", "frame_id", "policy_update_id")] for r in records]
            )
            next_states = torch.stack([r["next"] for r in records])
            meta = {
                **metadata,
                "families": [f"{metadata['collection_id']}:{r['env_id']}:{r['episode_id']}" for r in records],
            }
            diagnostics = {"query_identity": identities, "student_next_states": next_states}
            sequence_labels = None
            if self.neural is None:
                sequence_labels = torch.stack(
                    [torch.stack([frame["observed_next_state"] for frame in row["history"]]) for row in records]
                )
                diagnostics["history_frame_ids"] = torch.tensor(
                    [[frame["physical_frame_id"] for frame in row["history"]] for row in records], dtype=torch.int64
                )
                diagnostics["history_episode_ids"] = torch.tensor(
                    [[frame["physical_episode_id"] for frame in row["history"]] for row in records], dtype=torch.int64
                )
                meta["sequence_label_origin"] = "observed_mj_post_step"
            path = output / f"context_h{length:02d}.hdf5"
            write_context_shard(
                path,
                payload,
                metadata=meta,
                labels=next_states if self.neural is None else None,
                diagnostics=diagnostics,
                sequence_labels=sequence_labels,
            )
            paths.append(str(path.resolve()))
        return paths


def _raw_rollouts(file):
    data = file["data"]
    result = {key: torch.from_numpy(np.asarray(data[key])) for key in (*CONTEXT_FIELDS, "actions", "action_targets")}
    native = file["contacts/native_pairs"]
    counts = np.diff(native["frame_offsets"][:]).reshape(result["states"].shape[:2])
    result["native_contacts"] = {key: torch.from_numpy(np.asarray(native[key])) for key in NATIVE_CONTACT_VALUE_FIELDS}
    result["native_contacts"]["frame_counts"] = torch.from_numpy(counts)
    result["trajectory_context"] = {
        key: torch.from_numpy(np.asarray(file[f"context/trajectories/{key}"]))
        for key in ("state_world_id", "root_world_id", "contact_world_id")
    }
    return result


@torch.no_grad()
def label_queries(env, adapter, paths: list[str], output: Path, *, horizon: int = 10) -> list[dict]:
    """Label exact recorded actions in MJWarp without adding shadow PD as feedforward."""
    protocol = simulation_protocol(env)
    results = []
    for filename in paths:
        path = Path(filename)
        with h5py.File(path, "r") as file:
            if file["data"].attrs["mode"] not in {QUERY_MODE, "counterfactual_context"}:
                raise ValueError("Oracle input must be an OPD context shard.")
            metadata = json.loads(file.attrs["opd_metadata"])
            if metadata["protocol"] != protocol:
                raise ValueError("Oracle physics/actuation protocol differs from source.")
            payload = _raw_rollouts(file)
            diagnostics = {
                key: torch.from_numpy(np.asarray(value))
                for key, value in file["opd"].items()
                if key != "target_next_states"
            }
            original_labels = (
                torch.from_numpy(np.asarray(file["opd/target_next_states"]))
                if "target_next_states" in file["opd"]
                else None
            )
        count = len(payload["states"])
        predictions = []
        rollouts = []
        pd_errors = []
        for start in range(0, count, env.num_envs):
            end = min(start + env.num_envs, count)
            ids = list(range(start, end)) + [end - 1] * (env.num_envs - (end - start))
            state = payload["states"][ids, -1].to(env.device)
            env.scene["robot"].actuators.reset()
            env.action_manager.reset()
            adapter.backend.state = env.sim.physics_manager._state_0
            adapter.control.joint_f.zero_()
            adapter.reset(state)
            torch.testing.assert_close(adapter.solver.states, state, rtol=0, atol=1e-5)
            for key in ("root_body_q", "root_body_qd"):
                torch.testing.assert_close(
                    getattr(adapter.solver, key), payload[key][ids, -1].to(env.device), rtol=0, atol=1e-4
                )
            action = payload["actions"][ids, -1].to(env.device)
            env.action_manager.process_action(action)
            torch.testing.assert_close(
                env.action_manager.get_term("joint_pos").processed_actions,
                payload["action_targets"][ids, -1].to(env.device),
                rtol=0,
                atol=1e-6,
            )
            adapter.step_action_frame(action, process_action=False, overlay_applied_effort=False)
            if bool(wp.to_torch(adapter.control.joint_f).any()):
                raise RuntimeError("Analytic control contains unexpected feedforward effort.")
            effort = backend_applied_effort(env.scene["robot"])
            pd = (effort - payload["joint_f"][ids, -1, 6:].to(env.device)).abs().amax(dim=-1)
            if float(pd.max()) > 1e-3:
                raise RuntimeError(f"Oracle/source shadow PD mismatch: {float(pd.max())}")
            trajectory = [adapter.solver.states[: end - start].detach().cpu().clone()]
            for _ in range(horizon - 1):
                adapter.step_action_frame(action, process_action=False, overlay_applied_effort=False)
                trajectory.append(adapter.solver.states[: end - start].detach().cpu().clone())
            predictions.append(trajectory[0])
            rollouts.append(torch.stack(trajectory, dim=1))
            pd_errors.append(pd[: end - start].cpu())
        labels = torch.cat(predictions)
        diagnostics["first_pd_abs_error"] = torch.cat(pd_errors)
        diagnostics["oracle_rollout_states"] = torch.cat(rollouts)
        meta = {
            **metadata,
            "label_origin": "mjwarp",
            "source_shard_sha256": file_sha256(path),
            "oracle_horizon": horizon,
            "oracle_action_protocol": "hold_recorded_target",
        }
        target = output / path.name
        write_context_shard(target, payload, metadata=meta, labels=labels, diagnostics=diagnostics)
        row = {
            "source": str(path),
            "output": str(target.resolve()),
            "count": count,
            "history_length": payload["states"].shape[1],
            "pd_abs_max": float(torch.cat(pd_errors).max()),
            "oracle_physics_transitions": ((count + env.num_envs - 1) // env.num_envs) * env.num_envs * horizon,
            "oracle_horizon": horizon,
            "action_protocol": "hold_recorded_target",
        }
        if original_labels is not None:
            delta = labels - original_labels
            row["self_replay_state_abs_max"] = float(delta.abs().max())
        results.append(row)
    return results


def _run_condition_worker(env, adapter, args) -> None:
    from contextlib import nullcontext

    from isaaclab_neural.eval.target_conditioning import (
        evaluate_conditioned,
        make_pair_contexts,
        run_condition_fit,
        score_condition_pairs,
    )

    context = (
        torch.cuda.stream(wp.stream_to_torch(adapter.solver.device))
        if str(env.device).startswith("cuda")
        else nullcontext()
    )
    with context:
        if args.stage == "condition-pairs":
            make_pair_contexts(env, adapter, args.queries, args.output, seed=args.sampler_seed)
        elif args.stage == "condition-fit":
            run_condition_fit(env, adapter, args.dynamics, args.conditioning_config, args.output)
        elif args.stage == "condition-score":
            score_condition_pairs(env, adapter, args.dynamics, args.conditioned_model, args.queries, args.output)
        else:
            evaluate_conditioned(env, adapter, args.dynamics, args.conditioned_model, args.queries, args.output)


def _run_extended_worker(env, adapter, args) -> None:
    if args.stage == "active-loss-calibrate":
        from isaaclab_neural.rl.rsl_rl.active_cycle import calibrate_active_objective

        calibrate_active_objective(env, args)
        return
    if args.stage == "active-sequence-profile":
        from isaaclab_neural.train.opd_sequence import run_sequence_profile

        run_sequence_profile(env, args)
        return
    if args.stage == "active-joint-f-evaluate":
        from isaaclab_neural.eval.coverage import run_joint_f_retention

        run_joint_f_retention(env, args)
        return
    if args.stage == "active-stage-fit":
        from isaaclab_neural.rl.rsl_rl.active_cycle import fit_active_stage

        fit_active_stage(env, args)
        return
    if args.stage == "active-evaluate":
        from isaaclab_neural.eval.coverage import run_coverage_study

        run_coverage_study(env, args)
        return
    if args.stage == "active-fit":
        from isaaclab_neural.eval.learnability import run_active_fit_study

        run_active_fit_study(env, args)
        return
    if args.stage == "active-collect":
        from isaaclab_neural.generate.active_opd import run_active_collection

        run_active_collection(env, adapter, args)
        return
    if args.stage.startswith("sequence-"):
        from isaaclab_neural.generate.opd_sequence import run_sequence_worker

        run_sequence_worker(env, adapter, args)
        return
    if args.stage == "policy-benchmark":
        from isaaclab_neural.eval.policy_benchmark import run_policy_benchmark

        run_policy_benchmark(env, adapter, args)
        return
    if args.stage == "study-rl":
        from isaaclab_neural.rl.rsl_rl.matched_mj import run_policy_study

        run_policy_study(env, adapter, args)
        return
    if args.stage.startswith("matched-"):
        from isaaclab_neural.rl.rsl_rl.matched_mj import run_matched_baseline

        run_matched_baseline(env, adapter, args)
        return
    if args.stage == "ppo-diagnose":
        from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import run_ppo_diagnostics

        run_ppo_diagnostics(env, args)
        return
    if args.stage.startswith("condition-"):
        _run_condition_worker(env.unwrapped, adapter, args)
        return
    from isaaclab_neural.rl.rsl_rl.online_opd import evaluate_online_policy, fit_online_candidate, run_online_arm

    if args.stage == "online-arm":
        run_online_arm(env, adapter, args)
    elif args.stage == "online-fit":
        fit_online_candidate(env.unwrapped, adapter, args.dynamics, args.online_config, args.output)
    else:
        evaluate_online_policy(env, args)


def _configure_benchmark_mdp(env_cfg, args) -> None:
    if args.stage == "policy-benchmark":
        from isaaclab_neural.rl.rsl_rl.matched_mj import mdp_contract

        env_cfg.observations.policy.enable_corruption = False
        args.benchmark_mdp_contract = mdp_contract(env_cfg)


def _configure_sequence_env(env_cfg, args) -> None:
    if args.stage == "active-collect":
        from isaaclab_neural.generate.active_opd import configure_active_collection

        configure_active_collection(env_cfg, args)
    if args.stage.startswith("sequence-") and args.sequence_deterministic:
        env_cfg.sim.physics.deterministic_mode = "run_to_run"
        env_cfg.sim.physics.solver_cfg.disable_sensors = True


def _validate_sequence_fit(parser, args) -> None:
    if args.stage != "sequence-fit-probe":
        return
    if args.backend != "nerd" or len(args.queries) != 1 or not args.sequence_reference:
        parser.error("P2 fit requires NeRD, one labelled sequence shard and --sequence_reference.")
    if not args.sequence_train_env_ids or not args.sequence_validation_env_ids:
        parser.error("P2 fit requires explicit disjoint train/validation world IDs.")
    if (
        not 1 <= args.sequence_micro_batch_size <= args.num_envs
        or not 2 <= args.sequence_fit_epochs <= args.sequence_fit_max_epochs <= 8
    ):
        parser.error("P2 microbatch must fit the scratch worlds; epoch budget must lie in [2, 8].")
    if not 0 < args.probe_lr <= 1e-5 or not 1 <= args.sequence_fit_physics_steps <= 16:
        parser.error("P2 fit requires LR <=1e-5 and a horizon of1..16 physical steps.")


def _validate_active_collection(parser, args) -> None:
    if args.stage != "active-collect":
        return
    if args.backend != "mjwarp" or not args.policy_snapshot or not 8 <= args.num_envs <= 1024:
        parser.error("Active collection requires MJWarp, an explicit policy and8..1024 worlds.")
    if not 1 <= args.active_updates <= 100 or not 1 <= args.active_eval_steps <= 256:
        parser.error("Active collection is bounded to100 PPO updates and256 evaluation controls.")
    if not 0 <= args.active_reward_weight <= 1 or not 0 < args.active_error_cap <= 100:
        parser.error("Invalid bounded curiosity weight or error cap.")
    if not 4 <= args.active_record_envs <= 64 or not 64 <= args.active_capacity <= 4096:
        parser.error("Recording requires4..64 selected worlds and capacity64..4096 per iteration.")
    if args.active_eval_seed not in (20, 21, 22):
        parser.error("Use development seeds20/21/22, not reserved final policy-test seeds.")
    if getattr(args, "active_random_init", False) and args.active_mode != "collect_benchmark":
        parser.error("Fresh untrained policies are restricted to evaluation-only benchmark collection.")
    if args.active_mode == "collect_benchmark" and (args.seed not in range(60, 80) or args.split != "validation"):
        parser.error("RL coverage benchmarks require validation split and reserved seeds60..79.")
    if args.active_mode == "collect" and (args.seed not in (20, 21, 22) or args.split != "validation"):
        parser.error("Protected collection requires validation split and a development-validation seed.")
    if args.active_mode in {"train", "collect_train"} and (args.seed not in range(10, 20) or args.split != "train"):
        parser.error("Collector training-data collection requires train split and seed10..19.")


def _validate_probe_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _validate_sequence_fit(parser, args)
    _validate_active_collection(parser, args)
    if args.stage == "active-fit" and (args.backend != "nerd" or args.num_envs != 4 or not args.active_fit_config):
        parser.error("Active fit studies require NeRD, four scratch worlds and an explicit study config.")
    if args.stage in {"active-evaluate", "active-stage-fit", "active-loss-calibrate"} and (
        args.backend != "nerd" or args.num_envs not in (4, 16) or not args.active_fit_config
    ):
        parser.error("Fixed-data studies require NeRD,4/16 declared scratch worlds and an explicit config.")
    if args.stage == "active-joint-f-evaluate" and (args.backend != "nerd" or not args.active_fit_config):
        parser.error("Joint-force retention requires NeRD and an explicit held-out evaluation config.")
    if args.stage == "active-sequence-profile" and (
        args.backend != "nerd" or not args.active_fit_config or args.num_envs not in (4, 8, 16, 32, 64)
    ):
        parser.error("Sequence profiling requires NeRD, a training-only config and4/8/16/32/64 scratch worlds.")
    max_envs = 128 if args.stage == "ppo-diagnose" or args.stage.startswith("matched-") else 16
    max_envs = 64 if args.stage in {"active-joint-f-evaluate", "active-sequence-profile"} else max_envs
    max_envs = 1024 if args.stage == "active-collect" else max_envs
    max_envs = 8192 if args.stage == "study-rl" else max_envs
    if args.stage == "study-rl" and (not args.policy_snapshot or not args.rl_study_config):
        parser.error("Extended policy studies require a frozen policy source and an explicit study configuration.")
    if not 1 <= args.horizon <= 10 or args.num_envs > max_envs:
        parser.error(f"This bounded worker supports at most {max_envs} environments and 10 rollout physics steps.")
    if args.sequence_deterministic and (args.backend != "mjwarp" or not args.sequence_geometry_mdp):
        parser.error("Deterministic sequence diagnostics require MJWarp with --sequence_geometry_mdp.")
    if args.stage == "sequence-collect" and not args.policy_snapshot:
        parser.error("Sequence collection requires a policy snapshot.")
    if args.stage == "sequence-oracle" and (args.backend != "mjwarp" or not (args.queries or args.sequence_serve)):
        parser.error("Sequence oracle requires MJWarp and --queries or --sequence_serve.")
    if not 1 <= args.sequence_controls <= 4:
        parser.error("Initial sequence workers support one to four control steps.")
    if args.stage == "policy-benchmark" and (args.backend != "mjwarp" or not args.policy_snapshot):
        parser.error("Policy benchmarking requires MJWarp and an explicit frozen --policy-snapshot.")
    if args.stage.startswith("matched-"):
        if not args.policy_snapshot:
            parser.error("Matched-MDP workers require --policy-snapshot.")
        if args.stage != "matched-mdp-audit" and args.backend != "mjwarp":
            parser.error("Matched analytic training/evaluation requires MJWarp.")
    if args.stage == "ppo-diagnose" and (args.backend != "nerd" or not args.policy_snapshot):
        parser.error("PPO diagnostics require the NeRD backend and an initial --policy-snapshot.")
    if args.stage.startswith("online-"):
        expected = "nerd" if args.stage == "online-arm" else "mjwarp"
        if args.backend != expected:
            parser.error(f"{args.stage} requires backend {expected}.")
        if args.stage != "online-policy-eval" and not args.online_config:
            parser.error("Online arms/fitting require --online-config.")
        if args.stage != "online-fit" and not args.policy_snapshot:
            parser.error("Online policy workers require --policy-snapshot.")
    if args.stage.startswith("condition-"):
        expected_backend = "nerd" if args.stage == "condition-evaluate" else "mjwarp"
        if args.backend != expected_backend:
            parser.error(f"{args.stage} requires backend {expected_backend}.")
        if args.stage == "condition-fit" and not args.conditioning_config:
            parser.error("condition-fit requires --conditioning-config.")
        if args.stage != "condition-fit" and not args.queries:
            parser.error("Pair/evaluation diagnostics require --queries.")
    if args.stage == "fit-probe" and (len(args.queries) != 1 or not args.validation_query):
        parser.error("Fit probe requires one training shard and --validation-query.")
    if not 1 <= args.probe_updates <= 1000 or not 1 <= args.probe_samples <= 256 or not 0 < args.probe_lr <= 1e-3:
        parser.error("Fit probes are limited to 1000 updates, 256 examples and LR <= 1e-3.")
    if args.stage in {"evaluate", "fit-probe"} and args.backend != "nerd":
        parser.error("Dynamics evaluation requires the NeRD backend.")


def g1_worker_tasks(terrain: str, backend: str, matched_mdp: bool) -> tuple[str, str]:
    """Return the config-registration and runtime task IDs for a G1 worker."""
    if terrain not in {"flat", "rough"} or backend not in {"mjwarp", "nerd"}:
        raise ValueError("Unsupported G1 worker terrain or physics backend.")
    analytic = f"Isaac-Velocity-{terrain.title()}-G1"
    configured = f"{analytic}-NeRD-v0" if backend == "nerd" or matched_mdp else analytic
    return configured, analytic if backend == "mjwarp" else configured


def _configure_worker_terrain(env_cfg, args) -> None:
    if args.terrain == "rough":
        from isaaclab_neural.envs.g1_nerd_env import configure_g1_mbpo_rough

        if args.scene_config is None:
            raise ValueError("Rough workers require an explicit shared --scene-config.")
        configure_g1_mbpo_rough(env_cfg, json.loads(args.scene_config.read_text()))


def _worker_scene_selection(args, matched_mdp: bool, parser: argparse.ArgumentParser) -> tuple[dict, bool]:
    settings = json.loads(args.scene_config.read_text()) if args.scene_config else {}
    if settings.get("contact_mdp") == "stock":
        if args.backend != "mjwarp" or settings.get("role") != "evaluate":
            parser.error("Stock force-contact MBPO evaluation requires the explicit MJWarp evaluation role.")
        matched_mdp = False
    return settings, matched_mdp


def main():
    """Launch one bounded diagnostic worker without implicit live-dynamics promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "init",
            "collect",
            "oracle",
            "evaluate",
            "fit-probe",
            "target-probe",
            "condition-pairs",
            "condition-fit",
            "condition-score",
            "condition-evaluate",
            "online-arm",
            "online-fit",
            "online-policy-eval",
            "ppo-diagnose",
            "matched-mdp-audit",
            "matched-mj-train",
            "matched-mj-eval",
            "study-rl",
            "policy-benchmark",
            "sequence-collect",
            "sequence-oracle",
            "sequence-fit-probe",
            "active-collect",
            "active-fit",
            "active-evaluate",
            "active-joint-f-evaluate",
            "active-sequence-profile",
            "active-loss-calibrate",
            "active-stage-fit",
        ),
        required=True,
    )
    parser.add_argument("--backend", choices=("mjwarp", "nerd"), default="mjwarp")
    parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--benchmark_mdp", "--benchmark-mdp", choices=("stock", "geometry"), default="stock")
    parser.add_argument("--sequence_env_ids", type=int, nargs="+")
    parser.add_argument("--sequence_controls", type=int, default=4)
    parser.add_argument("--sequence_serve", action="store_true")
    parser.add_argument("--sequence_geometry_mdp", action="store_true")
    parser.add_argument("--sequence_deterministic", action="store_true")
    parser.add_argument("--sequence_reference", nargs="+", default=[])
    parser.add_argument("--sequence_reference_validation", nargs="+", default=[])
    parser.add_argument("--sequence_train_env_ids", nargs="+", type=int, default=[])
    parser.add_argument("--sequence_validation_env_ids", nargs="+", type=int, default=[])
    parser.add_argument("--sequence_micro_batch_size", type=int, default=2)
    parser.add_argument("--sequence_fit_epochs", type=int, default=4)
    parser.add_argument("--sequence_fit_max_epochs", type=int, default=8)
    parser.add_argument("--sequence_fit_physics_steps", type=int, default=4)
    parser.add_argument("--active_fit_config")
    parser.add_argument("--rl_study_config")
    parser.add_argument(
        "--active_mode", choices=("train", "collect", "collect_train", "collect_benchmark"), default="train"
    )
    parser.add_argument("--active_random_init", action="store_true")
    parser.add_argument("--active_resume", action="store_true")
    parser.add_argument("--active_policy_rng_seed", type=int, default=777)
    parser.add_argument("--active_reward_profile", choices=("base", "error", "base_error"), default="base_error")
    parser.add_argument("--active_record_envs", type=int, default=32)
    parser.add_argument("--active_capacity", type=int, default=128)
    parser.add_argument("--active_reward_weight", type=float, default=0.2)
    parser.add_argument("--active_error_cap", type=float, default=10.0)
    parser.add_argument("--active_error_scope", choices=("full", "root_legs"), default="full")
    parser.add_argument("--active_updates", type=int, default=4)
    parser.add_argument("--active_eval_steps", type=int, default=96)
    parser.add_argument("--active_eval_seed", type=int, default=21)
    parser.add_argument("--dynamics", required=True)
    parser.add_argument("--policy-snapshot")
    parser.add_argument("--reference-policy")
    parser.add_argument("--queries", nargs="+")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--validation-query")
    parser.add_argument("--conditioning-config")
    parser.add_argument("--conditioned-model")
    parser.add_argument("--online-config")
    parser.add_argument("--opd-mode", choices=("fixed", "online"), default="fixed")
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--ppo-initial-lr", type=float)
    parser.add_argument("--probe-updates", type=int, default=1000)
    parser.add_argument("--probe-samples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--capacity-per-length", type=int, default=128)
    parser.add_argument("--record-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--sampler-seed", type=int, default=100)
    parser.add_argument("--split", choices=("train", "validation", "test", "diagnostic"), default="train")
    parser.add_argument("--pool", choices=("initial_exploration", "reference"), default="initial_exploration")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--verify-recording-frames", type=int, default=0)
    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_tasks.utils.hydra import hydra_task_config

    from isaaclab_neural.eval import eval as evaluation
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint
    from isaaclab_neural.utils.python_utils import set_random_seed
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    add_launcher_args(parser)
    args, hydra = parser.parse_known_args()
    if min(args.num_envs, args.steps, args.capacity_per_length, args.record_envs) < 1:
        parser.error("Environment, step and recording budgets must be positive.")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stage in {"init", "oracle", "target-probe"} and args.backend != "mjwarp":
        parser.error("Init/oracle workers require MJWarp.")
    if args.stage == "init" and (args.policy_snapshot or args.reference_policy):
        parser.error("Initial policies cannot load pretrained RL weights.")
    if args.stage == "collect" and bool(args.policy_snapshot) == bool(args.reference_policy):
        parser.error("Collection requires exactly one policy source.")
    if args.reference_policy and (args.backend != "mjwarp" or args.pool != "reference"):
        parser.error("Reference policies are restricted to MJWarp reference collection.")
    if args.stage in {"oracle", "evaluate", "fit-probe", "target-probe"} and not args.queries:
        parser.error("Oracle/evaluation/probes require --queries.")
    _validate_probe_arguments(parser, args)
    roles = {"train": range(10, 20), "validation": range(20, 23), "test": range(30, 33)}
    if args.stage == "collect" and args.split in roles and args.seed not in roles[args.split]:
        parser.error("Data seed is outside the configured train/validation/test split.")
    args.output.mkdir(parents=True)
    matched_mdp = (
        (args.stage.startswith("matched-") or args.stage == "study-rl")
        or (args.stage == "active-collect" and args.active_mode != "collect_benchmark")
        or (args.stage == "policy-benchmark" and args.benchmark_mdp == "geometry")
        or (args.stage.startswith("sequence-") and args.sequence_geometry_mdp)
        or (args.terrain == "rough" and args.stage in {"init", "active-stage-fit"})
    )
    scene_settings, matched_mdp = _worker_scene_selection(args, matched_mdp, parser)
    task, factory_task = g1_worker_tasks(args.terrain, args.backend, matched_mdp)
    sys.argv = [
        sys.argv[0],
        *(["presets=newton_mjwarp"] if args.backend == "mjwarp" and not matched_mdp else []),
        *hydra,
    ]
    checkpoint = load_checkpoint(args.dynamics, device="cpu")
    cfg = copy.deepcopy(get_cfg_from_checkpoint(checkpoint, args.dynamics)["env"]["neural_solver_cfg"])
    cfg.pop("use_cuda_graph", None)
    cfg["neural_model_path"] = args.dynamics
    solver_cfg = NerdSolverCfg(**cfg)
    del checkpoint
    set_random_seed(args.seed)

    @hydra_task_config(task, "rsl_rl_cfg_entry_point")
    def run(env_cfg, agent_cfg):
        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.managers import TerminationTermCfg

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import G1FlatNeRDPPORunnerCfg, g1_mbpo_runner_cfg
        from isaaclab_neural.envs.g1_nerd_env import g1_invalid_state

        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        args.device = args.device or env_cfg.sim.device
        env_cfg.sim.device = args.device
        env_cfg.sim.physics.use_cuda_graph = False
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.push_robot = None
        env_cfg.terminations.invalid_state = TerminationTermCfg(func=g1_invalid_state)
        _configure_worker_terrain(env_cfg, args)
        if matched_mdp:
            from isaaclab_neural.rl.rsl_rl.matched_mj import configure_matched_backend

            args.matched_mdp_audit = configure_matched_backend(env_cfg, args.backend)
        _configure_benchmark_mdp(env_cfg, args)
        _configure_sequence_env(env_cfg, args)
        with launch_simulation(evaluation.build_launch_cfg(env_cfg), args):
            with newton_material_binding_api_autofix():
                kwargs = {"solver_cfg": solver_cfg} if args.backend == "nerd" else {}
                env = gym.make(factory_task, cfg=env_cfg, device=args.device, **kwargs)
            base = env.unwrapped
            base.sim.physics_manager.set_decimation(1)
            base._physics_handles_decimation = False
            protocol = simulation_protocol(base)
            backend = ShadowBackend(base)
            adapter = DataGenerationAdapter(base, copy.deepcopy(solver_cfg), backend=backend)
            if args.stage.startswith(
                (
                    "condition-",
                    "online-",
                    "ppo-diagnose",
                    "matched-",
                    "study-rl",
                    "policy-benchmark",
                    "sequence-",
                    "active-",
                )
            ):
                _run_extended_worker(env, adapter, args)
                env.close()
                return
            if args.stage == "oracle":
                result = label_queries(base, adapter, args.queries, args.output, horizon=args.horizon)
                (args.output / "report.json").write_text(json.dumps(result, indent=2))
                env.close()
                return
            if args.stage == "evaluate":
                from isaaclab_neural.eval.dynamics_consistency import evaluate_contexts

                result = evaluate_contexts(base, adapter, args.queries, args.output)
                result["dynamics_hash"] = file_sha256(args.dynamics)
                (args.output / "report.json").write_text(json.dumps(result, indent=2))
                env.close()
                return
            if args.stage == "fit-probe":
                from isaaclab_neural.eval.learnability import run_small_fit

                run_small_fit(
                    base,
                    args.dynamics,
                    args.queries[0],
                    args.validation_query,
                    args.output,
                    lr=args.probe_lr,
                    updates=args.probe_updates,
                    samples=args.probe_samples,
                    seed=args.sampler_seed,
                )
                env.close()
                return
            if args.stage == "target-probe":
                from isaaclab_neural.eval.learnability import run_target_probe

                run_target_probe(base, adapter, args.queries, args.output, seed=args.sampler_seed)
                env.close()
                return
            if args.reference_policy:
                saved = evaluation.load_saved_agent_cfg_for_policy(
                    args.reference_policy, agent_cfg, device=args.device, seed=args.seed
                )
                env, policy, runner = evaluation.build_policy_env(env, saved, args.reference_policy, args.device)
                runner_cfg = saved.to_dict() if hasattr(saved, "to_dict") else saved
                policy_path = args.reference_policy
            else:
                if args.stage == "init":
                    runner_config = (
                        g1_mbpo_runner_cfg(scene_settings.get("ppo_runner", "nerd_legacy"))
                        if args.terrain == "rough"
                        else G1FlatNeRDPPORunnerCfg()
                    )
                    runner_cfg, _, _, _ = evaluation.normalize_runner_cfg(runner_config, device=args.device)
                else:
                    snapshot = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
                    runner_cfg = copy.deepcopy(snapshot["infos"]["opd"]["runner_cfg"])
                    if snapshot["infos"]["opd"]["provenance"]["protocol"] != protocol:
                        raise ValueError("Initial policy and collection protocol differ.")
                    del snapshot
                env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg["clip_actions"])
                if args.stage == "init":
                    set_random_seed(args.policy_seed)
                runner = OnPolicyRunner(env, copy.deepcopy(runner_cfg), log_dir=None, device=args.device)
                if args.stage == "init":
                    save_opd_policy_snapshot(
                        args.output / "policy_init.pt",
                        runner,
                        runner_cfg,
                        updates_completed=0,
                        provenance={
                            "policy_seed": args.policy_seed,
                            "protocol": protocol,
                            "dynamics_hash": file_sha256(args.dynamics),
                        },
                    )
                    env.close()
                    return
                load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=True)
                policy = runner.get_inference_policy(device=args.device)
                policy_path = args.policy_snapshot
            collector = (
                None
                if args.no_record
                else ContextRecorder(
                    base,
                    adapter,
                    capacity=args.capacity_per_length,
                    selected_envs=args.record_envs,
                    seed=args.sampler_seed,
                    verify_frames=args.verify_recording_frames,
                )
            )
            digest = hashlib.sha256()
            rng_digest = hashlib.sha256()
            trace_actions, trace_states = [], []
            termination_counts = defaultdict(int)
            obs = env.get_observations()
            failure = None
            export_allowed = True
            completed = 0
            try:
                for step in range(args.steps):
                    with torch.inference_mode():
                        action = policy(obs, stochastic_output=True)
                    action_cpu = action.detach().cpu().numpy()
                    digest.update(action_cpu.tobytes())
                    rng_digest.update(torch.get_rng_state().numpy().tobytes())
                    if torch.cuda.is_initialized():
                        rng_digest.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
                    obs, reward, done, extras = env.step(action)
                    q = wp.to_torch(base.sim.physics_manager._state_0.joint_q).cpu().numpy().reshape(base.num_envs, -1)
                    qd = (
                        wp.to_torch(base.sim.physics_manager._state_0.joint_qd).cpu().numpy().reshape(base.num_envs, -1)
                    )
                    digest.update(q.tobytes())
                    digest.update(qd.tobytes())
                    trace_actions.append(action_cpu.copy())
                    trace_states.append(np.concatenate((q, qd), axis=-1))
                    for name in base.termination_manager.active_terms:
                        termination_counts[name] += int(base.termination_manager.get_term(name).sum())
                    if collector is not None:
                        collector.finish_control_step(done)
                    evaluation.reset_policy(policy, runner, done)
                    completed = step + 1
            except FloatingPointError as error:
                failure = str(error)
            except Exception as error:
                failure = repr(error)
                export_allowed = False
                raise
            finally:
                meta = {
                    "protocol": protocol,
                    "physics_dt": base.physics_dt,
                    "split": args.split,
                    "source_kind": args.backend,
                    "pool": args.pool,
                    "seed": args.seed,
                    "collection_id": str(uuid.uuid4()),
                    "policy_hash": file_sha256(policy_path),
                    "dynamics_hash": file_sha256(args.dynamics),
                    "stochastic_policy": True,
                    "label_origin": "mjwarp" if args.backend == "mjwarp" else "unlabelled",
                }
                paths = []
                if collector is not None:
                    collector.close()
                    if export_allowed:
                        paths = collector.export(args.output, meta)
                np.savez_compressed(
                    args.output / "control_trace.npz",
                    actions=np.asarray(trace_actions),
                    states=np.asarray(trace_states),
                )
                report = {
                    "metadata": meta,
                    "completed_control_steps": completed,
                    "rollout_digest": digest.hexdigest(),
                    "rng_digest": rng_digest.hexdigest(),
                    "termination_counts": dict(termination_counts),
                    "verified_readonly_callbacks": collector.verified_callbacks if collector else 0,
                    "physical_transitions": (collector.frame if collector else completed * 4) * base.num_envs,
                    "failure": failure,
                    "non_finite_output_count": len(collector.failures) if collector else 0,
                    "shards": paths,
                }
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(report, indent=2), flush=True)
                env.close()
            if failure:
                raise RuntimeError(failure)

    run()


if __name__ == "__main__":
    main()
