# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded task-regularized MJWarp collection against a frozen NeRD prediction-error score."""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict, deque

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import quat_apply_inverse

from isaaclab_neural.generate.opd import ContextRecorder, simulation_protocol
from isaaclab_neural.physics.applied_joint_f import backend_applied_effort
from isaaclab_neural.train.opd_sequence import _on_torch_stream


def bounded_model_error(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, *, cap: float):
    """Return frozen-scale relative-state MSE and a bounded dimensionless curiosity score.

    Targets are in the checkpoint's prediction representation and physical units;
    weights are the fixed pretrained coordinate scales, not updated reward RMS.
    Non-finite values are rejected rather than converted into high rewards.
    """
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Curiosity error cap must be finite and positive.")
    if prediction.ndim != 2 or prediction.shape != target.shape or weights.shape != prediction.shape[-1:]:
        raise ValueError("Expected matching [batch, features] predictions/targets and fixed feature weights.")
    error = ((prediction - target) * weights).square().mean(-1)
    if not torch.isfinite(error).all():
        raise FloatingPointError("Non-finite model error cannot be a curiosity reward.")
    return error, torch.log1p(error.clamp(max=cap)) / math.log1p(cap)


def active_query_validity(
    states: torch.Tensor,
    joint_limits: torch.Tensor,
    minimum_separations: torch.Tensor,
    *,
    joint_limit_tolerance_rad: float,
    minimum_root_height_m: float = 0.02,
    maximum_penetration_m: float = 0.05,
    quaternion_norm_tolerance: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Check floating-root G1 oracle inputs without clipping or modifying them.

    These are screening diagnostics, not proof of physical validity or a task-success gate.
    Valid falling states remain eligible. Solver joint-limit constraints can be soft; choose
    the required tolerance from real-backend controls rather than silently assuming exact limits.

    Args:
        states: Raw generalized states in SI units, shape [N, 13 + 2 * J], with XYZW root quaternion.
        joint_limits: Configured joint-position limits [rad], shape [J, 2].
        minimum_separations: Smallest signed native surface gaps [m], shape [N]; use zero for no contacts.
        joint_limit_tolerance_rad: Explicit calibrated tolerance outside configured joint limits [rad].
        minimum_root_height_m: Minimum root height above the flat zero-height floor [m].
        maximum_penetration_m: Maximum permitted native-contact penetration [m].
        quaternion_norm_tolerance: Maximum unit-quaternion norm discrepancy.

    Returns:
        Per-reason boolean masks of shape [N]. A query is eligible only when all masks are true.
    """
    if (
        joint_limits.ndim != 2
        or joint_limits.shape[1] != 2
        or states.ndim != 2
        or states.shape[1] != 13 + 2 * len(joint_limits)
        or minimum_separations.shape != states.shape[:1]
    ):
        raise ValueError("Oracle query validity requires matching floating-root states, joint limits and contact gaps.")
    tolerances = (minimum_root_height_m, maximum_penetration_m, joint_limit_tolerance_rad, quaternion_norm_tolerance)
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("Oracle query validity tolerances must be finite and nonnegative.")
    positions = states[:, 7 : 7 + len(joint_limits)]
    return {
        "finite": torch.isfinite(states).all(dim=-1) & torch.isfinite(minimum_separations),
        "root_above_floor": states[:, 2] >= minimum_root_height_m,
        "unit_quaternion": (states[:, 3:7].norm(dim=-1) - 1).abs() <= quaternion_norm_tolerance,
        "within_hard_joint_limits": (
            (positions >= joint_limits[:, 0] - joint_limit_tolerance_rad)
            & (positions <= joint_limits[:, 1] + joint_limit_tolerance_rad)
        ).all(dim=-1),
        "no_deep_contact_penetration": minimum_separations >= -maximum_penetration_m,
    }


def active_coverage_features(
    states: torch.Tensor,
    actions: torch.Tensor,
    foot_contacts: torch.Tensor,
    joint_names: list[str],
    *,
    scope: str = "locomotion",
) -> torch.Tensor:
    """Construct G1 coverage features without world-translation or world-yaw shortcuts.

    Args:
        states: Raw generalized states in SI units with XYZW root quaternion, shape [N, 13 + 2 * J].
        actions: Recorded dimensionless joint-position actions, shape [N, J].
        foot_contacts: Left/right native geometric contact flags, shape [N, 2].
        joint_names: Backend-order joint names.
        scope: ``locomotion`` selects hip/knee/ankle joints; ``full`` retains every joint.

    Returns:
        Features of shape [N, 12 + 4 * K]: root height [m], body-frame projected gravity,
        linear velocity [m/s], angular velocity [rad/s], joint-angle sine/cosine,
        joint velocity [rad/s], raw action and two contact flags. K is the selected joint count.
        Normalization scales must be estimated from a separately declared training reference.
    """
    joints = len(joint_names)
    if (
        scope not in {"locomotion", "full"}
        or not joints
        or len(set(joint_names)) != joints
        or states.ndim != 2
        or states.shape[1] != 13 + 2 * joints
        or actions.shape != (len(states), joints)
        or foot_contacts.shape != (len(states), 2)
        or foot_contacts.dtype != torch.bool
    ):
        raise ValueError("Coverage requires matching floating-root state/action/contact layouts and an explicit scope.")
    selected = [
        index
        for index, name in enumerate(joint_names)
        if scope == "full" or any(part in name for part in ("hip", "knee", "ankle"))
    ]
    quaternion = states[:, 3:7]
    if not selected or not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise ValueError("Coverage features require finite states/actions and selected joints.")
    norms = quaternion.norm(dim=-1, keepdim=True)
    if bool((norms <= 1e-6).any()):
        raise ValueError("Coverage features require a nondegenerate root quaternion.")
    quaternion = quaternion / norms
    gravity = torch.zeros_like(states[:, :3])
    gravity[:, 2] = -1
    num_q = 7 + joints
    angles = states[:, 7:num_q][:, selected]
    return torch.cat(
        (
            states[:, 2:3],
            quat_apply_inverse(quaternion, gravity),
            quat_apply_inverse(quaternion, states[:, num_q : num_q + 3]),
            quat_apply_inverse(quaternion, states[:, num_q + 3 : num_q + 6]),
            angles.sin(),
            angles.cos(),
            states[:, num_q + 6 :][:, selected],
            actions[:, selected],
            foot_contacts.to(states),
        ),
        dim=-1,
    )


def active_error_weights(weights: torch.Tensor, joint_names: list[str], scope: str = "full") -> torch.Tensor:
    """Select reward-only coordinates while retaining the full-state MSE denominator and checkpoint scales."""
    from isaaclab_neural.eval.learnability import g1_error_groups

    if weights.shape != (13 + 2 * len(joint_names),):
        raise ValueError("Error scope requires the floating-root generalized-state layout.")
    if scope == "full":
        return weights.detach().clone()
    if scope != "root_legs":
        raise ValueError("Unknown active error scope.")
    groups = g1_error_groups(joint_names)
    if not groups["legs"]:
        raise ValueError("Root/leg scoring requires named leg joints.")
    ids = [i for name, indices in groups.items() if name.startswith("root_") or name == "legs" for i in indices]
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask[ids] = True
    return weights.detach().clone().masked_fill(~mask, 0)


def dynamics_curiosity_reward(env) -> torch.Tensor:
    """Read the completed control interval's bounded score, excluding terminated/reset worlds."""
    observer = env.dynamics_error_observer
    bonus = observer.bonus_sum / observer.decimation
    return torch.where(env.reset_buf.bool(), torch.zeros_like(bonus), bonus).detach()


def configure_active_collection(env_cfg, args) -> None:
    """Select explicit reward profiles without changing the common action or termination contract."""
    from isaaclab.managers import RewardTermCfg

    if getattr(args, "active_mode", "train") == "collect_benchmark":
        args.active_reward_audit = {
            "profile": "evaluation_only_stock",
            "error_weight_per_control": 0.0,
            "observation_corruption": env_cfg.observations.policy.enable_corruption,
            "base_definition": "Unmodified stock G1 rewards/terminations and stochastic frozen policy; nominal physics",
        }
        return
    profile = args.active_reward_profile
    if profile not in {"base", "error", "base_error"}:
        raise ValueError("Unknown active-collection reward profile.")
    terms = {name: term for name, term in vars(env_cfg.rewards).items() if isinstance(term, RewardTermCfg)}
    original = {name: term.weight for name, term in terms.items()}
    if profile == "error":
        for term in terms.values():
            term.weight = 0.0
    weight = 0.0 if profile == "base" else args.active_reward_weight
    args.active_reward_audit = {
        "profile": profile,
        "original_base_weights": original,
        "active_base_weights": {name: term.weight for name, term in terms.items()},
        "error_weight_per_control": weight,
        "terminal_error_mask": True,
        "base_definition": "Unmodified existing G1 geometry task reward, including positive and negative terms",
    }
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.sim.physics.deterministic_mode = "run_to_run"
    env_cfg.sim.physics.solver_cfg.disable_sensors = True
    env_cfg.rewards.dynamics_curiosity = RewardTermCfg(
        func=dynamics_curiosity_reward, weight=weight / (env_cfg.sim.dt * env_cfg.decimation)
    )


def initialize_active_policy(runner, cfg: dict, path: str, *, resume: bool, protocol: dict) -> int:
    """Restore collector progress without resetting a resumed PPO optimizer or its adaptive learning rate."""
    from isaaclab_neural.utils.checkpoint import load_opd_policy_snapshot

    source = torch.load(path, map_location="cpu", weights_only=False)
    if source["infos"]["opd"]["provenance"]["protocol"] != protocol:
        raise ValueError("Collector continuation requires the same physics protocol.")
    info = load_opd_policy_snapshot(path, runner, require_initial=not resume)
    if not resume:
        runner.alg.learning_rate = 1e-4
        cfg["algorithm"]["learning_rate"] = 1e-4
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = 1e-4
    return info["ppo_updates_completed"]


def active_recording_worlds(num_envs: int, count: int, *, seed: int, iteration: int) -> list[int]:
    """Choose bounded, rotating recording worlds with permanent 3:1 model-data roles.

    This uses a private NumPy generator; unrecorded worlds are still scored on the GPU.
    The last quarter of world IDs belongs only to diagnostic model validation.
    """
    if num_envs < 4 or count < 4 or iteration < 0:
        raise ValueError("At least four worlds and a nonnegative iteration are required.")
    count = min(count, num_envs)
    if count == num_envs:
        return list(range(num_envs))
    boundary = num_envs - num_envs // 4
    valid_count = max(1, count // 4)
    rng = np.random.default_rng(np.random.SeedSequence([seed, iteration]))
    train = rng.choice(boundary, count - valid_count, replace=False)
    valid = rng.choice(np.arange(boundary, num_envs), valid_count, replace=False)
    return sorted(np.concatenate((train, valid)).tolist())


class DynamicsErrorRecorder(ContextRecorder):
    """Collect real MJ transitions and score them without modifying state, controls or frozen dynamics.

    Prediction is made before the physical transition. Targets are read before
    auto-reset. Scoring resumes only after ten real post-reset physical inputs;
    old episode histories are never used for a rewarded transition.
    """

    def __init__(
        self,
        env,
        adapter,
        model,
        loss_weights,
        *,
        cap: float,
        seed: int,
        record_envs: int = 32,
        capacity: int = 128,
        record_sequences: bool = False,
        error_scope: str = "full",
    ):
        from isaaclab_neural.eval.learnability import g1_error_groups

        ids = active_recording_worlds(env.num_envs, record_envs, seed=seed, iteration=0)
        super().__init__(env, adapter, capacity=capacity, env_ids=ids, seed=seed, verify_frames=32)
        self.record_envs, self.recording_seed = record_envs, seed
        if self.neural is not None:
            raise ValueError("Active data collection must run real MJWarp dynamics.")
        self.device = torch.device(env.device)
        self.model = model.eval().requires_grad_(False)
        self.weights = loss_weights.detach().to(self.device).clone()
        joint_names = simulation_protocol(env)["joint_names"]
        self.error_scope = error_scope
        self.reward_weights = active_error_weights(self.weights, joint_names, error_scope)
        self.error_groups = g1_error_groups(joint_names)
        self.coordinate_error_sum = torch.zeros_like(self.weights, dtype=torch.float64)
        self.reward_errors = []
        self.cap, self.decimation = cap, env.cfg.decimation
        self.raw_history = deque(maxlen=10)
        self.ages = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.bonus_sum = torch.zeros(env.num_envs, device=self.device)
        self.valid_count = torch.zeros_like(self.ages)
        self.pending = None
        self.errors, self.bonuses, self.scored_frames = [], [], []
        self.control_trace = []
        self.score_lookup = {}
        self.full_score_lookup = {}
        self.score_layout = {}
        self.windows = None
        if record_sequences:
            from isaaclab_neural.data.opd_sequence import SequenceWindowBuffer
            from isaaclab_neural.generate.opd_sequence import _require_stateless_pd

            _require_stateless_pd(env)
            self.windows = SequenceWindowBuffer(self.ids, decimation=self.decimation, record_future_inputs=True)
            self.windows.begin_iteration(0)

    def begin_recording_iteration(self, iteration: int) -> None:
        """Rotate only recorded worlds, retaining complete GPU scoring histories for every world."""
        if self.windows is not None:
            raise ValueError("Sequence validation uses fixed worlds and one frozen-policy collection block.")
        ids = active_recording_worlds(
            self.env.num_envs, self.record_envs, seed=self.recording_seed, iteration=iteration
        )
        self.history = {i: self.history.get(i, deque(maxlen=10)) for i in ids}
        self.ids = ids
        self.selected = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.policy_update_id = iteration

    def _record_solver_state(self):
        warm = wp.to_torch(self.manager._solver.mjw_data.qacc_warmstart).index_select(0, self.selected).cpu().clone()
        native = {}
        if self.frame % self.decimation == 0:
            for key in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
                raw = wp.to_torch(getattr(self.manager._state_0, key))
                values = raw.reshape(self.env.num_envs, -1, *raw.shape[1:])
                native[key] = values.index_select(0, self.selected).cpu().clone()
        for j, env_id in enumerate(self.ids):
            self.before_data[env_id]["solver_state"] = {"qacc_warmstart": warm[j]}
            if native:
                self.before_data[env_id]["newton_state"] = {key: value[j] for key, value in native.items()}

    @torch.no_grad()
    @_on_torch_stream
    def before(self):
        super().before()
        if self.windows is not None:
            self._record_solver_state()
        if self.frame % self.decimation == 0:
            self.bonus_sum.zero_()
            self.valid_count.zero_()
        solver = self.adapter.solver
        raw = solver.get_raw_neural_model_inputs()
        fields = ("states", "root_body_q", "joint_f", "gravity_dir", "contact_tokens", "contact_token_overflow")
        frame = {key: raw[key][:, 0].detach().clone() for key in fields}
        frame["joint_f"][:, 6:] = backend_applied_effort(self.env.scene["robot"])
        self.raw_history.append(frame)
        self.ages += 1
        ids = (self.ages >= 10).nonzero().flatten()
        self.pending = None
        if not len(ids):
            return
        history = {
            key: torch.stack([entry[key] for entry in self.raw_history], dim=1).index_select(0, ids) for key in fields
        }
        prepared = solver.process_neural_model_inputs(history)
        prediction = self.model(
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in prepared.items()},
            single_step=True,
        )[:, 0]
        self.pending = (ids, prediction, prepared["states"][:, -1:], prepared["root_body_q"][:, -1:])

    @torch.no_grad()
    @_on_torch_stream
    def after(self):
        if self.pending is not None:
            ids, prediction, initial, root = self.pending
            solver = self.adapter.solver
            actual = torch.empty_like(solver.states)
            solver._acquire_states_to_torch(self.manager._state_0, actual)
            solver.wrap2PI(actual)
            actual_local = solver._convert_states_w2b(
                root, actual.index_select(0, ids)[:, None], translation_only=False
            )
            target = solver.convert_next_states_to_prediction(initial, actual_local, self.env.physics_dt)[:, 0]
            full_error, bonus = bounded_model_error(prediction, target, self.weights, cap=self.cap)
            error = full_error
            if self.error_scope != "full":
                error, bonus = bounded_model_error(prediction, target, self.reward_weights, cap=self.cap)
            self.coordinate_error_sum.add_(((prediction - target) * self.weights).square().double().sum(0))
            self.bonus_sum[ids] += bonus
            self.valid_count[ids] += 1
            self.errors.append(full_error.detach())
            self.reward_errors.append(error.detach())
            recorded = self.selected[self.ages.index_select(0, self.selected) >= 10]
            positions = torch.searchsorted(ids, recorded)
            values = error.index_select(0, positions).cpu().tolist()
            full_values = full_error.index_select(0, positions).cpu().tolist()
            for env_id, value, full_value, position in zip(
                recorded.cpu().tolist(), values, full_values, positions.cpu().tolist()
            ):
                identity = (env_id, int(self.episodes[env_id]), self.frame, self.policy_update_id)
                self.score_lookup[identity] = value
                self.full_score_lookup[identity] = full_value
                self.score_layout[identity] = (len(ids), position)
        if bool(wp.to_torch(self.adapter.control.joint_f).any()):
            raise RuntimeError("The collector injected shadow PD into analytic MJ control.")
        selected_next = super().after()
        if self.windows is not None:
            self.windows.append_frame(
                self.frame - 1,
                self.history,
                {i: selected_next[j] for j, i in enumerate(self.ids)},
                {i: int(self.episodes[i]) for i in self.ids},
            )

    def finish_control_step(self, done: torch.Tensor):
        self.bonuses.append(torch.where(done.bool(), 0.0, self.bonus_sum / self.decimation).detach().clone())
        self.scored_frames.append(self.valid_count.detach().clone())
        self.control_trace.append(self.env.action_manager.action.detach().clone())
        if self.windows is not None:
            self.windows.finish_control_step(set(done.nonzero().flatten().cpu().tolist()))
        self.ages[done.bool()] = 0
        super().finish_control_step(done)
        self.metrics.finish_control_step(done)

    def reset_windows(self):
        self.raw_history.clear()
        self.ages.zero_()
        self.episodes += 1
        for history in self.history.values():
            history.clear()
        self.clear_samples()

    def clear_samples(self):
        self.rows = defaultdict(list)
        self.seen = defaultdict(int)
        self.failures = []
        self.score_lookup = {}
        self.full_score_lookup = {}
        self.score_layout = {}

    def report(self, *, clear: bool = False) -> dict:
        values = torch.cat(self.errors).double().cpu().numpy() if self.errors else np.empty(0)
        reward_values = torch.cat(self.reward_errors).double().cpu().numpy() if self.reward_errors else np.empty(0)
        coordinate_sums = self.coordinate_error_sum.cpu().numpy()
        total_error = float(coordinate_sums.sum())
        bonuses = torch.cat(self.bonuses).double().cpu().numpy() if self.bonuses else np.empty(0)
        frames = torch.cat(self.scored_frames).cpu().numpy() if self.scored_frames else np.empty(0)
        result = {
            "scored_physical_transitions": len(values),
            "normalized_error_mean": float(values.mean()) if len(values) else None,
            "normalized_error_p95": float(np.quantile(values, 0.95)) if len(values) else None,
            "cap_fraction": float((reward_values >= self.cap).mean()) if len(reward_values) else None,
            "reward_error_scope": self.error_scope,
            "reward_error_mean": float(reward_values.mean()) if len(reward_values) else None,
            "reward_error_p95": float(np.quantile(reward_values, 0.95)) if len(reward_values) else None,
            "normalized_error_scope": "full_state",
            "group_full_loss_fraction": {
                name: float(coordinate_sums[ids].sum() / total_error) if total_error else None
                for name, ids in self.error_groups.items()
            },
            "group_full_loss_contribution": {
                name: float(coordinate_sums[ids].sum() / (len(values) * len(coordinate_sums))) if len(values) else None
                for name, ids in self.error_groups.items()
            },
            "bounded_bonus_mean": float(bonuses.mean()) if len(bonuses) else None,
            "unwarmed_control_fraction": float(np.mean(frames == 0)) if len(frames) else None,
            "verified_readonly_callbacks": self.verified_callbacks,
            "scored_worlds": self.env.num_envs,
            "recorded_world_ids": list(self.ids),
            "H10_reservoir_capacity": self.capacity,
        }
        if clear:
            self.errors, self.bonuses, self.scored_frames, self.reward_errors = [], [], [], []
            self.coordinate_error_sum.zero_()
        return result


def _score_audit_batches(indices, identities, layouts):
    groups = defaultdict(list)
    for index in indices:
        size, position = layouts[tuple(map(int, identities[index]))]
        if not 0 <= position < size:
            raise ValueError("Invalid online scoring layout.")
        groups[size].append((int(index), position))
    for size, pending in groups.items():
        while pending:
            rows, remainder, occupied = [], [], set()
            for item in pending:
                if item[1] in occupied:
                    remainder.append(item)
                else:
                    rows.append(item)
                    occupied.add(item[1])
            yield size, rows
            pending = remainder


@torch.no_grad()
def _audit_exported_scores(paths, collector):
    import json

    import h5py
    from torch.utils.data import default_collate

    from isaaclab_neural.data.opd import ContextDataset

    compared, maximum, full_maximum = 0, 0.0, 0.0
    for path in paths:
        with h5py.File(path, "r") as file:
            split = json.loads(file.attrs["opd_metadata"])["split"]
            identities = file["opd/query_identity"][:]
        if any(tuple(map(int, row)) not in collector.score_lookup for row in identities):
            raise RuntimeError("A saved H10 anchor did not have a causal, post-reset online error score.")
        dataset = ContextDataset(
            [path],
            split=split,
            physics_dt=collector.env.physics_dt,
            state_dim=87,
            joint_f_dim=43,
            contact_filter="solver_active",
        )
        indices = np.linspace(0, len(dataset) - 1, min(16, len(dataset)), dtype=int)
        for size, rows in _score_audit_batches(indices, identities, collector.score_layout):
            selected, positions = zip(*rows)
            positions = torch.tensor(positions, device=collector.device, dtype=torch.long)
            raw = default_collate([dataset[i] for i in selected])
            batch = {}
            for key, value in raw.items():
                value = value.to(collector.device)
                expanded = value[:1].expand(size, *value.shape[1:]).clone()
                expanded.index_copy_(0, positions, value)
                batch[key] = expanded
            solver = collector.adapter.solver
            solver.process_neural_model_inputs(batch)
            target = solver.convert_next_states_to_prediction(
                batch["states"][:, -1:], batch["next_states"][:, -1:], collector.env.physics_dt
            )[:, 0].index_select(0, positions)
            predicted = collector.model({key: value.clone() for key, value in batch.items()}, single_step=True)[:, 0]
            values, _ = bounded_model_error(
                predicted.index_select(0, positions), target, collector.reward_weights, cap=collector.cap
            )
            full_values, _ = bounded_model_error(
                predicted.index_select(0, positions), target, collector.weights, cap=collector.cap
            )
            expected = torch.tensor(
                [collector.score_lookup[tuple(map(int, identities[i]))] for i in selected], device=collector.device
            )
            full_expected = torch.tensor(
                [collector.full_score_lookup[tuple(map(int, identities[i]))] for i in selected], device=collector.device
            )
            torch.testing.assert_close(values, expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(full_values, full_expected, rtol=1e-4, atol=1e-4)
            maximum = max(maximum, float((values - expected).abs().max()))
            full_maximum = max(full_maximum, float((full_values - full_expected).abs().max()))
            compared += len(selected)
    return {
        "compared_exported_anchors": compared,
        "online_offline_error_abs_max": maximum,
        "full_error_abs_max": full_maximum,
        "reward_error_scope": collector.error_scope,
        "all_saved_identities_have_online_scores": True,
        "layout": "Original online batch size and row slots; unscored padding rows are not observations or labels",
    }


@torch.no_grad()
def _collect_frozen_policy(runner, wrapped, obs, collector, frozen, args, provenance):
    from isaaclab_neural.data.opd import file_sha256
    from isaaclab_neural.data.opd_sequence import (
        observed_sequence_rollouts,
        partition_sequence_records,
        write_sequence_shard,
    )
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json

    actor_before = {k: v.detach().clone() for k, v in runner.alg._raw_actor.state_dict().items()}

    def write_report(details):
        actor_unchanged = all(torch.equal(v, runner.alg._raw_actor.state_dict()[k]) for k, v in actor_before.items())
        model_unchanged = all(torch.equal(v, collector.model.state_dict()[k]) for k, v in frozen.items())
        if not actor_unchanged or not model_unchanged:
            raise RuntimeError("Frozen collection changed policy or dynamics weights.")
        _write_json(
            args.output / "report.json",
            {
                "provenance": provenance,
                "policy_source": args.policy_snapshot,
                "policy_unchanged": actor_unchanged,
                "dynamics_frozen": model_unchanged,
                "collector_ppo_updates": 0,
                "nerd_optimizer_updates": 0,
                **details,
            },
        )

    policy = runner.get_inference_policy(device=str(collector.device))
    torch.cuda.synchronize(collector.device)
    started = time.perf_counter()
    for _ in range(args.active_eval_steps):
        actions = policy(obs, stochastic_output=True)
        obs, _, dones, _ = wrapped.step(actions)
        collector.finish_control_step(dones)
        policy.reset(dones)
    torch.cuda.synchronize(collector.device)
    elapsed = time.perf_counter() - started
    metadata = {
        "physics_dt": collector.env.physics_dt,
        "protocol": provenance["protocol"],
        "source_kind": "mjwarp",
        "label_origin": "mjwarp",
        "split": "validation",
        "policy_hash": file_sha256(args.policy_snapshot),
        "dynamics_hash": file_sha256(args.dynamics),
        "pool": "rl_coverage_benchmark" if args.active_mode == "collect_benchmark" else "protected_active_validation",
        "collection_id": str(args.output.resolve()),
        "recorded_world_ids": list(collector.ids),
        "evaluation_only": args.active_mode == "collect_benchmark",
    }
    records = collector.windows.end_iteration()
    for row in records:
        row["oracle_states"] = row["student_states"].clone()
    sequence_meta = {
        **metadata,
        "decimation": collector.decimation,
        "history_steps": 10,
        "sequence_controls": 4,
        "iteration_id": 0,
        "initial_physical_state_protocol": "native_newton_state_v1",
        "initial_solver_state_protocol": "recorded_mj_qacc_warmstart",
        "history_origin": "physical_frames",
        "future_inputs_origin": "observed_physical_frames",
        "oracle_execution": {
            "kind": "observed_real_mj_trace_not_model_prediction_or_replay",
            "deterministic_mode": collector.env.cfg.sim.physics.deterministic_mode,
            "num_substeps": collector.env.cfg.sim.physics.num_substeps,
        },
    }
    if args.active_mode == "collect_train":
        validation_ids = set(range(collector.env.num_envs - collector.env.num_envs // 4, collector.env.num_envs))
        training_ids = set(range(collector.env.num_envs)) - validation_ids
        train_sequences, valid_sequences = partition_sequence_records(
            records, train_env_ids=training_ids, validation_env_ids=validation_ids
        )
        paths, sequence_shards = [], {}
        for split, ids, selected_records in (
            ("train", training_ids, train_sequences),
            ("validation", validation_ids, valid_sequences),
        ):
            paths.extend(
                collector.export(
                    args.output / "contexts" / split,
                    {**metadata, "pool": "frozen_active_training", "split": split},
                    env_ids=ids,
                    lengths={10},
                )
            )
            sequence_path = args.output / f"sequences_{split}.pt"
            write_sequence_shard(
                sequence_path,
                selected_records,
                metadata={
                    **sequence_meta,
                    "split": split,
                    "pool": "frozen_active_training",
                    "recorded_world_ids": sorted(ids),
                },
            )
            sequence_shards[split] = {
                "path": str(sequence_path.resolve()),
                "sha256": file_sha256(sequence_path),
                "records": len(selected_records),
                "full16_records": sum(len(row["oracle_states"]) == 16 for row in selected_records),
                "terminal_records": sum(row["terminal"] for row in selected_records),
            }
        audit = _audit_exported_scores(paths, collector)
        write_report(
            {
                "labelled_training_and_validation_shards": paths,
                "sequence_shards": sequence_shards,
                "score_audit": audit,
                "physical_transitions": collector.env.num_envs * args.active_eval_steps * collector.decimation,
                "label_provenance": "Observed MJ post-step states with frame/episode IDs at every history position",
            },
        )
        return
    paths = collector.export(args.output / "contexts", metadata, lengths={10})
    audit = _audit_exported_scores(paths, collector)
    sequence_path = args.output / "sequences.pt"
    write_sequence_shard(sequence_path, records, metadata=sequence_meta)
    force_dataset = None
    if any(row["end_reason"] == "horizon" for row in records):
        import json

        import h5py

        from isaaclab_neural.data.hdf5 import write_rollouts_to_hdf5

        force_path = args.output / "joint_f_trajectories.hdf5"
        if force_path.exists():
            raise FileExistsError(force_path)
        payload = observed_sequence_rollouts(records, sequence_meta)
        write_rollouts_to_hdf5(force_path, payload, env_name="G1-Native")
        with h5py.File(force_path, "r+") as file:
            file.attrs["opd_metadata"] = json.dumps(sequence_meta, allow_nan=False, sort_keys=True)
        force_dataset = {
            "path": str(force_path.resolve()),
            "sha256": file_sha256(force_path),
            "trajectories": len(payload["states"]),
            "role": "evaluation_only",
        }
    write_report(
        {
            "control_transitions": collector.env.num_envs * args.active_eval_steps,
            "joint_f_dataset": force_dataset,
            "collection_seconds": elapsed,
            "curiosity": collector.report(),
            "task_metrics": collector.metrics.report(),
            "score_audit": audit,
            "context_shards": paths,
            "sequence_shard": str(sequence_path.resolve()),
            "sequence_records": len(records),
            "label_provenance": "Actual pre-reset MJ states under recorded action sequences",
        },
    )


def run_active_collection(env, adapter, args) -> None:
    """Run bounded collector training or protected data collection with frozen NeRD scoring."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.data.opd import file_sha256
    from isaaclab_neural.eval.policy_benchmark import load_benchmark_config
    from isaaclab_neural.rl.rsl_rl.matched_mj import MatchedGeometryBridge, RolloutMetrics
    from isaaclab_neural.rl.rsl_rl.online_opd import _ppo_update, _write_json
    from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics
    from isaaclab_neural.utils.checkpoint import (
        load_checkpoint,
        reconstruct_model_from_checkpoint,
        save_opd_policy_snapshot,
    )
    from isaaclab_neural.utils.python_utils import set_random_seed

    base = env.unwrapped
    checkpoint = load_checkpoint(args.dynamics, device=base.device)
    model = reconstruct_model_from_checkpoint(checkpoint, adapter.solver, device=base.device)
    frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    cfg, _ = load_benchmark_config(source, args.policy_snapshot)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    random_init = getattr(args, "active_random_init", False)
    template_path = args.policy_snapshot
    if random_init:
        set_random_seed(args.active_policy_rng_seed)
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(base.device))
    start_iteration = 0
    if args.active_mode == "train":
        start_iteration = initialize_active_policy(
            runner, cfg, args.policy_snapshot, resume=args.active_resume, protocol=simulation_protocol(base)
        )
    elif not random_init:
        runner.alg.load(
            source, {"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False}, strict=True
        )
    set_random_seed(args.seed)
    obs, _ = wrapped.reset()
    bridge = None if args.active_mode == "collect_benchmark" else MatchedGeometryBridge(base, adapter)
    collector = DynamicsErrorRecorder(
        base,
        adapter,
        model,
        checkpoint["loss_weights"],
        cap=args.active_error_cap,
        seed=args.sampler_seed,
        record_envs=args.active_record_envs,
        capacity=args.active_capacity,
        record_sequences=args.active_mode in {"collect", "collect_train", "collect_benchmark"},
        error_scope=args.active_error_scope,
    )
    collector.metrics = RolloutMetrics(base)
    base.dynamics_error_observer = collector
    protocol = simulation_protocol(base)
    if random_init:
        random_path = args.output / "random_policy.pt"
        save_opd_policy_snapshot(
            random_path,
            runner,
            cfg,
            updates_completed=0,
            provenance={
                "protocol": protocol,
                "policy_seed": args.active_policy_rng_seed,
                "configuration_template": template_path,
                "historical_initialization": False,
            },
        )
        args.policy_snapshot = str(random_path.resolve())
    provenance = {
        "protocol": protocol,
        "policy_configuration_template": template_path,
        "random_untrained_actor": random_init,
        "source_checkpoint_iteration": None if random_init else source.get("iter"),
        "training_backend": "mjwarp",
        "collector_reward_weight": args.active_reward_weight,
        "dynamics_sha256": file_sha256(args.dynamics),
        "scope": "Frozen-model real-MJ collection with explicit reward profiles",
        "reward_profile": args.active_reward_audit,
        "num_envs": base.num_envs,
        "record_envs": min(args.active_record_envs, base.num_envs),
        "capacity_per_iteration": args.active_capacity,
        "scoring": "All worlds; only selected recording worlds are copied to host per physical frame",
        "env_seed": args.seed,
        "policy_rng_seed": args.active_policy_rng_seed,
        "policy_updates_at_start": start_iteration,
        "policy_resumed": args.active_resume,
        "stage_boundary": "Fresh MJ worlds and histories; actor, critic and PPO optimizer are continued when requested",
        "sampler_seed": args.sampler_seed,
        "collection_mode": args.active_mode,
        "policy_source": args.policy_snapshot,
        "policy_source_sha256": file_sha256(args.policy_snapshot),
        "collector_update_budget": args.active_updates if args.active_mode == "train" else 0,
        "control_steps_per_rollout": cfg["num_steps_per_env"],
        "error_cap": args.active_error_cap,
        "error_scope": collector.error_scope,
        "error_denominator": len(collector.weights),
        "reward_coordinate_indices": collector.reward_weights.nonzero().flatten().cpu().tolist(),
        "reward_error_weights": collector.reward_weights.cpu().tolist(),
        "full_state_loss_weights": collector.weights.cpu().tolist(),
    }
    diagnostics = PpoDiagnostics()
    updates, data_paths = [], []
    set_random_seed(args.active_policy_rng_seed)
    try:
        if args.active_mode in {"collect", "collect_train", "collect_benchmark"}:
            _collect_frozen_policy(runner, wrapped, obs, collector, frozen, args, provenance)
            return
        save_opd_policy_snapshot(
            args.output / f"policy_{start_iteration:04d}.pt",
            runner,
            cfg,
            updates_completed=start_iteration,
            provenance=provenance,
        )
        _write_json(args.output / "provenance.json", provenance)
        for iteration in range(start_iteration, start_iteration + args.active_updates):
            collector.begin_recording_iteration(iteration)
            policy_path = args.output / f"policy_{iteration:04d}.pt"
            torch.cuda.synchronize(base.device)
            torch.cuda.reset_peak_memory_stats(base.device)
            started = time.perf_counter()
            obs, row = _ppo_update(runner, obs, collector, iteration, diagnostics=diagnostics)
            torch.cuda.synchronize(base.device)
            elapsed = time.perf_counter() - started
            controls = base.num_envs * cfg["num_steps_per_env"]
            row["performance"] = {
                "collect_and_ppo_seconds": elapsed,
                "control_transitions": controls,
                "physics_transitions": controls * base.cfg.decimation,
                "control_transitions_per_second": controls / elapsed,
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(base.device) / 2**20,
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(base.device) / 2**20,
            }
            export_started = time.perf_counter()
            row["curiosity"] = collector.report(clear=True)
            row["task_metrics"] = collector.metrics.report(clear=True)
            row["ppo_consistency"] = diagnostics.consistency[-1]
            row["ppo_minibatches"] = copy.deepcopy(diagnostics.minibatches)
            free, total = torch.cuda.mem_get_info(base.device)
            row["performance"]["device_memory_in_use_mib"] = (total - free) / 2**20
            metadata = {
                "physics_dt": base.physics_dt,
                "protocol": protocol,
                "source_kind": "mjwarp",
                "label_origin": "mjwarp",
                "policy_hash": file_sha256(policy_path),
                "dynamics_hash": file_sha256(args.dynamics),
                "pool": "active_collection",
                "collection_id": str(args.output.resolve()),
                "collector_iteration": iteration,
                "reward_weight": args.active_reward_weight,
                "reward_profile": args.active_reward_profile,
                "reward_error_scope": collector.error_scope,
                "recorded_world_ids": list(collector.ids),
            }
            validation_ids = set(range(base.num_envs - base.num_envs // 4, base.num_envs))
            iteration_paths = []
            for split, ids in (("train", set(range(base.num_envs)) - validation_ids), ("validation", validation_ids)):
                iteration_paths.extend(
                    collector.export(
                        args.output / f"data_{iteration:04d}" / split,
                        {**metadata, "split": split},
                        env_ids=ids,
                        lengths={10},
                    )
                )
            data_paths.extend(iteration_paths)
            torch.save(
                {
                    "identities": torch.tensor(list(collector.score_lookup), dtype=torch.long),
                    "normalized_error": torch.tensor(list(collector.score_lookup.values())),
                    "reward_error_scope": collector.error_scope,
                    "full_normalized_error": torch.tensor(
                        [collector.full_score_lookup[key] for key in collector.score_lookup]
                    ),
                    "online_layout": torch.tensor(
                        [collector.score_layout[key] for key in collector.score_lookup], dtype=torch.long
                    ),
                },
                args.output / f"score_trace_{iteration:04d}.pt",
            )
            row["score_audit"] = _audit_exported_scores(iteration_paths, collector)
            collector.clear_samples()
            save_opd_policy_snapshot(
                args.output / f"policy_{iteration + 1:04d}.pt",
                runner,
                cfg,
                updates_completed=iteration + 1,
                provenance=provenance,
            )
            if iteration == start_iteration:
                torch.save(torch.stack(collector.control_trace).cpu(), args.output / "first_rollout_actions.pt")
            collector.control_trace.clear()
            row["performance"]["export_audit_and_checkpoint_seconds"] = time.perf_counter() - export_started
            _write_json(args.output / f"update_{iteration + 1:04d}.json", row)
            updates.append(row)
        set_random_seed(args.active_eval_seed)
        obs, _ = wrapped.reset()
        bridge.refresh()
        collector.reset_windows()
        collector.report(clear=True)
        collector.metrics = RolloutMetrics(base)
        policy = runner.get_inference_policy(device=str(base.device))
        set_random_seed(999)
        with torch.inference_mode():
            for _ in range(args.active_eval_steps):
                actions = policy(obs, stochastic_output=True)
                obs, _, dones, _ = wrapped.step(actions)
                collector.finish_control_step(dones)
                policy.reset(dones)
        evaluation = {"curiosity": collector.report(), "task_metrics": collector.metrics.report()}
        unchanged = all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())
        if not unchanged:
            raise RuntimeError("The collector modified its frozen NeRD scorer.")
        _write_json(
            args.output / "report.json",
            {
                "provenance": provenance,
                "updates": updates,
                "evaluation": evaluation,
                "labelled_training_and_validation_shards": data_paths,
                "dynamics_frozen": unchanged,
                "nerd_optimizer_updates": 0,
                "collector_ppo_updates": args.active_updates,
                "total_collector_ppo_updates": runner.current_learning_iteration,
                "last_policy": str((args.output / f"policy_{runner.current_learning_iteration:04d}.pt").resolve()),
                "eval_seed": args.active_eval_seed,
                "eval_control_steps": args.active_eval_steps,
                "reward": (
                    "Explicit base/error/base_error profile in provenance; frozen capped-log NeRD scoring; "
                    "not RND, ICM, ensemble information gain, or proof of data utility"
                ),
                "observation": (
                    "Existing G1 policy observation; no next state or current-action-dependent torque in policy input"
                ),
                "limitations": (
                    "Collector-only experiment; needs independent learnability, model-retention and downstream-policy "
                    "evaluation before any generalization claim"
                ),
            },
        )
    finally:
        collector.close()
        if bridge is not None:
            bridge.close()
        del base.dynamics_error_observer
