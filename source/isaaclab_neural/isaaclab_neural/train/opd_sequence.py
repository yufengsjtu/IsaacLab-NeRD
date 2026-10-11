# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated student learning and bounded complete-current-data OPD epochs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import torch

from isaaclab_neural.data.opd import CurrentEpochSampler


def _profile_training_records(shard: dict, minimum_count: int) -> list[dict]:
    meta = shard["metadata"]
    if meta["split"] != "train" or meta.get("evaluation_only", False) or meta["source_kind"] != "mjwarp":
        raise ValueError("Sequence profiling must not backpropagate on evaluation data or synthetic histories.")
    full = [row for row in shard["records"] if len(row["oracle_states"]) == 16]
    if len(full) < minimum_count:
        raise ValueError("Profiling requires distinct complete training sequences; no repeated/padded futures.")
    return full


def run_sequence_profile(env, args) -> None:
    """Profile gradient-connected short rollouts on training-only real sequences without updating weights."""
    import time
    from pathlib import Path

    from isaaclab_neural.data.opd import file_sha256
    from isaaclab_neural.data.opd_sequence import read_sequence_shard
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint, _write_json
    from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint

    config = json.loads(Path(args.active_fit_config).read_text())
    base = env.unwrapped
    paused = _pause_fingerprint(base, base.obs_buf)
    initial = load_checkpoint(args.dynamics, device=base.device)
    model = reconstruct_model_from_checkpoint(initial, base.neural_adapter.solver, device=base.device)
    frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if file_sha256(args.dynamics) != config["dynamics_sha256"]:
        raise ValueError("Profile initialization changed.")
    if any(size not in (64, 256) for size in config["batch_sizes"]) or config["repeats"] not in (1, 2, 3):
        raise ValueError("Use bounded logical batches64/256 and at most3 profiling repetitions.")
    result = {
        "gradient_contract": G1SequenceStudent.gradient_contract,
        "microbatch_worlds": base.num_envs,
        "optimizer_updates": 0,
        "cohorts": {},
        "timing_scope": "Measured worker workload; inspect concurrent GPU jobs before treating throughput as exclusive",
    }
    for name, entry in config["training_sequences"].items():
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Training sequence shard changed.")
        shard = read_sequence_shard(entry["path"], require_labels=True)
        meta = shard["metadata"]
        full = _profile_training_records(shard, max(config["batch_sizes"]))
        indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(config["sampling_seed"])).tolist()
        selected = [full[i] for i in indices[: max(config["batch_sizes"])]]
        student = G1SequenceStudent(base, base.neural_adapter.solver, model, meta, initial["loss_weights"])
        probe = selected[: base.num_envs]
        model.eval()
        with torch.no_grad():
            prediction, *_ = student.rollout(probe, max_physics_steps=16)
            native = student.native_reference(probe, 16)
            parity = bool(torch.allclose(prediction, native, rtol=1e-4, atol=1e-3))
            parity_error = float((prediction - native).abs().max())
        if not parity:
            raise RuntimeError(f"Profile native/student forward parity failed: {parity_error}.")
        _, _, _, intermediate = student.rollout(probe, max_physics_steps=16, retain_states=True)
        endpoint = intermediate[-1][: len(probe), 44:47].square().mean()
        earlier = torch.autograd.grad(endpoint, intermediate[0])[0]
        if not torch.isfinite(earlier).all() or not bool(earlier.abs().any()):
            raise RuntimeError("Endpoint loss cannot reach the first predicted state.")
        gradient = {
            "native_forward_abs_max": parity_error,
            "native_forward_parity": parity,
            "endpoint_to_first_state_gradient_norm": float(earlier.norm()),
            "frozen_geometry_finite_difference": _audit_frozen_geometry_gradient(student, probe),
        }
        del prediction, native, intermediate, endpoint, earlier
        model.train()
        warm = student.sequence_loss_terms(probe, max_physics_steps=4)["sequence"].mean()
        warm.backward()
        del warm
        model.zero_grad(set_to_none=True)
        rows = []
        for horizon in (4, 16):
            for batch_size in config["batch_sizes"]:
                durations = []
                peak_allocated = peak_reserved = 0
                loss_value = gradient_norm = 0.0
                for _ in range(config["repeats"]):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(base.device)
                    torch.cuda.reset_peak_memory_stats(base.device)
                    started = time.perf_counter()
                    loss_value = 0.0
                    for start in range(0, batch_size, base.num_envs):
                        batch = selected[start : min(start + base.num_envs, batch_size)]
                        loss = student.sequence_loss_terms(batch, max_physics_steps=horizon)["sequence"].mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite sequence profiling loss.")
                        (loss * (len(batch) / batch_size)).backward()
                        loss_value += float(loss.detach()) * len(batch) / batch_size
                    torch.cuda.synchronize(base.device)
                    durations.append(time.perf_counter() - started)
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(base.device))
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(base.device))
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    if not all(torch.isfinite(value).all() for value in gradients):
                        raise FloatingPointError("Non-finite rollout gradients.")
                    gradient_norm = float(torch.sqrt(sum(value.double().square().sum() for value in gradients)))
                rows.append(
                    {
                        "horizon": horizon,
                        "logical_batch": batch_size,
                        "seconds": durations,
                        "torch_peak_allocated_mib": peak_allocated / 2**20,
                        "torch_peak_reserved_mib": peak_reserved / 2**20,
                        "sequence_loss": loss_value,
                        "parameter_gradient_norm": gradient_norm,
                    }
                )
        model.zero_grad(set_to_none=True)
        result["cohorts"][name] = {
            "training_records": len(shard["records"]),
            "full16_records": len(full),
            "sample_indices": indices[: max(config["batch_sizes"])],
            "gradient_audit": gradient,
            "profiles": rows,
        }
        _write_json(args.output / f"{name}.json", result["cohorts"][name])
    if paused != _pause_fingerprint(base, base.obs_buf) or not all(
        torch.equal(value, model.state_dict()[key]) for key, value in frozen.items()
    ):
        raise RuntimeError("Read-only profiling changed live state or model parameters/RMS.")
    result["model_and_live_environment_unchanged"] = True
    _write_json(args.output / "result.json", result)


@dataclass(frozen=True)
class OpdEpochConfig:
    """Training budgets count logical optimizer steps, not gradient microbatches.

    Fixed-budget comparisons may continue after a finite quality rejection by
    disabling ``stop_on_validation_gate``; rejected epochs remain ineligible for
    selection. Non-finite values and changed frozen statistics always abort.
    """

    min_epochs: int = 2
    num_epochs: int = 4
    max_epochs: int = 8
    max_optimizer_steps: int = 32
    micro_batch_size: int = 4
    patience_epochs: int = 2
    min_delta_relative: float = 0.005
    min_delta_absolute: float = 1e-12
    max_grad_norm: float = 1.0
    stop_on_validation_gate: bool = True

    def validate(self, batches_per_epoch: int) -> None:
        if not 1 <= self.min_epochs <= self.num_epochs <= self.max_epochs:
            raise ValueError("Invalid epoch budgets.")
        if min(batches_per_epoch, self.micro_batch_size, self.patience_epochs) < 1:
            raise ValueError("Batch counts, microbatch size and patience must be positive.")
        if self.min_epochs * batches_per_epoch > self.max_optimizer_steps:
            raise ValueError("Optimizer budget cannot cover the minimum number of complete epochs.")
        if any(not math.isfinite(v) or v < 0 for v in (self.min_delta_relative, self.min_delta_absolute)):
            raise ValueError("Early-stop deltas must be finite and nonnegative.")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("Gradient clipping must be finite and positive.")


def sequence_pd_effort(
    states, targets, stiffness, damping, limits, *, num_q: int, pd_mode: str = "torch"
) -> torch.Tensor:
    """Compute differentiable free-root/actuated effort [N or N·m] for G1-style position PD.

    Joint positions/targets use [rad], joint velocities [rad/s]. Clipped regions
    have zero state derivative, exactly as the explicit torch clamp specifies.
    The default preserves legacy Torch rounding. ``newton_fma`` explicitly matches
    the FP32 fused stiffness operation in current Newton implicit-PD telemetry;
    this numerical mode does not alter gains, limits or the clipping derivative.
    """
    position = targets - states[..., 7:num_q]
    if pd_mode == "torch":
        effort = stiffness * position - damping * states[..., num_q + 6 :]
    elif pd_mode == "newton_fma" and states.dtype == torch.float32:
        effort = torch.addcmul(damping * (-states[..., num_q + 6 :]), stiffness, position)
    else:
        raise ValueError("Unknown PD mode or non-FP32 input for Newton FMA replay.")
    effort = torch.clamp(effort, -limits, limits)
    return torch.cat((torch.zeros_like(states[..., :6]), effort), dim=-1)


def sequence_root_correction_loss(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    *,
    height_scale_m: float,
    velocity_scale_mps: float,
    upright_scale: float,
) -> torch.Tensor:
    """Balance physical root corrections without signed-error cancellation or hand-coordinate dominance.

    Args:
        predicted: Predicted raw G1 states in SI units, shape [N, T, 87], with XYZW quaternion.
        truth: Actual MJ future states in SI units, shape [N, T, 87]; always treated as fixed labels.
        height_scale_m: Fixed positive root-height normalization scale [m].
        velocity_scale_mps: Fixed positive root-linear-velocity normalization scale [m/s].
        upright_scale: Fixed positive vertical body-axis projection scale.

    Returns:
        Mean dimensionless correction loss per sequence, shape [N]. Height, linear velocity
        and upright projection receive equal group weights across actual, unpadded future frames.
    """
    if predicted.ndim != 3 or predicted.shape != truth.shape or predicted.shape[-1] != 87:
        raise ValueError("Root rollout correction requires matching real G1 future-state sequences.")
    if any(not math.isfinite(value) or value <= 0 for value in (height_scale_m, velocity_scale_mps, upright_scale)):
        raise ValueError("Root correction scales must be finite and positive.")
    if not torch.isfinite(predicted).all() or not torch.isfinite(truth).all():
        raise ValueError("Root correction cannot consume non-finite predicted states or MJ labels.")
    truth = truth.detach()
    height = ((predicted[..., 2] - truth[..., 2]) / height_scale_m).square()
    velocity = ((predicted[..., 44:47] - truth[..., 44:47]) / velocity_scale_mps).square().mean(-1)
    up = []
    for states in (predicted, truth):
        quaternion = torch.nn.functional.normalize(states[..., 3:7], dim=-1)
        up.append(1 - 2 * quaternion[..., :2].square().sum(-1))
    upright = ((up[0] - up[1]) / upright_scale).square()
    return ((height + velocity + upright) / 3).mean(-1)


def sequence_wrap_angles(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Wrap periodic angles [rad] out of place with the runtime's exact floor convention."""
    mask = torch.as_tensor(mask, device=states.device, dtype=torch.bool)
    delta = torch.floor((states + math.pi) / (2 * math.pi)) * (2 * math.pi)
    return torch.where(mask, states - delta, states)


def _on_torch_stream(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        import warp as wp

        if self.device.type == "cuda":
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
                return method(self, *args, **kwargs)
        return method(self, *args, **kwargs)

    return call


class G1SequenceStudent:
    """Train a G1 student through state/PD/history paths with detached, refreshed contact geometry.

    This is not fully differentiable collision physics. Newton FK, contact
    detection/packing and their geometry features are explicitly stop-gradient.
    Predicted generalized state, root-frame transforms, PD and neural history
    remain connected across physical steps. No live simulation state is written.

    The default ``pd_mode="torch"`` preserves historical evaluation and training.
    Explicit ``newton_fma`` retains the recorded anchor effort and uses runtime-matched
    FP32 implicit PD on predicted future states. Matching the recording's batch layout
    is also necessary when auditing bitwise autoregressive replay. ``deterministic_geometry=None``
    inherits the task's collision options without reducing its contact capacity; the default True
    preserves historical flat studies. Randomized per-world physical properties are replayed into
    private model arrays rather than mutating the live simulator.
    """

    gradient_contract = {
        "connected": ["model_parameters", "predicted_state", "root_frame", "PD", "state_history"],
        "stop_gradient": ["Newton_FK", "collision_detection", "contact_packing", "contact_geometry_features"],
        "full_physics_differentiability": False,
    }

    def __init__(
        self,
        env,
        solver,
        model: torch.nn.Module,
        metadata: dict,
        loss_weights: torch.Tensor,
        *,
        pd_mode: str = "torch",
        deterministic_geometry: bool | None = True,
    ):
        import newton

        from isaaclab_neural.contacts import NewtonContactAdapter
        from isaaclab_neural.generate.opd import simulation_protocol

        if metadata["protocol"] != simulation_protocol(env):
            raise ValueError("Student and oracle physical/actuation protocols differ.")
        if solver.state_dim != 87 or solver.dof_q_per_env != 44 or solver.states_embedding_type != "identical":
            raise ValueError("P2 student currently supports the 37-joint G1 identical-state layout only.")
        if solver.prediction_type != "relative" or solver.orientation_prediction_parameterization != "quaternion":
            raise ValueError("P2 requires the existing relative/quaternion G1 checkpoint.")
        if pd_mode not in {"torch", "newton_fma"}:
            raise ValueError("Student PD mode must preserve legacy Torch or explicitly match Newton FMA.")
        self.solver, self.model, self.metadata = solver, model, metadata
        self.pd_mode = pd_mode
        self.device = torch.device(env.device)
        self.num_envs = env.num_envs
        self.physics_context = None
        if metadata["protocol"].get("physics_parameter_scope") == "per_world_recorded":
            from isaaclab_neural.data.physics_context import WorldPhysicsContext

            self.physics_context = WorldPhysicsContext.isolated(solver.model, self.num_envs)
        self.geometry_model = self.physics_context.model if self.physics_context is not None else solver.model
        self.state = self.geometry_model.state()
        pipeline_args = env.sim.physics_manager._collision_cfg.to_pipeline_args()
        if deterministic_geometry is not None:
            pipeline_args["deterministic"] = deterministic_geometry
        self.pipeline = newton.CollisionPipeline(self.geometry_model, **pipeline_args)
        self.solver_cfg = copy.deepcopy(env.cfg.sim.physics.solver_cfg)
        self.contacts = self.pipeline.contacts()
        self.contact_adapter = NewtonContactAdapter(
            self.geometry_model,
            num_contacts_per_env=solver.num_contacts_per_env,
            device=str(self.device),
            packing_policy="body_round_robin_pair_atomic",
            contact_representation="contact_tokens",
            max_contact_tokens=solver.max_contact_tokens,
            contact_filter=solver.contact_filter,
        )
        protocol = metadata["protocol"]
        self.kp, self.kd, self.limits = [
            torch.tensor(protocol[key], device=self.device) for key in ("stiffness", "damping", "effort_limits")
        ]
        self.loss_weights = loss_weights.detach().to(self.device).clone()
        if self.loss_weights.shape != (87,):
            raise ValueError("Unexpected frozen loss-weight layout.")
        self.geometry_refreshes = 0
        if self.device.type == "cuda":
            import warp as wp

            wp.synchronize_device(solver.device)

    def _history(self, records: list[dict]) -> dict[str, torch.Tensor]:
        from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_VALUE_FIELDS
        from isaaclab_neural.contacts.native_pair_projector import project_native_pairs

        if self.physics_context is not None:
            from isaaclab_neural.data.physics_context import WORLD_PHYSICS_FIELDS

            if any(set(row.get("physics", {})) != set(WORLD_PHYSICS_FIELDS) for row in records):
                raise ValueError("Randomized model targets require their actual per-world physical parameters.")
            parameters = {key: torch.stack([row["physics"][key] for row in records]) for key in WORLD_PHYSICS_FIELDS}
            self.physics_context.restore(torch.arange(len(records), device=self.device), parameters)
        frames = [frame for row in records for frame in row["history"]]
        h = self.metadata["history_steps"]
        if any(len(row["history"]) != h for row in records):
            raise ValueError("Student cannot pad or truncate the initial history.")
        native = {
            key: torch.cat([frame["native"][key] for frame in frames]).to(self.device)
            for key in NATIVE_CONTACT_VALUE_FIELDS
        }
        counts = torch.tensor([len(frame["native"]["world_ids"]) for frame in frames], device=self.device)
        projected = project_native_pairs(
            native,
            num_frames=len(frames),
            representation="contact_tokens",
            packing_policy="body_round_robin_pair_atomic",
            frame_ids=torch.arange(len(frames), device=self.device).repeat_interleave(counts),
            max_contact_tokens=self.solver.max_contact_tokens,
            contact_filter=self.solver.contact_filter,
        )
        history = {
            key: torch.stack([torch.stack([f[key] for f in row["history"]]) for row in records]).to(self.device)
            for key in ("states", "joint_f", "root_body_q", "gravity_dir")
        }
        history["contact_tokens"] = projected["contact_tokens"].reshape(
            len(records), h, self.solver.max_contact_tokens, -1
        )
        history["contact_token_overflow"] = projected["contact_token_overflow"].reshape(len(records), h)
        torch.testing.assert_close(history["root_body_q"], history["states"][..., :7], rtol=0, atol=1e-6)
        return history

    @_on_torch_stream
    def _geometry(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        import newton
        import warp as wp

        with torch.no_grad():
            self.solver._assign_states_from_torch(self.state, state.detach().contiguous())
            newton.eval_fk(self.geometry_model, self.state.joint_q, self.state.joint_qd, self.state)
            root = wp.to_torch(self.state.body_q).index_select(0, self.solver.root_body_ids)
            torch.testing.assert_close(root, state.detach()[:, :7], rtol=0, atol=1e-5)
            self.pipeline.collide(self.state, self.contacts)
            self.contact_adapter.update(self.contacts, self.state)
            inputs = {key: value.detach().clone() for key, value in self.contact_adapter.to_neural_inputs().items()}
            inputs["root_body_q"] = root.detach().clone()
        self.geometry_refreshes += 1
        return inputs

    def _forward(self, history):
        prepared = self.solver.process_neural_model_inputs({key: value.clone() for key, value in history.items()})
        prediction = self.model(
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in prepared.items()},
            single_step=True,
        ).squeeze(1)
        decoded = self.solver.convert_prediction_to_next_states(
            prepared["states"][:, -1], prediction, self.metadata["physics_dt"]
        )
        world = self.solver.convert_states_back_to_world(prepared["root_body_q"], decoded)
        return sequence_wrap_angles(world, self.solver.is_continuous_dof), prediction, prepared

    def _advance_history(self, history, state, target):
        geometry = self._geometry(state)
        root = state[:, :7] + (geometry.pop("root_body_q") - state[:, :7]).detach()
        frame = {
            "states": state,
            "root_body_q": root,
            "joint_f": sequence_pd_effort(state, target, self.kp, self.kd, self.limits, num_q=44, pd_mode=self.pd_mode),
            "gravity_dir": history["gravity_dir"][:, -1],
            **geometry,
        }
        return {key: torch.cat((value[:, 1:], frame[key][:, None]), dim=1) for key, value in history.items()}

    @torch.no_grad()
    @_on_torch_stream
    def native_reference(self, records: list[dict], steps: int) -> torch.Tensor:
        """Run the independent inference-only solver in scratch buffers at the learner's numerical layout."""
        import newton
        import warp as wp

        from isaaclab_neural.solvers import create_neural_solver

        padded = records + [records[-1]] * (self.num_envs - len(records))
        history = self._history(padded)
        state_a, state_b = self.geometry_model.state(), self.geometry_model.state()
        control = self.geometry_model.control()
        runtime = create_neural_solver(
            self.solver_cfg,
            model=self.geometry_model,
            contacts=self.contacts,
            neural_model=self.model,
            num_contacts_per_env=self.solver.num_contacts_per_env,
            contact_mode="newton_native",
            contact_adapter=self.contact_adapter,
            contact_representation="contact_tokens",
            max_contact_tokens=self.solver.max_contact_tokens,
            use_feature_cache=False,
            use_kv_cache=False,
        )
        states = history["states"][:, -1].clone()
        runtime._assign_states_from_torch(state_a, states)
        newton.eval_fk(self.geometry_model, state_a.joint_q, state_a.joint_qd, state_a)
        target0 = torch.stack([r["action_targets"][0] for r in padded]).to(self.device)
        initial_pd = sequence_pd_effort(states, target0, self.kp, self.kd, self.limits, num_q=44, pd_mode=self.pd_mode)
        if self.pd_mode == "torch":
            history["joint_f"] = torch.cat((history["joint_f"][:, :-1], initial_pd[:, None]), dim=1)
        else:
            torch.testing.assert_close(initial_pd, history["joint_f"][:, -1], rtol=0, atol=1e-3)
        runtime.preload_states_history(history)
        inputs = runtime.get_neural_model_inputs()
        cur = inputs["states"][:, -1].clone()
        root = inputs["root_body_q"].clone()
        first = runtime._model_forward(inputs).squeeze(1)
        states = runtime.convert_states_back_to_world(
            root, runtime.convert_prediction_to_next_states(cur, first, self.metadata["physics_dt"])
        )
        runtime.wrap2PI(states)
        runtime._assign_states_from_torch(state_a, states)
        newton.eval_fk(self.geometry_model, state_a.joint_q, state_a.joint_qd, state_a)
        outputs = [states[: len(records)].clone()]
        for step in range(1, steps):
            target = torch.stack([r["action_targets"][step // self.metadata["decimation"]] for r in padded]).to(
                self.device
            )
            pd = sequence_pd_effort(states, target, self.kp, self.kd, self.limits, num_q=44, pd_mode=self.pd_mode)
            control.joint_f.assign(wp.from_torch(pd.contiguous().flatten()))
            self.pipeline.collide(state_a, self.contacts)
            runtime.step(state_a, state_b, control, self.contacts, self.metadata["physics_dt"])
            states = torch.empty_like(states)
            runtime._acquire_states_to_torch(state_b, states)
            runtime.wrap2PI(states)
            outputs.append(states[: len(records)].clone())
            state_a, state_b = state_b, state_a
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def recorded_state_audit(self, records: list[dict], steps: int) -> list[float]:
        """Diagnose one-step parity along recorded states; this path is never used for training."""
        padded = records + [records[-1]] * (self.num_envs - len(records))
        history = self._history(padded)
        errors = []
        for step in range(steps):
            predicted, _, _ = self._forward(history)
            truth = torch.stack([r["student_states"][step] for r in padded]).to(self.device)
            errors.append(float((predicted[: len(records)] - truth[: len(records)]).abs().max()))
            if step + 1 < steps:
                target = torch.stack(
                    [r["action_targets"][(step + 1) // self.metadata["decimation"]] for r in padded]
                ).to(self.device)
                history = self._advance_history(history, truth, target)
        return errors

    def rollout(self, records: list[dict], *, max_physics_steps: int | None = None, retain_states: bool = False):
        if not records or len(records) > self.num_envs:
            raise ValueError("Student microbatch exceeds its isolated geometry world count.")
        lengths = {len(row["actions"]) * self.metadata["decimation"] for row in records}
        if len(lengths) != 1:
            raise ValueError("Group student microbatches by real future length; do not invent actions.")
        steps = lengths.pop()
        if max_physics_steps is not None:
            steps = min(steps, max_physics_steps)
        padded = records + [records[-1]] * (self.num_envs - len(records))
        history = self._history(padded)
        targets = torch.stack([row["action_targets"] for row in padded]).to(self.device)
        pd = sequence_pd_effort(
            history["states"][:, -1], targets[:, 0], self.kp, self.kd, self.limits, num_q=44, pd_mode=self.pd_mode
        )
        torch.testing.assert_close(pd.detach(), history["joint_f"][:, -1], rtol=0, atol=1e-3)
        if self.pd_mode == "torch":
            history["joint_f"] = torch.cat((history["joint_f"][:, :-1], pd[:, None]), dim=1)
        states, first_prediction, first_prepared = [], None, None
        for step in range(steps):
            state, prediction, prepared = self._forward(history)
            if retain_states and state.requires_grad:
                state.retain_grad()
            states.append(state)
            if step == 0:
                first_prediction, first_prepared = prediction, prepared
            if step + 1 == steps:
                break
            history = self._advance_history(history, state, targets[:, (step + 1) // self.metadata["decimation"]])
        return torch.stack(states, dim=1)[: len(records)], first_prediction[: len(records)], first_prepared, states

    def sequence_loss(
        self, records: list[dict], *, sequence_weight: float = 1.0, max_physics_steps: int | None = None
    ) -> torch.Tensor:
        terms = self.sequence_loss_terms(records, max_physics_steps=max_physics_steps)
        return terms["local"] + sequence_weight * terms["sequence"]

    def sequence_loss_terms(
        self,
        records: list[dict],
        *,
        max_physics_steps: int | None = None,
        root_correction_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return local/sequence losses with optional separately scaled real-MJ root correction.

        Existing local and full-state sequence weights remain unchanged. Root scales are
        explicit physical units accepted by :func:`sequence_root_correction_loss`.
        """
        predicted, first, prepared, _ = self.rollout(records, max_physics_steps=max_physics_steps)
        count, steps = predicted.shape[:2]
        truth = torch.stack([row["oracle_states"][:steps] for row in records]).to(self.device).detach()
        root = prepared["root_body_q"][:count, -1:]
        initial = prepared["states"][:count, -1:]
        truth_first = self.solver._convert_states_w2b(root, truth[:, :1], translation_only=False)
        target = self.solver.convert_next_states_to_prediction(
            initial.detach(), truth_first, self.metadata["physics_dt"]
        )[:, 0]
        local = ((first - target.detach()) * self.loss_weights).square().mean(-1)
        anchor = root.detach().expand(count, steps, 7)
        p = self.solver._convert_states_w2b(anchor, predicted, translation_only=False)
        t = self.solver._convert_states_w2b(anchor, truth, translation_only=False)
        pq = torch.nn.functional.normalize(p[..., 3:7], dim=-1)
        tq = torch.nn.functional.normalize(t[..., 3:7], dim=-1)
        tq = tq * torch.where((pq * tq).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        joints = p[..., 7:44] - t[..., 7:44]
        delta = torch.cat(
            (p[..., :3] - t[..., :3], pq - tq, torch.atan2(joints.sin(), joints.cos()), p[..., 44:] - t[..., 44:]),
            dim=-1,
        )
        sequence = (delta * self.loss_weights).square().mean(dim=(-1, -2))
        result = {"local": local, "sequence": sequence}
        if root_correction_scales is not None:
            result["root_sequence"] = sequence_root_correction_loss(predicted, truth, **root_correction_scales)
        return result

    def reference_loss(self, samples: list[dict]) -> torch.Tensor:
        data = {
            key: torch.stack([sample[key] for sample in samples]).to(self.device)
            for key in ("states", "next_states", "root_body_q", "joint_f", "gravity_dir", "contact_tokens")
        }
        prepared = self.solver.process_neural_model_inputs(data)
        target = self.solver.convert_next_states_to_prediction(
            prepared["states"], prepared["next_states"], self.metadata["physics_dt"]
        )[:, -1]
        prediction = self.model(
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in prepared.items()},
            single_step=True,
        )[:, 0]
        return ((prediction - target.detach()) * self.loss_weights).square().mean(-1)


def _audit_frozen_geometry_gradient(student, records):
    padded = records + [records[-1]] * (student.num_envs - len(records))
    history = student._history(padded)

    def score(delta):
        state = history["states"][:, -1]
        state = torch.cat((state[:, :44], state[:, 44:45] + delta, state[:, 45:]), dim=-1)
        current = {key: value.clone() for key, value in history.items()}
        current["states"] = torch.cat((history["states"][:, :-1], state[:, None]), dim=1)
        return student._forward(current)[0][: len(records), 44].mean()

    delta = torch.zeros((), device=student.device, requires_grad=True)
    derivative = torch.autograd.grad(score(delta), delta)[0]
    epsilon = 1e-3
    with torch.no_grad():
        finite_difference = (score(epsilon) - score(-epsilon)) / (2 * epsilon)
    error = float((derivative - finite_difference).abs() / finite_difference.abs().clamp(min=1.0))
    if not math.isfinite(error) or error > 0.01:
        raise RuntimeError(f"Frozen-geometry Torch transition derivative mismatch: {error}.")
    return {
        "autograd": float(derivative),
        "finite_difference": float(finite_difference),
        "scaled_absolute_error": error,
        "epsilon_mps": epsilon,
        "scope": "Torch state/root-frame/decoder path with fixed contact features, not a full collision Jacobian",
    }


def _training_fingerprint(model, optimizer) -> dict[str, str]:
    def tensor_digest(digest, name, value):
        digest.update(json.dumps([str(name), str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())

    model_digest, optimizer_digest = hashlib.sha256(), hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor_digest(model_digest, key, value)
    state = optimizer.state_dict()
    optimizer_digest.update(json.dumps(state["param_groups"], sort_keys=True, allow_nan=False).encode())
    for parameter, values in sorted(state["state"].items()):
        for key, value in sorted(values.items()):
            name = f"{parameter}:{key}"
            if isinstance(value, torch.Tensor):
                tensor_digest(optimizer_digest, name, value)
            else:
                optimizer_digest.update(json.dumps([name, value], sort_keys=True, allow_nan=False).encode())
    return {"model_sha256": model_digest.hexdigest(), "optimizer_sha256": optimizer_digest.hexdigest()}


def _snapshot(model, optimizer):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def _restore(model, optimizer, snapshot):
    model.load_state_dict(snapshot[0], strict=True)
    optimizer.load_state_dict(snapshot[1])
    optimizer.zero_grad(set_to_none=True)


def _validation(validate_fn):
    with torch.no_grad():
        metrics = validate_fn()
    score = float(metrics["loss"])
    if not math.isfinite(score):
        raise FloatingPointError("Non-finite OPD validation loss.")
    return score, bool(metrics.get("admissible", True)), metrics


def _logical_update(model, optimizer, batch, loss_fn, config):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for source in ("current", "reference", "replay"):
        indices = batch[source]
        weight = batch["source_weights"][source]
        if not indices or weight == 0:
            continue
        for start in range(0, len(indices), config.micro_batch_size):
            selected = indices[start : start + config.micro_batch_size]
            losses = loss_fn(source, selected)
            if losses.shape != (len(selected),):
                raise ValueError("OPD loss must contain one scalar per real anchor, excluding padding.")
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Non-finite OPD training loss.")
            loss = losses.sum() * (weight / len(indices))
            loss.backward()
            total += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": total,
        "gradient_before_clip": float(norm),
        "logical_batch_size": sum(len(batch[key]) for key in ("current", "reference", "replay")),
    }


def _assert_frozen_statistics(model, state):
    current = model.state_dict()
    for key, value in state.items():
        if key.startswith(("input_rms.", "output_rms.", "body_net_contact_force_rms.")):
            if not torch.equal(value, current[key]):
                raise RuntimeError("OPD changed frozen model statistics.")


def _fit_epochs(model, optimizer, sampler, loss_fn, validate_fn, config, baseline):
    score, _, metrics = _validation(validate_fn)
    significant = best_score = score
    best = baseline
    best_epoch = executed = stale = 0
    rows, coverage = [], []
    reason = "max_epochs"
    for epoch in range(1, config.max_epochs + 1):
        if executed + len(sampler) > config.max_optimizer_steps:
            reason = "optimizer_budget"
            break
        seen, steps = [], []
        for batch in sampler:
            seen.extend(batch["current"])
            steps.append(_logical_update(model, optimizer, batch, loss_fn, config))
            executed += 1
        if sorted(seen) != list(range(sampler.current_count)):
            raise RuntimeError("OPD epoch omitted or duplicated current rows.")
        coverage.append(len(seen))
        score, admissible, validation = _validation(validate_fn)
        _assert_frozen_statistics(model, baseline[0])
        threshold = max(config.min_delta_absolute, config.min_delta_relative * abs(significant))
        improved = admissible and significant - score > threshold
        if improved:
            significant, stale = score, 0
        else:
            stale += 1
        if admissible and epoch >= config.min_epochs and score < best_score:
            best_score, best_epoch, best = score, epoch, _snapshot(model, optimizer)
        rows.append(
            {
                "epoch": epoch,
                "steps": steps,
                "validation": validation,
                "admissible": admissible,
                "significant_improvement": improved,
            }
        )
        if not admissible and config.stop_on_validation_gate:
            reason = "validation_gate"
            break
        if epoch >= config.min_epochs and stale >= config.patience_epochs:
            reason = "validation_patience"
            break
        if epoch >= config.num_epochs and not improved:
            reason = "default_epoch_budget"
            break
    _restore(model, optimizer, best)
    _assert_frozen_statistics(model, baseline[0])
    return {
        "epochs_completed": len(rows),
        "optimizer_steps_executed": executed,
        "selected_epoch": best_epoch,
        "accepted": best_epoch > 0,
        "selected_validation_loss": best_score,
        "baseline_validation": metrics,
        "coverage_per_epoch": coverage,
        "stop_reason": reason,
        "epochs": rows,
    }


def fit_opd_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: CurrentEpochSampler,
    loss_fn: Callable,
    validate_fn: Callable,
    config: OpdEpochConfig,
) -> dict:
    """Train whole epochs, accumulate weighted microbatches and restore the best model/optimizer pair.

    ``loss_fn(source, indices)`` returns per-anchor losses. Validation is fixed for
    the entire call and never participates in backpropagation. Safety exceptions
    roll back the whole attempt, while a finite quality stop may retain an earlier
    admissible epoch. CPU/CUDA torch RNG is preserved for the paused policy.
    """
    config.validate(len(sampler))
    model_ids = {id(p) for p in model.parameters()}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if not optimizer_ids <= model_ids or not trainable_ids <= optimizer_ids:
        raise ValueError("OPD optimizer must own exactly this student's trainable parameters, not policy parameters.")
    baseline = _snapshot(model, optimizer)
    original_mode = model.training
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            return _fit_epochs(model, optimizer, sampler, loss_fn, validate_fn, config, baseline)
    except BaseException:
        _restore(model, optimizer, baseline)
        raise
    finally:
        model.train(original_mode)


def run_sequence_fit_probe(env, adapter, args) -> None:
    """Audit gradients and fit bounded fixed P1 data; never promote into the live environment."""
    from dataclasses import asdict

    from isaaclab_neural.data.opd import ContextDataset, file_sha256
    from isaaclab_neural.data.opd_sequence import partition_sequence_records, read_sequence_shard
    from isaaclab_neural.eval.dynamics_consistency import state_errors, summarize_errors
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint, _write_json
    from isaaclab_neural.train.trainers import validate_finetune_checkpoint
    from isaaclab_neural.utils.checkpoint import (
        get_cfg_from_checkpoint,
        load_checkpoint,
        reconstruct_model_from_checkpoint,
        save_checkpoint,
    )

    base = env.unwrapped
    before = _pause_fingerprint(base, base.obs_buf)
    shard = read_sequence_shard(args.queries[0], require_labels=True)
    if shard["metadata"]["split"] not in {"train", "diagnostic"}:
        raise ValueError("Do not train on validation/test shards.")
    if shard["metadata"]["dynamics_hash"] != file_sha256(args.dynamics):
        raise ValueError("The fixed-data probe must start from the collection model.")
    train, valid = partition_sequence_records(
        shard["records"],
        train_env_ids=set(args.sequence_train_env_ids),
        validation_env_ids=set(args.sequence_validation_env_ids),
    )
    if len(train) < 4 or not valid or len(train) + len(valid) > 128:
        raise ValueError(
            "Fixed-data P2 probe requires at least4 train records, held-out worlds, and at most128 total records."
        )
    solver = base.neural_adapter.solver
    checkpoint = load_checkpoint(args.dynamics, device=base.device)
    model = reconstruct_model_from_checkpoint(checkpoint, solver, device=base.device)
    validate_finetune_checkpoint(checkpoint, model, update_statistics=False)
    model.eval()
    student = G1SequenceStudent(base, solver, model, shard["metadata"], checkpoint["loss_weights"])
    reference = ContextDataset(
        args.sequence_reference,
        split="train",
        physics_dt=base.physics_dt,
        state_dim=solver.state_dim,
        joint_f_dim=solver.joint_f_dim,
        contact_filter=solver.contact_filter,
        max_contact_tokens=solver.max_contact_tokens,
    )
    if set(reference.lengths) != {10}:
        raise ValueError("This probe requires genuine H10 reference anchors.")
    reference_validation = None
    if args.sequence_reference_validation:
        reference_validation = ContextDataset(
            args.sequence_reference_validation,
            split="validation",
            physics_dt=base.physics_dt,
            state_dim=solver.state_dim,
            joint_f_dim=solver.joint_f_dim,
            contact_filter=solver.contact_filter,
            max_contact_tokens=solver.max_contact_tokens,
        )
        if set(reference_validation.lengths) != {10} or reference.families & reference_validation.families:
            raise ValueError("Reference validation requires H10 anchors and disjoint episode families.")

    def groups(records, size):
        for horizon in sorted({len(r["actions"]) for r in records}):
            selected = [r for r in records if len(r["actions"]) == horizon]
            for start in range(0, len(selected), size):
                yield selected[start : start + size]

    @torch.no_grad()
    def evaluate(records):
        losses, local, sequence = [], [], []
        errors = {key: {} for key in ("metrics", "one_step_metrics", "endpoint_metrics")}
        for batch in groups(records, args.sequence_micro_batch_size):
            terms = student.sequence_loss_terms(batch, max_physics_steps=args.sequence_fit_physics_steps)
            local.append(terms["local"].cpu())
            sequence.append(terms["sequence"].cpu())
            losses.append((terms["local"] + terms["sequence"]).cpu())
            prediction, _, _, _ = student.rollout(batch, max_physics_steps=args.sequence_fit_physics_steps)
            truth = torch.stack([r["oracle_states"][: prediction.shape[1]] for r in batch]).to(base.device)
            views = (
                ("metrics", prediction, truth),
                ("one_step_metrics", prediction[:, 0], truth[:, 0]),
                ("endpoint_metrics", prediction[:, -1], truth[:, -1]),
            )
            for group, predicted, target in views:
                for name, value in state_errors(predicted, target, 44).items():
                    errors[group].setdefault(name, []).append(value.cpu().flatten())
        return {
            "loss": float(torch.cat(losses).mean()),
            "local_loss": float(torch.cat(local).mean()),
            "sequence_loss": float(torch.cat(sequence).mean()),
            "admissible": True,
            **{key: summarize_errors({k: torch.cat(v) for k, v in values.items()}) for key, values in errors.items()},
        }

    @torch.no_grad()
    def evaluate_reference(dataset):
        if dataset is None:
            return None
        losses = []
        for start in range(0, len(dataset), args.sequence_micro_batch_size):
            samples = [dataset[i] for i in range(start, min(start + args.sequence_micro_batch_size, len(dataset)))]
            losses.append(student.reference_loss(samples).cpu())
        values = torch.cat(losses)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite reference diagnostic loss.")
        return {"loss": float(values.mean()), "p95_loss": float(values.quantile(0.95)), "records": len(dataset)}

    longest = max(len(row["actions"]) for row in train)
    first_batch = [r for r in train if len(r["actions"]) == longest][: args.sequence_micro_batch_size]
    audit_steps = min(longest * shard["metadata"]["decimation"], max(4, args.sequence_fit_physics_steps))
    with torch.no_grad():
        prediction, _, _, _ = student.rollout(first_batch, max_physics_steps=audit_steps)
        old = torch.stack([r["student_states"][:audit_steps] for r in first_batch]).to(base.device)
        forward_error = float((prediction[:, 0] - old[:, 0]).abs().max())
        torch.testing.assert_close(prediction[:, 0], old[:, 0], rtol=1e-5, atol=1e-4)
        full_forward_error = float((prediction - old).abs().max())
        native_reference = student.native_reference(first_batch, audit_steps)
        native_forward_error = float((prediction - native_reference).abs().max())
        forward_parity = bool(torch.allclose(prediction, native_reference, rtol=1e-4, atol=1e-3))
    _, _, _, intermediate = student.rollout(first_batch, max_physics_steps=audit_steps, retain_states=True)
    last_loss = intermediate[-1][: len(first_batch), 44:47].square().mean()
    earlier = torch.autograd.grad(last_loss, intermediate[0], retain_graph=True)[0]
    gradients = torch.autograd.grad(last_loss, list(model.parameters()), allow_unused=True)
    norm = sum(float(g.detach().square().sum()) for g in gradients if g is not None) ** 0.5
    if not torch.isfinite(earlier).all() or not earlier.abs().any() or not math.isfinite(norm) or norm == 0:
        raise RuntimeError("Student temporal/parameter gradient audit failed.")
    gradient_audit = {
        "gradient_contract": student.gradient_contract,
        "first_prediction_abs_max": forward_error,
        "audited_physics_steps": audit_steps,
        "recorded_trace_full_forward_abs_max": full_forward_error,
        "same_layout_native_forward_abs_max": native_forward_error,
        "native_error_per_step": (prediction - native_reference).abs().amax(dim=(0, 2)).cpu().tolist(),
        "continuous_coordinate_count": int(solver.is_continuous_dof.sum()),
        "forward_parity_passed": forward_parity,
        "numerical_profile": (
            "Deterministic scratch collision pipeline; compare against native NeRD solver at identical "
            "microbatch/world layout. Old collection layout is a separate diagnostic."
        ),
        "recorded_state_one_step_errors": student.recorded_state_audit(first_batch, audit_steps),
        "later_loss_to_first_state_gradient_norm": float(earlier.norm()),
        "parameter_gradient_norm": norm,
        "finite_difference": _audit_frozen_geometry_gradient(student, first_batch),
    }
    _write_json(args.output / "gradient_audit.json", gradient_audit)
    if not forward_parity:
        raise RuntimeError("Student short-unroll forward parity failed; refusing to fit a mismatched forward model.")
    del intermediate, gradients, earlier, last_loss

    def loss(source, indices):
        if source == "reference":
            return student.reference_loss([reference[i] for i in indices])
        if source != "current":
            raise ValueError("The bounded probe starts with empty historical replay.")
        result = [None] * len(indices)
        for horizon in sorted({len(train[i]["actions"]) for i in indices}):
            positions = [j for j, i in enumerate(indices) if len(train[i]["actions"]) == horizon]
            values = student.sequence_loss(
                [train[indices[j]] for j in positions], max_physics_steps=args.sequence_fit_physics_steps
            )
            for j, value in zip(positions, values):
                result[j] = value
        return torch.stack(result)

    if not 0 < args.probe_lr <= 1e-5 or not 1 <= args.sequence_fit_physics_steps <= 16:
        raise ValueError("P2 probe allows LR<=1e-5 and up to16 physical steps.")
    config = OpdEpochConfig(
        num_epochs=args.sequence_fit_epochs,
        max_epochs=args.sequence_fit_max_epochs,
        micro_batch_size=args.sequence_micro_batch_size,
    )
    sampler = CurrentEpochSampler(
        len(train), reference_count=len(reference), replay_stage_sizes=[], seed=args.sampler_seed, current_batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.probe_lr)
    baseline_train, baseline_valid = evaluate(train), evaluate(valid)
    baseline_reference = evaluate_reference(reference)
    baseline_reference_validation = evaluate_reference(reference_validation)

    def validate_probe():
        result = evaluate(valid)
        result["training_diagnostic"] = evaluate(train)
        result["training_loss_diagnostic"] = result["training_diagnostic"]["loss"]
        result["reference_train"] = evaluate_reference(reference)
        result["reference_validation"] = evaluate_reference(reference_validation)
        result["state_fingerprint"] = _training_fingerprint(model, optimizer)
        return result

    report = fit_opd_epochs(model, optimizer, sampler, loss, validate_probe, config)
    selected_validation = (
        report["epochs"][report["selected_epoch"] - 1]["validation"]
        if report["selected_epoch"]
        else report["baseline_validation"]
    )
    restored_fingerprint = _training_fingerprint(model, optimizer)
    if restored_fingerprint != selected_validation["state_fingerprint"]:
        raise RuntimeError("Selected model and optimizer were not restored as the same epoch pair.")
    final_train, final_valid = evaluate(train), evaluate(valid)
    final_reference = evaluate_reference(reference)
    final_reference_validation = evaluate_reference(reference_validation)
    if _pause_fingerprint(base, base.obs_buf) != before:
        raise RuntimeError("P2 scratch training changed the live environment.")
    cfg = copy.deepcopy(get_cfg_from_checkpoint(checkpoint, args.dynamics))
    cfg["sequence_p2_probe"] = {
        "config": asdict(config),
        "gradient_contract": student.gradient_contract,
        "source": file_sha256(args.queries[0]),
        "physical_steps": args.sequence_fit_physics_steps,
    }
    save_checkpoint(
        args.output / "candidate.pt",
        model,
        "G1-Native-ContactTokens",
        cfg,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=report["selected_epoch"],
        loss_weights=checkpoint["loss_weights"],
    )
    _write_json(
        args.output / "report.json",
        {
            "controller": report,
            "gradient_audit": gradient_audit,
            "config": asdict(config),
            "training_records": len(train),
            "validation_records": len(valid),
            "train_env_ids": args.sequence_train_env_ids,
            "validation_env_ids": args.sequence_validation_env_ids,
            "baseline_train": baseline_train,
            "baseline_validation": baseline_valid,
            "final_train": final_train,
            "final_validation": final_valid,
            "baseline_reference_train": baseline_reference,
            "baseline_reference_validation": baseline_reference_validation,
            "final_reference_train": final_reference,
            "final_reference_validation": final_reference_validation,
            "selected_state_fingerprint": restored_fingerprint,
            "selected_model_optimizer_pair_verified": True,
            "data_sha256": {
                str(path): file_sha256(path)
                for path in [
                    args.queries[0],
                    *reference.paths,
                    *(reference_validation.paths if reference_validation is not None else []),
                ]
            },
            "scratch_geometry_refreshes": student.geometry_refreshes,
            "live_environment_unchanged": True,
            "oracle_execution": shard["metadata"].get("oracle_execution"),
            "ppo_updates": 0,
            "promoted_to_live_model": False,
            "scope": "Reduced-data P2 diagnostic with logical batch8/current4; not the formal512 batch or PPO+OPD run",
            "validation_gate": "Finite held-out loss only for this fixed-data probe; not a production promotion gate",
        },
    )
