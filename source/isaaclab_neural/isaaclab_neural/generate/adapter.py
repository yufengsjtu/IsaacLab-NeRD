# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapters that expose Newton simulation data for dataset generation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def build_primary_body_sensor_indices(sensor_names: list[str], primary_body_names: list[str]) -> torch.Tensor:
    """Map contact-sensor rows into the primary-body slot ordering.

    Names may be full prim paths or leaf names. Every primary body must map
    exactly once; extra sensor rows are allowed.
    """

    def leaf_name(name: str) -> str:
        return str(name).rstrip("/").rsplit("/", maxsplit=1)[-1]

    primary_leaves = [leaf_name(name) for name in primary_body_names]
    if len(set(primary_leaves)) != len(primary_leaves):
        raise ValueError(f"Primary body names are ambiguous after path normalization: {primary_leaves}.")

    sensor_rows: dict[str, list[int]] = {}
    for index, name in enumerate(sensor_names):
        sensor_rows.setdefault(leaf_name(name), []).append(index)

    missing = [name for name in primary_leaves if name not in sensor_rows]
    ambiguous = {name: sensor_rows[name] for name in primary_leaves if len(sensor_rows.get(name, [])) > 1}
    if missing or ambiguous:
        raise ValueError(
            "Contact sensor names do not map one-to-one to primary robot bodies: "
            f"missing={missing}, ambiguous={ambiguous}."
        )
    return torch.tensor([sensor_rows[name][0] for name in primary_leaves], dtype=torch.long)


class BodyNetContactForceRecorder:
    """Read resultant Newton contact forces in primary-body slot order."""

    def __init__(self, sensor: Any, contact_adapter: Any, *, num_envs: int, device: str | torch.device):
        self.sensor = sensor
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        body_ids_by_world = contact_adapter.primary_body_ids_by_world
        if body_ids_by_world.ndim != 2 or body_ids_by_world.shape[0] != self.num_envs:
            raise ValueError(
                "Primary body ids must have shape [num_envs, num_primary_bodies], "
                f"got {tuple(body_ids_by_world.shape)}."
            )
        labels = getattr(contact_adapter.model, "body_label", None)
        if labels is None:
            labels = getattr(contact_adapter.model, "body_key", None)
        if labels is None:
            raise RuntimeError("Newton model does not expose body_label or body_key metadata.")
        labels = list(labels)
        primary_names_by_world = [
            [str(labels[int(body_id)]) for body_id in body_ids.tolist()] for body_ids in body_ids_by_world
        ]
        primary_leaves_by_world = [
            [name.rstrip("/").rsplit("/", maxsplit=1)[-1] for name in names] for names in primary_names_by_world
        ]
        if any(names != primary_leaves_by_world[0] for names in primary_leaves_by_world[1:]):
            raise ValueError("Primary body names and slot ordering must be identical across Newton worlds.")
        sensor_names = getattr(sensor, "sensor_names", None)
        if not sensor_names:
            raise RuntimeError("Newton ContactSensor sensor_names metadata is unavailable.")
        self.primary_body_names = primary_leaves_by_world[0]
        self.sensor_indices = build_primary_body_sensor_indices(list(sensor_names), self.primary_body_names).to(
            self.device
        )

    @property
    def num_primary_bodies(self) -> int:
        """Number of primary robot body slots."""
        return int(self.sensor_indices.numel())

    def read_post_solve(self) -> torch.Tensor:
        """Return post-solve resultant linear contact forces [N] in the world frame.

        The forces include normal and tangential/friction components but exclude
        contact moments and torques.
        """
        net_forces = self.sensor.data.net_forces_w
        if net_forces is None:
            raise RuntimeError("Newton ContactSensor.data.net_forces_w is unavailable.")
        values = net_forces.torch if hasattr(net_forces, "torch") else torch.as_tensor(net_forces)
        values = values.to(self.device)
        expected_prefix = (self.num_envs, len(getattr(self.sensor, "sensor_names", [])))
        if values.ndim != 3 or tuple(values.shape[:2]) != expected_prefix or values.shape[-1] != 3:
            raise RuntimeError(
                "Newton ContactSensor.data.net_forces_w must have shape "
                f"[{self.num_envs}, {expected_prefix[1]}, 3], got {tuple(values.shape)}."
            )
        return values.index_select(1, self.sensor_indices)


class NewtonDataGenerationBackend:
    """Small wrapper around NewtonManager state needed by data generation."""

    def __init__(self, device: str | torch.device):
        from isaaclab_newton.physics import NewtonManager

        self._manager = NewtonManager
        self.device = device
        self.model = NewtonManager._model
        self.state = NewtonManager._state_0
        self.control = NewtonManager._control
        if self.model is None or self.state is None or self.control is None:
            raise RuntimeError("NewtonManager is not initialized. Create the env inside launch_simulation() first.")

    @property
    def native_contacts(self):
        """Return Newton native contacts."""
        return self._manager._contacts

    def collide_native_contacts(self):
        """Refresh and return Newton native contacts."""
        contacts = self._manager._contacts
        pipeline = self._manager._collision_pipeline
        if pipeline is not None and contacts is not None:
            pipeline.collide(self.state, contacts)
        return contacts

    def sync_solver(self, solver, contacts) -> None:
        """Synchronize a recording solver from current Newton state."""
        if contacts is None and int(getattr(solver, "num_contacts_per_env", 0)) > 0:
            raise RuntimeError("Newton contacts are not initialized; cannot record NeRD contact tensors.")
        if self.control is None:
            raise RuntimeError("Newton control is not initialized; cannot record joint forces.")
        solver.sync_from_newton(self.state, contacts, self.control.joint_f, update_history=False)

    def assign_solver_states(self, solver, states: torch.Tensor) -> None:
        """Write generalized states through the solver's frame-aware helper."""
        solver._assign_states_from_torch(self.state, states)
        self._manager.forward()

    @contextmanager
    def single_physics_step(self):
        """Temporarily force NewtonManager to advance a single physics frame."""
        old_decimation = self._manager._decimation
        self._manager._decimation = 1
        try:
            yield
        finally:
            self._manager._decimation = old_decimation

    def step_physics_only(self) -> None:
        """Step Newton physics without actuator processing."""
        import warp as wp

        with wp.ScopedDevice(str(self.device)):
            self._manager._simulate_physics_only()

    def assign_joint_f(self, joint_f: torch.Tensor) -> None:
        """Write direct joint forces into Newton control."""
        import warp as wp

        if self.control is None:
            raise RuntimeError("Newton control is not initialized; cannot apply direct joint forces.")
        control_joint_f = self.control.joint_f
        if control_joint_f is None:
            raise RuntimeError("Newton control.joint_f is not allocated; cannot apply direct joint forces.")
        control_joint_f.assign(wp.from_torch(joint_f.reshape(-1)))


class DataGenerationAdapter:
    """Expose NeRD training fields from a ground-truth Newton environment.

    The wrapped IsaacLab env remains a normal Newton env. This adapter creates a
    NeRD neural-solver input pipeline only for recording states, forces, and
    contacts into training datasets.
    """

    def __init__(self, env: Any, solver_cfg: Any, *, record_body_net_contact_forces: bool = False):
        self.env = getattr(env, "unwrapped", env)
        self.solver_cfg = solver_cfg
        self.device = getattr(self.env, "device", "cpu")
        self.backend = NewtonDataGenerationBackend(self.device)
        self.model = self.backend.model
        self.state = self.backend.state
        self.control = self.backend.control
        self._body_world_torch: torch.Tensor | None = None
        self.body_net_contact_force_recorder: BodyNetContactForceRecorder | None = None

        from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode

        self.contact_mode = normalize_contact_mode(solver_cfg.contact_mode)
        solver_cfg.contact_mode = self.contact_mode
        from isaaclab_neural.contacts import NewtonContactAdapter

        num_contacts_per_env = int(solver_cfg.num_contacts_per_env)
        if num_contacts_per_env < 0:
            raise ValueError("num_contacts_per_env must be non-negative.")
        contact_representation = getattr(solver_cfg, "contact_representation", "flat")
        if num_contacts_per_env == 0 and contact_representation == "contact_tokens":
            raise ValueError("contact_representation='contact_tokens' requires a positive contact capacity.")
        if num_contacts_per_env == 0:
            contacts = None
            self.contact_adapter = None
        else:
            contacts = self.backend.native_contacts
            self.contact_adapter = NewtonContactAdapter(
                self.model,
                num_contacts_per_env=num_contacts_per_env,
                device=str(self.device),
                packing_policy=solver_cfg.contact_packing_policy,
                contact_representation=contact_representation,
                max_contact_tokens=int(getattr(solver_cfg, "max_contact_tokens", 64)),
            )

        if record_body_net_contact_forces:
            if self.contact_adapter is None:
                raise ValueError("Body net contact-force labels require a positive contact capacity.")
            if contact_representation != "contact_tokens":
                raise ValueError("Body net contact-force labels require contact_representation='contact_tokens'.")
            scene = getattr(self.env, "scene", None)
            sensors = getattr(scene, "sensors", {})
            sensor = sensors.get("contact_forces") if sensors is not None else None
            if sensor is None:
                raise RuntimeError("Body net contact-force recording requires scene sensor 'contact_forces'.")
            self.body_net_contact_force_recorder = BodyNetContactForceRecorder(
                sensor,
                self.contact_adapter,
                num_envs=int(getattr(self.env, "num_envs")),
                device=self.device,
            )

        from isaaclab_neural.solvers import create_neural_solver

        self.solver = create_neural_solver(
            solver_cfg,
            model=self.model,
            contacts=contacts,
            neural_model=None,
            num_contacts_per_env=num_contacts_per_env,
            contact_mode=self.contact_mode,
            contact_adapter=self.contact_adapter,
            contact_representation=contact_representation,
            max_contact_tokens=int(getattr(self.solver_cfg, "max_contact_tokens", 0)),
        )
        self.sync()

    @property
    def num_envs(self) -> int:
        """Number of vectorized environments."""
        return int(getattr(self.env, "num_envs", self.solver.num_envs))

    @property
    def action_dim(self) -> int:
        """Flat action dimension."""
        return int(self._action_shape()[0])

    @property
    def state_dim(self) -> int:
        """Flat generalized state dimension."""
        return int(self.solver.states.shape[-1])

    @property
    def state_world_ids(self) -> torch.Tensor:
        """Newton world id represented by each generalized-state row."""
        return self.solver.state_world_ids

    @property
    def root_world_ids(self) -> torch.Tensor:
        """Newton world id represented by each cached root-body row."""
        return self.solver.root_world_ids

    @property
    def contact_world_ids(self) -> torch.Tensor:
        """Newton world id represented by each packed contact row."""
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    @property
    def contact_token_body_ids(self) -> torch.Tensor | None:
        """Packed global owner-body ids, if contact tokens are active."""
        if self.contact_adapter is None:
            return None
        return getattr(self.contact_adapter, "contact_token_body_ids", None)

    @property
    def contact_token_world_ids(self) -> torch.Tensor | None:
        """Newton world ids derived from each packed token's owner body."""
        body_ids = self.contact_token_body_ids
        if body_ids is None:
            return None
        world_ids = torch.full_like(body_ids, -1)
        valid = body_ids >= 0
        if not valid.any():
            return world_ids
        body_world = getattr(self.model, "body_world", None)
        if body_world is None:
            raise RuntimeError("Newton body_world metadata is required to record contact-token owner worlds.")
        body_world_torch = getattr(self, "_body_world_torch", None)
        if body_world_torch is None or body_world_torch.device != body_ids.device:
            body_world_torch = torch.as_tensor(body_world.numpy(), device=body_ids.device, dtype=torch.long)
            self._body_world_torch = body_world_torch
        world_ids[valid] = body_world_torch.index_select(0, body_ids[valid])
        return world_ids

    @property
    def joint_f_dim(self) -> int:
        """Flat joint-force dimension."""
        return int(self.solver.joint_f.shape[-1])

    @property
    def num_contacts_per_env(self) -> int:
        """Number of fixed contact slots per environment."""
        return int(self.solver.num_contacts_per_env)

    @property
    def num_primary_bodies(self) -> int:
        """Number of primary body slots available for force labels."""
        if self.body_net_contact_force_recorder is None:
            return 0
        return self.body_net_contact_force_recorder.num_primary_bodies

    @property
    def body_net_contact_force_body_names(self) -> tuple[str, ...]:
        """Primary body names in force-label slot order."""
        if self.body_net_contact_force_recorder is None:
            return ()
        return tuple(self.body_net_contact_force_recorder.primary_body_names)

    def body_net_contact_forces_w(self) -> torch.Tensor:
        """Return current per-primary-body resultant contact forces [N].

        Values are read after the latest solve in the world frame and include
        normal and tangential/friction components, but not contact moments or
        torques.
        """
        if self.body_net_contact_force_recorder is None:
            raise RuntimeError("Body net contact-force recording is not enabled.")
        return self.body_net_contact_force_recorder.read_post_solve()

    def contact_truncation_summary(self) -> dict[str, int | float] | None:
        """Return native contact truncation statistics when available."""
        if self.contact_adapter is None:
            return None
        return self.contact_adapter.truncation_summary()

    def _action_shape(self) -> tuple[int, ...]:
        shape = tuple(self.env.action_space.shape)
        if len(shape) == 1:
            return shape
        if len(shape) == 2:
            return (shape[-1],)
        raise ValueError(f"Unsupported action space shape: {shape}")

    def _prepare_contacts(self):
        if self.contact_adapter is None:
            return None
        return self.backend.collide_native_contacts()

    def sync(self) -> None:
        """Synchronize recorded tensors from Newton state/control."""
        contacts = self._prepare_contacts()
        self.backend.sync_solver(self.solver, contacts)

    def action_bounds(self, low: float = -1.0, high: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite action sampling bounds."""
        action_dim = self.action_dim
        device = self.device
        action_low = torch.full((action_dim,), low, device=device, dtype=torch.float32)
        action_high = torch.full((action_dim,), high, device=device, dtype=torch.float32)

        gym_low = getattr(self.env.action_space, "low", None)
        gym_high = getattr(self.env.action_space, "high", None)
        if gym_low is not None and gym_high is not None:
            low_tensor = torch.as_tensor(gym_low, device=device, dtype=torch.float32).flatten()[-action_dim:]
            high_tensor = torch.as_tensor(gym_high, device=device, dtype=torch.float32).flatten()[-action_dim:]
            action_low = torch.where(torch.isfinite(low_tensor), low_tensor, action_low)
            action_high = torch.where(torch.isfinite(high_tensor), high_tensor, action_high)
        return action_low, action_high

    def joint_force_bounds(self, joint_f_lim: float | np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        """Return absolute joint-force sampling limits."""
        if joint_f_lim is None:
            return torch.full((self.joint_f_dim,), 1.0e6, device=self.device, dtype=torch.float32)
        if isinstance(joint_f_lim, torch.Tensor):
            return joint_f_lim.to(device=self.device, dtype=torch.float32).flatten()
        if isinstance(joint_f_lim, np.ndarray):
            return torch.as_tensor(joint_f_lim, device=self.device, dtype=torch.float32).flatten()
        return torch.full((self.joint_f_dim,), float(joint_f_lim), device=self.device, dtype=torch.float32)

    def state_bounds(
        self,
        joint_q_min: float | np.ndarray | torch.Tensor | None = None,
        joint_q_max: float | np.ndarray | torch.Tensor | None = None,
        joint_qd_lim: float | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return generalized-state sampling bounds."""
        dof_q = int(self.solver.dof_q_per_env)
        dof_qd = int(self.solver.dof_qd_per_env)

        q_min = self._limit_tensor(joint_q_min, dof_q, -1.0e6)
        q_max = self._limit_tensor(joint_q_max, dof_q, 1.0e6)
        qd_lim = self._limit_tensor(joint_qd_lim, dof_qd, 1.0e6)

        joint_limit_lower = getattr(self.solver, "joint_limit_lower", None)
        joint_limit_upper = getattr(self.solver, "joint_limit_upper", None)
        if joint_limit_lower is not None and joint_limit_upper is not None:
            lower = joint_limit_lower.to(device=self.device, dtype=torch.float32).flatten()
            upper = joint_limit_upper.to(device=self.device, dtype=torch.float32).flatten()
            if lower.numel() >= dof_qd and upper.numel() >= dof_qd:
                base_joint_type = getattr(self.solver, "base_joint_type", None)
                if getattr(base_joint_type, "name", "") == "FREE":
                    q_min[7:] = torch.maximum(q_min[7:], lower[6:dof_qd])
                    q_max[7:] = torch.minimum(q_max[7:], upper[6:dof_qd])
                else:
                    q_min = torch.maximum(q_min, lower[:dof_q])
                    q_max = torch.minimum(q_max, upper[:dof_q])

        return torch.cat([q_min, -qd_lim]), torch.cat([q_max, qd_lim])

    def _limit_tensor(self, value, dim: int, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((dim,), default, device=self.device, dtype=torch.float32)
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32).flatten()
        elif isinstance(value, np.ndarray):
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32).flatten()
        else:
            tensor = torch.full((dim,), float(value), device=self.device, dtype=torch.float32)
        if tensor.numel() != dim:
            raise ValueError(f"Expected limit with {dim} entries, got {tensor.numel()}.")
        return tensor

    def zero_action(self) -> torch.Tensor:
        """Return zero actions with env batch dimension."""
        return torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)

    def sample_action(self, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
        """Sample uniformly bounded actions."""
        action = torch.rand((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)
        return action * (action_high - action_low) + action_low

    def sample_joint_f(self, joint_f_lim: torch.Tensor) -> torch.Tensor:
        """Sample uniformly bounded joint forces."""
        joint_f = torch.rand((self.num_envs, self.joint_f_dim), device=self.device, dtype=torch.float32)
        return joint_f * (2.0 * joint_f_lim) - joint_f_lim

    def sample_initial_states(self, states_min: torch.Tensor, states_max: torch.Tensor) -> torch.Tensor:
        """Sample uniformly bounded generalized states."""
        states = torch.rand((self.num_envs, self.state_dim), device=self.device, dtype=torch.float32)
        return states * (states_max - states_min) + states_min

    def reset(self, initial_states: torch.Tensor | None = None):
        """Reset env or write explicit generalized states, then refresh recording tensors."""
        if initial_states is None:
            result = self.env.reset()
            self.sync()
            return result

        states = initial_states.to(device=self.device, dtype=torch.float32)
        if states.shape != (self.num_envs, self.state_dim):
            raise ValueError(
                f"initial_states must have shape {(self.num_envs, self.state_dim)}, got {tuple(states.shape)}."
            )

        self.backend.assign_solver_states(self.solver, states)
        if hasattr(self.env, "scene"):
            self.env.scene.update(dt=getattr(self.env, "physics_dt", 0.0))
        self.sync()
        return None

    def step(self, action: torch.Tensor):
        """Step the wrapped env and refresh recorded tensors."""
        result = self.env.step(action)
        self.sync()
        return result

    def step_action_frame(self, action: torch.Tensor, *, process_action: bool = True):
        """Step one physics frame using IsaacLab's action pipeline."""
        action = action.to(self.device)
        if process_action:
            self.env.action_manager.process_action(action)
        self.env.action_manager.apply_action()
        self.env.scene.write_data_to_sim()
        with self.backend.single_physics_step():
            self.env.sim.step(render=False)
        if hasattr(self.env, "_sim_step_counter"):
            self.env._sim_step_counter += 1
        self.env.scene.update(dt=getattr(self.env, "physics_dt", 0.0))
        self.sync()
        return self.neural_inputs()["states"]

    def step_joint_f_frame(self, joint_f: torch.Tensor):
        """Step one physics frame with direct joint forces, bypassing actuators."""
        joint_f = joint_f.to(device=self.device, dtype=torch.float32)
        if joint_f.shape != (self.num_envs, self.joint_f_dim):
            raise ValueError(
                f"joint_f must have shape {(self.num_envs, self.joint_f_dim)}, got {tuple(joint_f.shape)}."
            )
        self.backend.assign_joint_f(joint_f)
        self.backend.step_physics_only()
        if hasattr(self.env, "_sim_step_counter"):
            self.env._sim_step_counter += 1
        if hasattr(self.env, "scene"):
            self.env.scene.update(dt=getattr(self.env, "physics_dt", 0.0))
        self.sync()
        return self.neural_inputs()["states"]

    def neural_inputs(self):
        """Current neural-model input tensors."""
        return self.solver.get_neural_model_inputs()

    def raw_neural_inputs(self):
        """Current raw Newton tensors before neural-model preprocessing."""
        return self.solver.get_raw_neural_model_inputs()
