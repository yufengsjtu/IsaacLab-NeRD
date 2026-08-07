# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager subclass that integrates NeRD without modifying isaaclab_newton."""

from __future__ import annotations

import logging

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsEvent, PhysicsManager

logger = logging.getLogger(__name__)


class NewtonNerdManager(NewtonManager):
    """Newton manager variant that runs a NeRD neural solver."""

    _nerd_abstract_contacts = None
    _nerd_contact_adapter = None
    _nerd_contact_mode: str | None = None
    _nerd_ground_shape_index: int = -1
    _nerd_active: bool = False

    _UPSTREAM_SYNC_ATTRS = (
        "_builder",
        "_model",
        "_state_0",
        "_state_1",
        "_control",
        "_contacts",
        "_solver",
        "_num_substeps",
        "_solver_dt",
        "_needs_collision_pipeline",
        "_collision_pipeline",
        "_use_single_state",
        "_graph",
        "_graph_capture_pending",
        "_usdrt_stage",
        "_model_changes",
        "_views",
        "_newton_contact_sensors",
        "_report_contacts",
    )

    @classmethod
    def clear(cls):
        super().clear()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._nerd_abstract_contacts = None
        cls._nerd_contact_adapter = None
        cls._nerd_contact_mode = None
        cls._nerd_ground_shape_index = -1
        cls._nerd_active = False

    @classmethod
    def dispatch_event(cls, event: PhysicsEvent, payload=None) -> None:
        if event == PhysicsEvent.PHYSICS_READY:
            cls._sync_upstream_newton_state()
        super().dispatch_event(event, payload)

    @classmethod
    def _sync_upstream_newton_state(cls) -> None:
        """Mirror subclass runtime state onto upstream NewtonManager.

        isaaclab_newton assets currently import ``NewtonManager`` directly for
        model/state views. Since ``NewtonNerdManager`` is a subclass, class
        assignments such as ``cls._model = ...`` land on the subclass. Mirroring
        keeps those upstream reads valid without editing isaaclab_newton files.
        """
        if cls is NewtonManager:
            return
        for attr_name in cls._UPSTREAM_SYNC_ATTRS:
            setattr(NewtonManager, attr_name, getattr(cls, attr_name))

    @classmethod
    def initialize_solver(cls) -> None:
        super().initialize_solver()
        cls._sync_upstream_newton_state()

    @classmethod
    def _disable_cuda_graph_for_nerd(cls) -> None:
        """NeRD neural solver steps include PyTorch and cannot use upstream CUDA graphs."""
        cfg = PhysicsManager._cfg
        if cfg is None or not cfg.use_cuda_graph:
            return
        cfg.use_cuda_graph = False
        cls._graph = None
        cls._graph_capture_pending = False
        logger.warning("use_cuda_graph is not supported with NeRD neural solvers; running physics eagerly.")
        cls._sync_upstream_newton_state()

    @classmethod
    def _capture_or_defer_graph(cls) -> None:
        if cls._nerd_active:
            cls._disable_cuda_graph_for_nerd()
            return
        super()._capture_or_defer_graph()

    @classmethod
    def _capture_relaxed_graph(cls, device: str):
        if cls._nerd_active:
            cls._sync_upstream_newton_state()
            return None
        return super()._capture_relaxed_graph(device)

    @classmethod
    def _load_nerd_neural_model(cls, cfg_dict: dict, neural_solver=None):
        neural_model_path = cfg_dict.get("neural_model_path")
        if neural_model_path is None:
            return None

        from isaaclab_neural.utils.checkpoint import install_legacy_module_aliases

        install_legacy_module_aliases()
        device = wp.device_to_torch(PhysicsManager._device)
        loaded = torch.load(neural_model_path, map_location=device, weights_only=False)
        if isinstance(loaded, torch.nn.Module):
            loaded.to(device)
            loaded.eval()
            return loaded

        if isinstance(loaded, dict) and "legacy_model" in loaded:
            model = loaded["legacy_model"]
            model.to(device)
            model.eval()
            return model

        if isinstance(loaded, dict) and loaded.get("version") == 2:
            if neural_solver is None:
                raise ValueError("A NeuralSolver instance is required to reconstruct v2 NeRD checkpoints.")
            from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint

            return reconstruct_model_from_checkpoint(loaded, neural_solver, device=device)

        raise ValueError(f"Unrecognized NeRD checkpoint format at {neural_model_path}")

    @classmethod
    def _build_solver(cls, model, solver_cfg) -> None:
        from isaaclab_neural.contacts import (
            AbstractContact,
            NewtonContactAdapter,
            resolve_contact_packing_policy,
        )
        from isaaclab_neural.solvers import create_neural_solver

        cfg_dict = solver_cfg.to_dict() if hasattr(solver_cfg, "to_dict") else dict(vars(solver_cfg))
        contact_mode = cfg_dict.get("contact_mode", "fixed_ground")
        cls._nerd_active = True
        cls._nerd_contact_mode = contact_mode
        NewtonManager._use_single_state = False

        if contact_mode == "fixed_ground":
            NewtonManager._needs_collision_pipeline = False
            NewtonManager._collision_pipeline = None
            cls._nerd_abstract_contacts = AbstractContact(model)
            NewtonManager._contacts = cls._nerd_abstract_contacts.newton_contacts
            cls._nerd_ground_shape_index = cls._nerd_abstract_contacts.ground_shape_index
            num_contacts_per_env = cls._nerd_abstract_contacts.num_contacts_per_env
            contact_adapter = None
        elif contact_mode == "newton_native":
            NewtonManager._needs_collision_pipeline = True
            num_contacts_per_env = int(cfg_dict.get("num_contacts_per_env", 0))
            if num_contacts_per_env <= 0:
                raise ValueError(
                    "NerdSolverCfg.num_contacts_per_env must be positive when contact_mode='newton_native'."
                )
            packing_policy = resolve_contact_packing_policy(
                contact_mode,
                cfg_dict.get("contact_packing_policy"),
            )
            contact_adapter = NewtonContactAdapter(
                model,
                num_contacts_per_env=num_contacts_per_env,
                device=str(wp.device_to_torch(PhysicsManager._device)),
                packing_policy=packing_policy,
                contact_representation=cfg_dict.get("contact_representation", "flat"),
                max_contact_tokens=int(cfg_dict.get("max_contact_tokens", 64)),
            )
            cls._nerd_contact_adapter = contact_adapter
        else:
            raise ValueError(f"Unsupported NerdSolverCfg.contact_mode: {contact_mode}")

        NewtonManager._solver = create_neural_solver(
            solver_cfg,
            model=model,
            contacts=cls._contacts,
            neural_model=None,
            num_contacts_per_env=num_contacts_per_env,
            contact_mode=contact_mode,
            contact_adapter=contact_adapter,
            contact_representation=cfg_dict.get("contact_representation", "flat"),
            max_contact_tokens=int(cfg_dict.get("max_contact_tokens", 0)),
        )
        neural_model = cls._load_nerd_neural_model(cfg_dict, cls._solver)
        if neural_model is not None:
            NewtonManager._solver.set_neural_solver_model(neural_model)
            NewtonManager._solver.eval()
        cls._disable_cuda_graph_for_nerd()
        cls._sync_upstream_newton_state()

    @classmethod
    def _prepare_nerd_contacts(cls):
        from isaaclab_neural.utils import step_profile

        with step_profile.section("contact_prepare"):
            if cls._nerd_contact_mode == "fixed_ground":
                from isaaclab_neural.contacts import collision_detection_fixed_ground

                if cls._contacts is None:
                    raise RuntimeError("NeRD fixed_ground contact buffer has not been initialized.")
                collision_detection_fixed_ground(
                    cls._model,
                    cls._state_0,
                    cls._contacts,
                    ground_shape_index=cls._nerd_ground_shape_index,
                )
                return cls._contacts
            elif cls._nerd_contact_mode == "newton_native":
                if cls._collision_pipeline is None or cls._contacts is None:
                    raise RuntimeError("NeRD newton_native contact pipeline has not been initialized.")
                cls._collision_pipeline.collide(cls._state_0, cls._contacts)
                return cls._contacts
            else:
                raise RuntimeError(f"Unsupported active NeRD contact mode: {cls._nerd_contact_mode}")

    @classmethod
    def _simulate_full(cls) -> None:
        if not cls._nerd_active:
            super()._simulate_full()
            return

        from isaaclab_neural.utils import step_profile

        physics_dt = cls._solver_dt * cls._num_substeps
        contacts = cls._contacts
        for _ in range(cls._decimation):
            contacts = cls._prepare_nerd_contacts()

            with step_profile.section("actuator"):
                if cls._adapter is not None:
                    cls._adapter.step(cls._state_0, cls._control, physics_dt)
                for cb in cls._post_actuator_callbacks:
                    cb()

            cls._run_solver_substeps(contacts)

        cls._update_sensors(contacts)
        cls._sync_upstream_newton_state()

    @classmethod
    def _simulate_physics_only(cls) -> None:
        if not cls._nerd_active:
            super()._simulate_physics_only()
            return

        contacts = cls._prepare_nerd_contacts()
        cls._run_solver_substeps(contacts)
        cls._update_sensors(contacts)
        cls._sync_upstream_newton_state()

    @classmethod
    def _update_sensors(cls, contacts) -> None:
        if cls._newton_frame_transform_sensors:
            for sensor in cls._newton_frame_transform_sensors:
                sensor.update(cls._state_0)
        if cls._newton_imu_sensors:
            for sensor in cls._newton_imu_sensors:
                sensor.update(cls._state_0)

    @classmethod
    def sync_neural_solver(cls, *, update_history: bool = True) -> None:
        """Refresh neural-solver input caches from the current Newton state."""
        if not cls._nerd_active or cls._solver is None:
            return
        if cls._control is None:
            raise RuntimeError("Cannot sync NeRD solver before Newton control is initialized.")

        contacts = cls._prepare_nerd_contacts()
        sync_from_newton = getattr(cls._solver, "sync_from_newton", None)
        if sync_from_newton is None:
            raise RuntimeError("Active NeRD solver does not implement sync_from_newton().")
        sync_from_newton(cls._state_0, contacts, cls._control.joint_f, update_history=update_history)
        cls._sync_upstream_newton_state()

    @classmethod
    def reset_neural_solver(cls, env_ids=None, *, reset_history: bool = True) -> None:
        """Synchronize Newton kinematics and reset NeRD recurrent/history state.

        IsaacLab task resets usually write generalized coordinates directly.
        NeRD consumes body/root poses and may maintain transformer/RNN history,
        so reset handling needs the same two follow-up operations as the
        original ``NeuralEnvironment`` wrapper: run FK and reset solver memory.
        """
        if not cls._nerd_active or cls._solver is None:
            return

        cls.forward()
        cls.sync_neural_solver(update_history=False)

        reset_solver = getattr(cls._solver, "reset", None)
        if reset_history and reset_solver is not None:
            should_reset = env_ids is None
            if env_ids is not None:
                try:
                    should_reset = len(env_ids) > 0
                except TypeError:
                    should_reset = True
            if should_reset:
                reset_solver(env_ids)
