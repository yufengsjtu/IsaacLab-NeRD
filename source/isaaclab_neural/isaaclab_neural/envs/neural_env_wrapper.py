# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic wrappers for NeRD-backed Isaac Lab environments."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.physics.nerd_newton_manager import NewtonNerdManager as DefaultNerdManager


class NeuralEnvAdapter:
    """Adapter that exposes NeRD runtime utilities for an existing Isaac Lab env.

    The adapter does not create an environment or choose a task. It only patches
    reset handling and provides a small state-sync surface around the active
    NeRD Newton manager.
    """

    def __init__(
        self,
        env: Any,
        *,
        manager: type = DefaultNerdManager,
        patch_reset: bool = True,
    ):
        self.env = env
        self.manager = manager
        self.isaaclab_env = self._resolve_isaaclab_env(env)
        if patch_reset:
            self.install_reset_patch()

    @staticmethod
    def _resolve_isaaclab_env(env: Any) -> Any:
        """Return the low-level Isaac Lab env that owns ``_reset_idx``."""
        target_env = env
        for attr_name in ("_isaaclab_env", "isaaclab_env", "unwrapped"):
            if hasattr(target_env, "_reset_idx"):
                return target_env
            target_env = getattr(target_env, attr_name, target_env)
        if not hasattr(target_env, "_reset_idx"):
            raise AttributeError("Could not find an IsaacLab env object with _reset_idx to patch.")
        return target_env

    def install_reset_patch(self) -> None:
        """Patch Isaac Lab's ``_reset_idx`` to keep NeRD solver state in sync."""
        if getattr(self.isaaclab_env, "_isaaclab_neural_reset_patched", False):
            return

        original_reset_idx = self.isaaclab_env._reset_idx
        manager = self.manager

        def _reset_idx_with_nerd_solver(env_ids):
            original_reset_idx(env_ids)
            manager.reset_neural_solver(env_ids)

        self.isaaclab_env._reset_idx = _reset_idx_with_nerd_solver
        self.isaaclab_env._isaaclab_neural_reset_patched = True

    @property
    def solver(self):
        """Active NeRD neural solver."""
        return self.manager._solver

    @property
    def model(self):
        """Active Newton model."""
        return self.manager._model

    @property
    def state(self):
        """Current Newton input state."""
        return self.manager._state_0

    @property
    def control(self):
        """Current Newton control buffer."""
        return self.manager._control

    @property
    def states_torch(self) -> torch.Tensor:
        """Current generalized state cache owned by the neural solver."""
        return self.solver.states

    @property
    def root_body_q_torch(self) -> torch.Tensor:
        """Current root body pose cache owned by the neural solver."""
        return self.solver.root_body_q

    def sync(self, *, update_history: bool = False):
        """Synchronize solver input caches from the current Newton state."""
        self.manager.sync_neural_solver(update_history=update_history)
        return self.solver

    def reset_history(self, env_ids=None) -> None:
        """Reset recurrent/history state on the active neural solver."""
        reset_solver = getattr(self.solver, "reset", None)
        if reset_solver is not None:
            reset_solver(env_ids)

    def reset(
        self,
        initial_states: torch.Tensor | None = None,
        *,
        reset_history: bool = True,
        **reset_kwargs,
    ):
        """Reset the wrapped environment or write explicit generalized states.

        Args:
            initial_states: Optional generalized states to write directly into
                Newton before synchronizing NeRD solver caches.
            reset_history: Whether to clear recurrent/history state.
            **reset_kwargs: Forwarded to ``env.reset`` when ``initial_states``
                is ``None``.

        Returns:
            The wrapped env's reset return value, or ``None`` for explicit
            state writes.
        """
        if initial_states is None:
            result = self.env.reset(**reset_kwargs)
            self.sync(update_history=False)
            if reset_history:
                self.reset_history()
            return result

        if self.solver is None:
            raise RuntimeError("Cannot reset NeRD states before the neural solver is initialized.")
        if initial_states.shape[0] > self.solver.num_envs:
            raise ValueError(
                f"initial_states has {initial_states.shape[0]} envs, but solver has {self.solver.num_envs}."
            )

        states = initial_states.to(device=self.solver.torch_device)
        self.solver._assign_states_from_torch(self.state, states)
        self.manager.forward()
        self.sync(update_history=False)
        if reset_history:
            self.reset_history()
        return None

    def get_neural_model_inputs(self):
        """Return neural-model inputs synchronized to the current Newton state."""
        self.sync(update_history=False)
        return self.solver.get_neural_model_inputs()


class NerdManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based Isaac Lab env base class wired to NeRD reset handling."""

    neural_adapter: NeuralEnvAdapter

    def __init__(
        self,
        cfg,
        *,
        solver_cfg: NerdSolverCfg | None = None,
        neural_model_path: str | None = None,
        **kwargs: Any,
    ):
        if not isinstance(cfg.sim.physics, NerdNewtonCfg):
            raise ValueError("cfg.sim.physics must be an instance of NerdNewtonCfg")

        if solver_cfg is not None:
            cfg.sim.physics.solver_cfg = solver_cfg
        if neural_model_path is not None:
            cfg.sim.physics.solver_cfg.neural_model_path = neural_model_path

        super().__init__(cfg=cfg, **kwargs)
        self.neural_adapter = NeuralEnvAdapter(self)
