# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import deque
from collections.abc import Mapping

import torch
from newton import Contacts, State

from .neural_solver import NeuralSolver


class TransformerNeuralSolver(NeuralSolver):
    """Neural solver variant that feeds a fixed history window to transformer models."""

    def __init__(
        self,
        num_states_history: int = 1,
        use_feature_cache: bool = False,
        use_kv_cache: bool = False,
        **kwargs,
    ):
        if use_feature_cache and use_kv_cache:
            raise ValueError("use_feature_cache and use_kv_cache cannot both be enabled.")
        self.num_states_history = num_states_history
        self.use_feature_cache = use_feature_cache
        self.use_kv_cache = use_kv_cache
        self.feature_cache = None
        self.feature_cache_rebuild_env_ids = None
        self.kv_cache = None
        self.kv_cache_rebuild_env_ids = None
        self.kv_cache_rebuild_inputs = None
        super().__init__(**kwargs)
        self.reset_states_history()

    def reset_states_history(self):
        self.states_history = deque(maxlen=self.num_states_history)
        self.feature_cache = None
        self.feature_cache_rebuild_env_ids = None
        self.kv_cache = None
        self.kv_cache_rebuild_env_ids = None
        self.kv_cache_rebuild_inputs = None

    def preload_states_history(self, history: Mapping[str, torch.Tensor]) -> None:
        """Replace solver history with validated raw dataset inputs.

        Args:
            history: Raw model inputs with shape ``[num_envs, history, features]``.
        """
        required = {"root_body_q", "states", "joint_f", "gravity_dir", *self.contacts.keys()}
        if getattr(self, "contact_token_frame", "root") == "owner":
            required.add("primary_body_q")
        missing = sorted(required - set(history))
        if missing:
            raise ValueError(f"History is missing required model inputs: {missing}.")

        states = history["states"]
        if states.ndim != 3 or states.shape[0] != self.num_envs:
            raise ValueError(
                f"History states must have shape [num_envs, history, state_dim], got {tuple(states.shape)}."
            )
        history_length = states.shape[1]
        if history_length == 0 or history_length > self.num_states_history:
            raise ValueError(f"History length must be between 1 and {self.num_states_history}, got {history_length}.")

        live_inputs = {
            "root_body_q": self.root_body_q,
            "states": self.states,
            "joint_f": self.joint_f,
            "gravity_dir": self.gravity_dir,
            **self.contacts,
        }
        if getattr(self, "contact_token_frame", "root") == "owner":
            if self.primary_body_q is None:
                raise ValueError("contact_token_frame='owner' requires primary_body_q on the solver.")
            live_inputs["primary_body_q"] = self.primary_body_q
        normalized_history: dict[str, torch.Tensor] = {}
        for key in required:
            value = history[key]
            expected = live_inputs[key]
            if expected.ndim == 1 and value.ndim == 3 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            expected_shape = (self.num_envs, history_length, *expected.shape[1:])
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"History {key} must have shape {expected_shape}, got {tuple(value.shape)}.")
            normalized_history[key] = value

        self.reset_states_history()
        for step in range(history_length):
            entry = {
                key: value[:, step].to(device=self.torch_device).clone() for key, value in normalized_history.items()
            }
            entry["states_embedding"] = self.embed_states(entry["states"])
            self.states_history.append(entry)
        if self.use_kv_cache:
            self.kv_cache_rebuild_env_ids = torch.arange(self.num_envs, device=self.torch_device)

    def reset(self, env_ids=None, world_mask=None, flags=None):
        """Reset transformer history globally or for selected environments.

        A full reset (``env_ids is None`` or every environment) drops the
        history window so later steps refill it the same way as episode start.
        A partial reset copies the current post-reset solver inputs into the
        selected rows. Zero-filling those rows previously fed the model a
        window of invalid zero poses and caused immediate re-falls.

        Isaac Lab 3's Newton manager also calls this as
        ``reset(state, world_mask=..., flags=0)``. That path has no history to
        clear; history is reset with ``env_ids``.

        Args:
            env_ids: Environment indices to reset. ``None`` resets every
                environment. Newton-style calls pass the Newton ``State`` here.
            world_mask: Unused Newton SolverBase reset mask.
            flags: Unused Newton SolverBase reset flags.
        """
        if world_mask is not None or flags is not None:
            super().reset(env_ids, world_mask=world_mask, flags=flags)
            return
        super().reset(env_ids)
        if env_ids is None:
            self.reset_states_history()
            return

        env_ids = torch.unique(self._normalize_env_ids(env_ids))
        if env_ids.numel() == 0:
            return
        all_env_ids = torch.arange(self.num_envs, device=env_ids.device, dtype=env_ids.dtype)
        if torch.equal(env_ids, all_env_ids):
            self.reset_states_history()
            return
        if len(self.states_history) == 0:
            return

        self._fill_history_rows_from_current_state(env_ids)
        self._mark_history_rows_for_rebuild(env_ids)

    def _fill_history_rows_from_current_state(self, env_ids: torch.Tensor) -> None:
        """Replace selected history rows with the current solver snapshot."""
        snapshot = self._history_snapshot()
        for history_entry in self.states_history:
            for key, value in history_entry.items():
                source = snapshot.get(key)
                if source is None:
                    continue
                value.index_copy_(0, env_ids, source.index_select(0, env_ids).to(dtype=value.dtype))

    def _mark_history_rows_for_rebuild(self, env_ids: torch.Tensor) -> None:
        """Rebuild cached transformer features for the selected environments."""
        if self.feature_cache_rebuild_env_ids is None:
            self.feature_cache_rebuild_env_ids = env_ids
        else:
            self.feature_cache_rebuild_env_ids = torch.unique(torch.cat((self.feature_cache_rebuild_env_ids, env_ids)))
        if self.kv_cache is not None:
            self.neural_model.reset_transformer_cache(self.kv_cache, env_ids)
        if self.kv_cache_rebuild_env_ids is None:
            self.kv_cache_rebuild_env_ids = env_ids
        else:
            self.kv_cache_rebuild_env_ids = torch.unique(torch.cat((self.kv_cache_rebuild_env_ids, env_ids)))

    def _normalize_env_ids(self, env_ids) -> torch.Tensor:
        """Return env ids as a 1-D tensor on the solver torch device."""
        if isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.torch_device, dtype=torch.long)
        else:
            normalized = torch.as_tensor(env_ids, device=self.torch_device, dtype=torch.long)
        return normalized.reshape(-1)

    def sync_from_newton(
        self, newton_states: State, contacts: Contacts, joint_f, *, update_history: bool = True
    ) -> None:
        """Synchronize cached inputs, optionally without appending to history."""
        if update_history:
            self._update_states(newton_states, contacts, joint_f)
        else:
            NeuralSolver._update_states(self, newton_states, contacts, joint_f)

    def _update_states(self, newton_states: State, contacts: Contacts, joint_f):
        super()._update_states(newton_states, contacts, joint_f)
        self.states_history.append(self._history_snapshot())

    def _history_snapshot(self) -> dict[str, torch.Tensor]:
        """Clone mutable solver inputs before storing one online history frame."""
        snapshot = {
            "root_body_q": self.root_body_q.clone(),
            "states": self.states.clone(),
            "states_embedding": self.states_embedding.clone(),
            "joint_f": self.joint_f.clone(),
            "gravity_dir": self.gravity_dir.clone(),
            **{key: value.clone() for key, value in self.contacts.items()},
        }
        if getattr(self, "contact_token_frame", "root") == "owner":
            if self.primary_body_q is None:
                raise ValueError("contact_token_frame='owner' requires primary_body_q on the solver.")
            snapshot["primary_body_q"] = self.primary_body_q.clone()
        return snapshot

    def get_neural_model_inputs(self):
        if len(self.states_history) == 0:  # for dummy call
            model_inputs = {
                "root_body_q": torch.zeros_like(self.root_body_q),
                "states": torch.zeros_like(self.states),
                "states_embedding": torch.zeros_like(self.states_embedding),
                "joint_f": torch.zeros_like(self.joint_f),
                "gravity_dir": torch.zeros_like(self.gravity_dir),
                **{key: torch.zeros_like(value) for key, value in self.contacts.items()},
            }
            return {key: value.unsqueeze(1) for key, value in model_inputs.items()}

        history = tuple(self.states_history)
        if self.use_kv_cache:
            latest_inputs = {key: value.unsqueeze(1) for key, value in history[-1].items()}
            rebuild_env_ids = self.kv_cache_rebuild_env_ids
            if rebuild_env_ids is not None and rebuild_env_ids.numel() > 0:
                rebuild_inputs = {}
                for key in history[0]:
                    rebuild_inputs[key] = torch.stack(
                        tuple(entry[key].index_select(0, rebuild_env_ids) for entry in history),
                        dim=1,
                    )
                self.kv_cache_rebuild_inputs = self.process_neural_model_inputs(rebuild_inputs)
            else:
                self.kv_cache_rebuild_inputs = None
            return self.process_neural_model_inputs(latest_inputs)

        # Stack directly into the model's [B, T, ...] layout. The previous
        # default_collate + transpose + contiguous path materialized the full
        # history twice on every simulation step.
        model_inputs = {}
        for key in history[0]:
            value = torch.stack(tuple(entry[key] for entry in history), dim=1)
            if value.ndim < 2:
                raise ValueError(
                    f"History field {key!r} must have at least two dimensions after collate "
                    f"(got shape {tuple(value.shape)})."
                )
            model_inputs[key] = value

        return self.process_neural_model_inputs(model_inputs)

    def _model_forward(self, model_inputs):
        if self.use_kv_cache:
            return self._model_forward_kv_cached(model_inputs)
        if not self.use_feature_cache:
            return super()._model_forward(model_inputs)
        if not hasattr(self.neural_model, "forward_feature_cached"):
            raise RuntimeError("use_feature_cache requires a model that implements forward_feature_cached().")
        prediction, self.feature_cache = self.neural_model.forward_feature_cached(
            model_inputs,
            self.feature_cache,
            self.feature_cache_rebuild_env_ids,
        )
        self.feature_cache_rebuild_env_ids = None
        return prediction

    def _model_forward_kv_cached(self, model_inputs):
        """Run one incremental step and selectively rebuild reset cache rows."""
        if not hasattr(self.neural_model, "forward_cached"):
            raise RuntimeError("use_kv_cache requires a model that implements forward_cached().")

        prediction, self.kv_cache = self.neural_model.forward_cached(
            model_inputs,
            self.kv_cache,
            max_cache_length=self.num_states_history,
        )

        rebuild_env_ids = self.kv_cache_rebuild_env_ids
        self.kv_cache_rebuild_env_ids = None
        if rebuild_env_ids is None or rebuild_env_ids.numel() == 0:
            return prediction

        rebuild_inputs = self.kv_cache_rebuild_inputs
        self.kv_cache_rebuild_inputs = None
        if rebuild_inputs is None:
            raise RuntimeError("KV-cache reset rows require prepared history inputs.")
        rebuilt_cache = None
        rebuilt_prediction = None
        history_length = next(value.shape[1] for value in rebuild_inputs.values() if isinstance(value, torch.Tensor))
        for step in range(history_length):
            step_inputs = {
                key: value[:, step : step + 1] if isinstance(value, torch.Tensor) else value
                for key, value in rebuild_inputs.items()
            }
            rebuilt_prediction, rebuilt_cache = self.neural_model.forward_cached(
                step_inputs,
                rebuilt_cache,
                max_cache_length=self.num_states_history,
            )

        self.kv_cache.copy_rows_from(rebuilt_cache, rebuild_env_ids)
        prediction.index_copy_(0, rebuild_env_ids, rebuilt_prediction)
        return prediction
