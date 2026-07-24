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

    def __init__(self, num_states_history: int = 1, **kwargs):
        self.num_states_history = num_states_history
        super().__init__(**kwargs)
        self.reset_states_history()

    def reset_states_history(self):
        self.states_history = deque(maxlen=self.num_states_history)

    def preload_states_history(self, history: Mapping[str, torch.Tensor]) -> None:
        """Replace solver history with validated raw dataset inputs.

        Args:
            history: Raw model inputs with shape ``[num_envs, history, features]``.
        """
        required = {"root_body_q", "states", "joint_f", "gravity_dir", *self.contacts.keys()}
        missing = sorted(required - set(history))
        if missing:
            raise ValueError(f"History is missing required model inputs: {missing}.")

        states = history["states"]
        if states.ndim != 3 or states.shape[0] != self.num_envs:
            raise ValueError(
                f"History states must have shape [num_envs, history, state_dim], got {tuple(states.shape)}."
            )
        history_length = states.shape[1]
        if history_length > self.num_states_history:
            raise ValueError(
                f"History length {history_length} exceeds configured maximum {self.num_states_history}."
            )

        self.reset_states_history()
        for step in range(history_length):
            entry = {
                key: value[:, step].to(device=self.torch_device).clone()
                for key, value in history.items()
                if key in required
            }
            entry["states_embedding"] = self.embed_states(entry["states"])
            self.states_history.append(entry)

    def reset(self, env_ids=None):
        """Reset transformer history globally or for selected environments."""
        if env_ids is None:
            self.reset_states_history()
            return

        if len(self.states_history) == 0:
            return

        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        for history_entry in self.states_history:
            for value in history_entry.values():
                value[env_ids] = 0.0

    def _normalize_env_ids(self, env_ids) -> torch.Tensor:
        """Return env ids as a 1-D tensor on the solver torch device."""
        if isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.torch_device, dtype=torch.long)
        else:
            normalized = torch.as_tensor(env_ids, device=self.torch_device, dtype=torch.long)
        return normalized.reshape(-1)

    def sync_from_newton(self, newton_states: State, contacts: Contacts, joint_f, *, update_history: bool = True) -> None:
        """Synchronize cached inputs, optionally without appending to history."""
        if update_history:
            self._update_states(newton_states, contacts, joint_f)
        else:
            NeuralSolver._update_states(self, newton_states, contacts, joint_f)

    def _update_states(self, newton_states: State, contacts: Contacts, joint_f):
        super()._update_states(newton_states, contacts, joint_f)
        self.states_history.append(
            {
                "root_body_q": self.root_body_q.clone(),
                "states": self.states.clone(),
                "states_embedding": self.states_embedding.clone(),
                "joint_f": self.joint_f.clone(),
                "gravity_dir": self.gravity_dir.clone(),
                **self.contacts,
            }
        )

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

        # assemble the model inputs in world frame
        model_inputs = torch.utils.data.default_collate(list(self.states_history))
        for key in model_inputs:
            model_inputs[key] = model_inputs[key].permute(1, 0, 2)

        return self.process_neural_model_inputs(model_inputs)
