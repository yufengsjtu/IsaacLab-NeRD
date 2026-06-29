# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from .neural_solver import NeuralSolver


class RNNNeuralSolver(NeuralSolver):
    """Neural solver variant that manages recurrent hidden state resets."""

    def __init__(self, reset_seq_length: int = 1, **kwargs):
        self.reset_seq_length = reset_seq_length
        self._step_count = 0
        super().__init__(**kwargs)

    def reset(self, env_ids=None):
        """Reset recurrent state globally or for selected environments."""
        self._step_count = 0
        if env_ids is None:
            self.neural_model.init_rnn(self.num_envs)
            return

        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() > 0:
            self.neural_model.reset_rnn_hidden_states(batch_indices=env_ids)

    def _normalize_env_ids(self, env_ids) -> torch.Tensor:
        """Return env ids as a 1-D tensor on the solver torch device."""
        if isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.torch_device, dtype=torch.long)
        else:
            normalized = torch.as_tensor(env_ids, device=self.torch_device, dtype=torch.long)
        return normalized.reshape(-1)

    def _before_model_forward(self):
        if self._step_count % self.reset_seq_length == 0:
            self.neural_model.reset_rnn_hidden_states()
            self._step_count = 0
        self._step_count += 1
