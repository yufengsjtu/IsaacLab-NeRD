# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .neural_solver import NeuralSolver


class RNNNeuralSolver(NeuralSolver):
    """Neural solver variant that manages recurrent hidden state resets."""

    def __init__(self, reset_seq_length: int = 1, **kwargs):
        self.reset_seq_length = reset_seq_length
        self._step_count = 0
        super().__init__(**kwargs)

    def reset(self):
        self._step_count = 0
        self.neural_model.init_rnn(self.num_envs)

    def _before_model_forward(self):
        if self._step_count % self.reset_seq_length == 0:
            self.neural_model.reset_rnn_hidden_states()
            self._step_count = 0
        self._step_count += 1
