# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE.md for details].

import numpy as np
import torch
import torch.nn as nn

from isaaclab_neural.models import model_utils
from isaaclab_neural.models.spatial_softmax import SpatialSoftArgmax


class MLPBase(nn.Module):
    def __init__(self, in_features, network_cfg, device="cuda:0"):
        super().__init__()

        self.device = device

        layer_sizes = network_cfg["layer_sizes"]
        modules = []
        for i in range(len(layer_sizes)):
            modules.append(nn.Linear(in_features, layer_sizes[i]))
            modules.append(model_utils.get_activation_func(network_cfg["activation"]))
            if network_cfg.get("layernorm", False):
                modules.append(torch.nn.LayerNorm(layer_sizes[i]))
            in_features = layer_sizes[i]

        self.body = nn.Sequential(*modules).to(device)
        self.out_features = in_features

    def forward(self, inputs):
        return self.body(inputs)

    def to(self, device):
        self.device = device
        self.body.to(device)


class CNNBase(nn.Module):
    def __init__(self, input_shape, network_cfg, device="cuda:0"):
        super().__init__()

        assert len(input_shape) == 3  # (feature_channels, rows, cols)

        self.device = device

        in_channels = input_shape[0]

        kernel_sizes = network_cfg["kernel_sizes"]
        layer_sizes = network_cfg["layer_sizes"]
        num_layers = len(kernel_sizes)
        stride_sizes = network_cfg.get("stride_sizes", [1] * num_layers)
        padding_sizes = network_cfg.get("padding_sizes", [0] * num_layers)

        num_layers = len(kernel_sizes)
        assert len(layer_sizes) == num_layers and len(stride_sizes) == num_layers and len(padding_sizes) == num_layers

        out_shape = np.array(input_shape[1:])
        modules = []
        for i in range(len(kernel_sizes)):
            modules.append(
                nn.Conv2d(
                    in_channels, layer_sizes[i], kernel_sizes[i], stride=stride_sizes[i], padding=padding_sizes[i]
                )
            )
            modules.append(model_utils.get_activation_func(network_cfg["activation"]))
            if network_cfg.get("layernorm", False):
                modules.append(nn.LayerNorm(layer_sizes[i]))
            in_channels = layer_sizes[i]
            out_shape = (out_shape + 2 * padding_sizes[i] - kernel_sizes[i]) // (stride_sizes[i]) + 1

        pool_cfg = network_cfg["pooling"]
        if pool_cfg["name"] == "MLP":
            out_features = pool_cfg["out_features"]
            modules.append(nn.Flatten())
            modules.append(nn.Linear(out_shape[0] * out_shape[1] * in_channels, out_features))
            modules.append(model_utils.get_activation_func(pool_cfg["activation"]))
            self.out_features = out_features
        elif pool_cfg["name"] == "SpatialSoftmax":
            modules.append(SpatialSoftArgmax(normalize=pool_cfg.get("normalize", False)))
            self.out_features = in_channels * 2
        else:
            raise NotImplementedError

        self.body = nn.Sequential(*modules).to(device)

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view((-1, *shape[-3:]))
        out = self.body(inputs).view((*shape[:-3], self.out_features))
        return out

    def to(self, device):
        self.device = device
        self.body.to(device)


class LSTMBase(nn.Module):
    def __init__(self, in_features, network_cfg, device="cuda:0"):
        super().__init__()

        self.hidden_size = network_cfg["hidden_size"]
        self.num_layers = network_cfg["num_layers"]
        self.device = device
        self.lstm = nn.LSTM(
            input_size=in_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )
        self.lstm.to(self.device)

    def forward(self, x):
        output, hidden_states_next = self.lstm(x, self.hidden_states)
        self.hidden_states = hidden_states_next
        return output

    def initialize_hidden_states(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device)
        self.hidden_states = (h, c)

    # Hidden_states is in shape (B, 2 * L * H)
    def reset_hidden_states(self, batch_indices=None, hidden_states=None):
        if hidden_states is None:
            if batch_indices is None:
                self.hidden_states[0][:, :, :] = 0.0
                self.hidden_states[1][:, :, :] = 0.0
            else:
                self.hidden_states[0][:, batch_indices, :] = 0.0
                self.hidden_states[1][:, batch_indices, :] = 0.0
        else:
            hidden_states_right_order = hidden_states.view(
                self.hidden_states[0].shape[1], 2, self.num_layers, self.hidden_size
            ).permute(1, 2, 0, 3)
            if batch_indices is None:
                self.hidden_states[0][:, :, :] = hidden_states_right_order[0]
                self.hidden_states[1][:, :, :] = hidden_states_right_order[1]
            else:
                self.hidden_states[0][:, batch_indices, :] = hidden_states_right_order[0][:, batch_indices, :]
                self.hidden_states[1][:, batch_indices, :] = hidden_states_right_order[1][:, batch_indices, :]

    # Get the hidden states in shape B, 2*L*H
    def get_hidden_states(self):
        hidden_states = (
            torch.stack((self.hidden_states[0], self.hidden_states[1]))
            .permute(2, 0, 1, 3)
            .view(self.hidden_states[0].shape[1], 2 * self.num_layers * self.hidden_size)
        )

        return hidden_states

    def hidden_states_size(self):
        return 2 * self.num_layers * self.hidden_size

    def to(self, device):
        self.device = device
        self.lstm.to(device)


class GRUBase(nn.Module):
    def __init__(self, in_features, network_cfg, device="cuda:0"):
        super().__init__()

        self.hidden_size = network_cfg["hidden_size"]
        self.num_layers = network_cfg["num_layers"]
        self.device = device
        self.gru = nn.GRU(
            input_size=in_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )
        self.gru.to(device)

    def forward(self, x):
        output, hidden_states_next = self.gru(x, self.hidden_states)
        self.hidden_states = hidden_states_next
        return output

    def initialize_hidden_states(self, batch_size):
        self.hidden_states = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device)

    # Hidden_states is in shape (B, L * H)
    def reset_hidden_states(self, batch_indices=None, hidden_states=None):
        if hidden_states is None:
            if batch_indices is None:
                self.hidden_states[:, :, :] = 0.0
            else:
                self.hidden_states[:, batch_indices, :] = 0.0
        else:
            hidden_states_right_order = hidden_states.view(
                self.hidden_states.shape[1], self.num_layers, self.hidden_size
            ).permute(1, 0, 2)
            if batch_indices is None:
                self.hidden_states[:, :, :] = hidden_states_right_order
            else:
                self.hidden_states[:, batch_indices, :] = hidden_states_right_order[:, batch_indices, :]

    # Get the hidden states in shape B, L*H
    def get_hidden_states(self):
        hidden_states = self.hidden_states.permute(1, 0, 2).view(
            self.hidden_states.shape[1], self.num_layers * self.hidden_size
        )
        return hidden_states

    def hidden_states_size(self):
        return self.num_layers * self.hidden_size

    def to(self, device):
        self.device = device
        self.gru.to(device)
