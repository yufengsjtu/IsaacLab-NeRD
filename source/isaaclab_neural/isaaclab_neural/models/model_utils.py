# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE.md for details].

import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_activation_func(activation_name):
    if activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "identity":
        return nn.Identity()
    elif activation_name.lower() == "silu":
        return nn.SiLU()
    else:
        raise NotImplementedError(f"Activation func {activation_name} not defined")
