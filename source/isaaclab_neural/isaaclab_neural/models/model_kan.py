# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import List

import torch
from torch import nn, Tensor


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_num: int = 5,
        order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=torch.nn.SiLU,
        grid_range: List[float] = [-1, 1],
    ):
        try:
            from warpkan.torch.kan import KANLinear
        except ImportError:
            raise ImportError(
                "Please install warpkan to use KAN, see https://gitlab-master.nvidia.com/eheiden/warpkan"
            )

        super(KAN, self).__init__()
        self.grid_num = grid_num
        self.order = order

        self.layers = torch.nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_dim,
                    out_dim,
                    order,
                    grid_num,
                    grid_range,
                    scale_noise,
                    scale_base,
                    scale_spline,
                    enable_standalone_scale_spline,
                    base_activation,
                )
            )

    def forward(self, x: Tensor, deterministic=False):
        for layer in self.layers:
            x = layer(x)
        return x

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
