# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE.md for details].

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

class RunningMeanStd(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        device = 'cuda:0'
    ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, dtype = torch.float32, device = device))
        self.register_buffer('var', torch.ones(shape, dtype = torch.float32, device = device))
        self.register_buffer('count', torch.tensor(epsilon, dtype = torch.float32, device = device))

    @torch.no_grad()
    def update(
        self,
        arr: torch.tensor,
        batch_dim = False,
        time_dim = False
    ) -> None:
        mean_dims = [i for i in range(int(batch_dim) + int(time_dim))]
        batch_mean = torch.mean(arr, dim = mean_dims)
        batch_var = torch.var(arr, dim = mean_dims, unbiased = False)
        batch_count = np.prod(arr.shape[:int(batch_dim) + int(time_dim)])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: torch.tensor,
        batch_var: torch.tensor,
        batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a + m_b +
            torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.fill_(new_count)

    def normalize(self, arr:torch.tensor, un_norm = False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
        else:
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean
        return result
