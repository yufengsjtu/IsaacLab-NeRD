# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rough dataset terrain-level sampling."""

from types import SimpleNamespace

import torch
from isaaclab_neural.envs.anymal_c_dataset_gen_cfg import AnymalCDatasetGenRoughEnvCfg


def test_rough_dataset_config_uses_uniform_terrain_level_sampling() -> None:
    cfg = AnymalCDatasetGenRoughEnvCfg()

    assert cfg.scene.terrain.max_init_terrain_level is None
    assert cfg.scene.terrain.terrain_generator.curriculum is True
    assert cfg.curriculum.terrain_levels.func.__name__ == "_sample_terrain_levels_uniform"


def test_uniform_terrain_level_sampling_preserves_types_and_updates_origins() -> None:
    cfg = AnymalCDatasetGenRoughEnvCfg()
    sample_levels = cfg.curriculum.terrain_levels.func
    terrain_origins = torch.arange(5 * 3 * 3, dtype=torch.float32).reshape(5, 3, 3)
    terrain = SimpleNamespace(
        terrain_levels=torch.tensor([0, 1, 2, 3, 4, 0]),
        terrain_types=torch.tensor([0, 0, 1, 1, 2, 2]),
        terrain_origins=terrain_origins,
        env_origins=torch.zeros(6, 3),
        max_terrain_level=5,
    )
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))
    env_ids = torch.tensor([0, 2, 5])
    original_levels = terrain.terrain_levels.clone()
    original_types = terrain.terrain_types.clone()

    torch.manual_seed(7)
    expected_levels = torch.randint_like(terrain.terrain_levels[env_ids], terrain.max_terrain_level)
    torch.manual_seed(7)
    mean_level = sample_levels(env, env_ids)

    torch.testing.assert_close(terrain.terrain_levels[env_ids], expected_levels)
    torch.testing.assert_close(terrain.terrain_levels[[1, 3, 4]], original_levels[[1, 3, 4]])
    torch.testing.assert_close(terrain.terrain_types, original_types)
    torch.testing.assert_close(
        terrain.env_origins[env_ids],
        terrain_origins[expected_levels, original_types[env_ids]],
    )
    torch.testing.assert_close(mean_level, terrain.terrain_levels.float().mean())
