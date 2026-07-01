# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment adapters for NeRD-backed Isaac Lab environments."""

import gymnasium as gym
from isaaclab_tasks.manager_based.classic.cartpole import agents as cartpole_agents
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c import agents as anymal_c_agents

from .anymal_nerd_env import NerdAnymalCFlatEnv, NerdAnymalCFlatEnvCfg
from .anymal_c_dataset_gen_cfg import AnymalCDatasetGenFlatEnvCfg
from .cartpole_nerd_env import NerdCartpoleEnv, NerdCartpoleEnvCfg
from .neural_env_wrapper import NerdManagerBasedRLEnv, NeuralEnvAdapter

gym.register(
    id="Isaac-Cartpole-NeRD-v0",
    entry_point=f"{NerdCartpoleEnv.__module__}:NerdCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdCartpoleEnvCfg.__module__}:NerdCartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{cartpole_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{cartpole_agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{cartpole_agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{cartpole_agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{cartpole_agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-NeRD-v0",
    entry_point=f"{NerdAnymalCFlatEnv.__module__}:NerdAnymalCFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdAnymalCFlatEnvCfg.__module__}:NerdAnymalCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{anymal_c_agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{anymal_c_agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{AnymalCDatasetGenFlatEnvCfg.__module__}:AnymalCDatasetGenFlatEnvCfg",
    },
)


__all__ = [
    "NerdAnymalCFlatEnv",
    "NerdAnymalCFlatEnvCfg",
    "AnymalCDatasetGenFlatEnvCfg",
    "NerdCartpoleEnv",
    "NerdCartpoleEnvCfg",
    "NerdManagerBasedRLEnv",
    "NeuralEnvAdapter",
]
