# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment adapters for NeRD-backed Isaac Lab environments."""

import gymnasium as gym
from isaaclab_tasks.core.cartpole import agents as cartpole_agents
from isaaclab_tasks.contrib.velocity.config.anymal_c import agents as anymal_c_agents

from .anymal_nerd_env import NerdAnymalCFlatEnv, NerdAnymalCFlatEnvCfg
from .anymal_c_dataset_gen_cfg import AnymalCDatasetGenFlatEnvCfg, AnymalCDatasetGenRoughEnvCfg
from .anymal_nerd_env import NerdAnymalCRoughEnv, NerdAnymalCRoughEnvCfg
from .cartpole_nerd_env import NerdCartpoleEnv, NerdCartpoleEnvCfg
from .g1_dataset_gen_cfg import G1DatasetGenFlatEnvCfg, G1DatasetGenRoughEnvCfg
from .g1_nerd_env import NerdG1FlatEnv, NerdG1FlatEnvCfg, NerdG1RoughEnv, NerdG1RoughEnvCfg
from .neural_env_wrapper import NerdManagerBasedRLEnv, NeuralEnvAdapter
from .agents import anymal_c_nerd_rsl_rl_ppo_cfg as anymal_c_nerd_agents
from .agents import g1_nerd_rsl_rl_ppo_cfg as g1_nerd_agents

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
        "rsl_rl_cfg_entry_point": (
            f"{anymal_c_nerd_agents.__name__}:AnymalCFlatNeRDPPORunnerCfg"
        ),
        "rsl_rl_with_symmetry_cfg_entry_point": f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{anymal_c_agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-NeRD-v0",
    entry_point=f"{NerdAnymalCRoughEnv.__module__}:NerdAnymalCRoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdAnymalCRoughEnvCfg.__module__}:NerdAnymalCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{anymal_c_agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "skrl_cfg_entry_point": f"{anymal_c_agents.__name__}:skrl_rough_ppo_cfg.yaml",
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

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{AnymalCDatasetGenRoughEnvCfg.__module__}:AnymalCDatasetGenRoughEnvCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-NeRD-v0",
    entry_point=f"{NerdG1FlatEnv.__module__}:NerdG1FlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdG1FlatEnvCfg.__module__}:NerdG1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G1FlatNeRDPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-NeRD-v0",
    entry_point=f"{NerdG1RoughEnv.__module__}:NerdG1RoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdG1RoughEnvCfg.__module__}:NerdG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G1RoughNeRDPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{G1DatasetGenFlatEnvCfg.__module__}:G1DatasetGenFlatEnvCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{G1DatasetGenRoughEnvCfg.__module__}:G1DatasetGenRoughEnvCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29DoF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:G129DofFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G129DofFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-29DoF-Posture-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:G129DofPostureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G129DofPosturePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29DoF-NeRD-v0",
    entry_point=f"{NerdG1FlatEnv.__module__}:NerdG1FlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:NerdG129DofFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G129DofFlatNeRDPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-29DoF-Posture-NeRD-v0",
    entry_point=f"{NerdG1FlatEnv.__module__}:NerdG1FlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:NerdG129DofPostureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G129DofPostureNeRDPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29DoF-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:G129DofDatasetGenFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{g1_nerd_agents.__name__}:G129DofFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-29DoF-BFM-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:G129DofBfmEnvCfg"},
)

gym.register(
    id="Isaac-G1-29DoF-BFM-Perturbed-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_neural.envs.g1_29dof_env_cfg:G129DofBfmPerturbedEnvCfg"},
)


def register_gym_tasks() -> None:
    """Register NeRD gym ids. Safe to use as ``--external_callback``.

    Importing this module already calls :func:`gymnasium.register`. This
    function exists so ``./isaaclab.sh train`` / ``play`` can load those ids
    before they call :func:`gymnasium.spec`.
    """
    return None


__all__ = [
    "AnymalCDatasetGenFlatEnvCfg",
    "AnymalCDatasetGenRoughEnvCfg",
    "G1DatasetGenFlatEnvCfg",
    "G1DatasetGenRoughEnvCfg",
    "NerdAnymalCFlatEnv",
    "NerdAnymalCFlatEnvCfg",
    "NerdAnymalCRoughEnv",
    "NerdAnymalCRoughEnvCfg",
    "NerdCartpoleEnv",
    "NerdCartpoleEnvCfg",
    "NerdG1FlatEnv",
    "NerdG1FlatEnvCfg",
    "NerdG1RoughEnv",
    "NerdG1RoughEnvCfg",
    "NerdManagerBasedRLEnv",
    "NeuralEnvAdapter",
    "register_gym_tasks",
]
