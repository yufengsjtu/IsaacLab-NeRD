# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment adapters for NeRD-backed Isaac Lab environments."""

import gymnasium as gym

from isaaclab_tasks.contrib.velocity.config.anymal_c import agents as anymal_c_agents
from isaaclab_tasks.core.cartpole import agents as cartpole_agents
from isaaclab_tasks.core.lift.config.franka import agents as franka_lift_agents
from isaaclab_tasks.core.reach.config.franka import agents as franka_reach_agents
from isaaclab_tasks.core.reorient.config.allegro_hand import agents as allegro_reorient_agents

from .agents import anymal_c_nerd_rsl_rl_ppo_cfg as anymal_c_nerd_agents
from .allegro_reorient_nerd_env import (
    AllegroReorientDatasetGenEnvCfg,
    AllegroReorientPolicyEnvCfg,
    NerdAllegroReorientEnv,
    NerdAllegroReorientEnvCfg,
)
from .anymal_c_dataset_gen_cfg import AnymalCDatasetGenFlatEnvCfg, AnymalCDatasetGenRoughEnvCfg
from .anymal_nerd_env import (
    AnymalCFlatPolicyEnvCfg,
    NerdAnymalCFlatControlStepEnvCfg,
    NerdAnymalCFlatEnv,
    NerdAnymalCFlatEnvCfg,
    NerdAnymalCRoughEnv,
    NerdAnymalCRoughEnvCfg,
)
from .cartpole_nerd_env import NerdCartpoleEnv, NerdCartpoleEnvCfg
from .franka_lift_nerd_env import (
    FrankaLiftCubeDatasetGenEnvCfg,
    NerdFrankaLiftEnv,
    NerdFrankaLiftEnvCfg,
)
from .franka_reach_nerd_env import (
    FrankaReachDatasetGenEnvCfg,
    FrankaReachPolicyEnvCfg,
    NerdFrankaReachEnv,
    NerdFrankaReachEnvCfg,
)
from .keyboard_nerd_env import NerdSO101KeyboardEnv, NerdSO101KeyboardEnvCfg, NerdSO101KeyboardRandomizedEnvCfg
from .neural_env_wrapper import NerdManagerBasedRLEnv, NeuralEnvAdapter

gym.register(
    id="Isaac-Cartpole-NeRD-v0",
    entry_point=f"{NerdCartpoleEnv.__module__}:NerdCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdCartpoleEnvCfg.__module__}:NerdCartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{cartpole_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{cartpole_agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{cartpole_agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerWithSymmetryCfg"
        ),
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
        "rsl_rl_cfg_entry_point": (f"{anymal_c_nerd_agents.__name__}:AnymalCFlatNeRDPPORunnerCfg"),
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg"
        ),
        "skrl_cfg_entry_point": f"{anymal_c_agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Policy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{AnymalCFlatPolicyEnvCfg.__module__}:AnymalCFlatPolicyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-ControlStep-NeRD-v0",
    entry_point=f"{NerdAnymalCFlatEnv.__module__}:NerdAnymalCFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdAnymalCFlatControlStepEnvCfg.__module__}:NerdAnymalCFlatControlStepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
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
    id="Isaac-Reach-Franka-NeRD-v0",
    entry_point=f"{NerdFrankaReachEnv.__module__}:NerdFrankaReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdFrankaReachEnvCfg.__module__}:NerdFrankaReachEnvCfg",
        "rl_games_cfg_entry_point": f"{franka_reach_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{franka_reach_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{franka_reach_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reach-Franka-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{FrankaReachDatasetGenEnvCfg.__module__}:FrankaReachDatasetGenEnvCfg",
        "rsl_rl_cfg_entry_point": f"{franka_reach_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-Franka-Policy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{FrankaReachPolicyEnvCfg.__module__}:FrankaReachPolicyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{franka_reach_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-NeRD-v0",
    entry_point=f"{NerdFrankaLiftEnv.__module__}:NerdFrankaLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdFrankaLiftEnvCfg.__module__}:NerdFrankaLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{franka_lift_agents.__name__}.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{FrankaLiftCubeDatasetGenEnvCfg.__module__}:FrankaLiftCubeDatasetGenEnvCfg",
        "rsl_rl_cfg_entry_point": f"{franka_lift_agents.__name__}.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

_ALLEGRO_REORIENT_RSL_RL_CFG = f"{allegro_reorient_agents.__name__}.rsl_rl_ppo_cfg:AllegroHandManagerPPORunnerCfg"

gym.register(
    id="Isaac-Reorient-Cube-Allegro-Policy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{AllegroReorientPolicyEnvCfg.__module__}:AllegroReorientPolicyEnvCfg",
        "rsl_rl_cfg_entry_point": _ALLEGRO_REORIENT_RSL_RL_CFG,
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Allegro-Dataset-Gen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{AllegroReorientDatasetGenEnvCfg.__module__}:AllegroReorientDatasetGenEnvCfg",
        "rsl_rl_cfg_entry_point": _ALLEGRO_REORIENT_RSL_RL_CFG,
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Allegro-NeRD-v0",
    entry_point=f"{NerdAllegroReorientEnv.__module__}:NerdAllegroReorientEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdAllegroReorientEnvCfg.__module__}:NerdAllegroReorientEnvCfg",
        "rsl_rl_cfg_entry_point": _ALLEGRO_REORIENT_RSL_RL_CFG,
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="IsaacContrib-Keyboard-SO101-NeRD",
    entry_point=f"{NerdSO101KeyboardEnv.__module__}:NerdSO101KeyboardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdSO101KeyboardEnvCfg.__module__}:NerdSO101KeyboardEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.keyboard.agents.rsl_rl_ppo_cfg:SO101PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Keyboard-SO101-NeRD-Randomized",
    entry_point=f"{NerdSO101KeyboardEnv.__module__}:NerdSO101KeyboardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{NerdSO101KeyboardRandomizedEnvCfg.__module__}:NerdSO101KeyboardRandomizedEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.keyboard.agents.rsl_rl_ppo_cfg:SO101PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


__all__ = [
    "AllegroReorientDatasetGenEnvCfg",
    "AllegroReorientPolicyEnvCfg",
    "NerdAllegroReorientEnv",
    "NerdAllegroReorientEnvCfg",
    "NerdSO101KeyboardEnv",
    "NerdSO101KeyboardEnvCfg",
    "NerdSO101KeyboardRandomizedEnvCfg",
    "AnymalCDatasetGenFlatEnvCfg",
    "AnymalCDatasetGenRoughEnvCfg",
    "AnymalCFlatPolicyEnvCfg",
    "FrankaLiftCubeDatasetGenEnvCfg",
    "FrankaReachDatasetGenEnvCfg",
    "FrankaReachPolicyEnvCfg",
    "NerdAnymalCFlatControlStepEnvCfg",
    "NerdAnymalCFlatEnv",
    "NerdAnymalCFlatEnvCfg",
    "NerdAnymalCRoughEnv",
    "NerdAnymalCRoughEnvCfg",
    "NerdCartpoleEnv",
    "NerdCartpoleEnvCfg",
    "NerdFrankaLiftEnv",
    "NerdFrankaLiftEnvCfg",
    "NerdFrankaReachEnv",
    "NerdFrankaReachEnvCfg",
    "NerdManagerBasedRLEnv",
    "NeuralEnvAdapter",
]
