# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configs for NeRD-backed Unitree G1 policy learning."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1FlatNeRDPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for flat G1 policy learning on NeRD."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    clip_actions = None
    experiment_name = "g1_flat_nerd"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RoughNeRDPPORunnerCfg(G1FlatNeRDPPORunnerCfg):
    """PPO runner config for rough G1 policy learning on NeRD."""

    experiment_name = "g1_rough_nerd"


@configclass
class G129DofFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """Stock flat G1 runner with a separate experiment name for the 29-DoF robot."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_29dof_flat"


@configclass
class G129DofPosturePPORunnerCfg(G1FlatPPORunnerCfg):
    """Stock flat G1 runner for the 29-DoF height and torso posture task."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_29dof_posture"


def g1_mbpo_runner_cfg(profile: str) -> RslRlOnPolicyRunnerCfg:
    """Resolve an explicit rough MBPO runner without changing legacy registered-task defaults.

    Args:
        profile: ``stock_rough`` for the default Isaac Lab runner, or ``nerd_legacy``.

    Returns:
        A fresh runner configuration with all preset values resolved.

    Raises:
        ValueError: The requested runner profile is unsupported.
    """
    from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    if profile == "stock_rough":
        return resolve_presets(G1RoughPPORunnerCfg())
    if profile == "nerd_legacy":
        return G1RoughNeRDPPORunnerCfg()
    raise ValueError(f"Unsupported rough MBPO runner profile: {profile}")
