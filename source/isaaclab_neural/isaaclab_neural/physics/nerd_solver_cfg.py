# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NeRD neural solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .nerd_newton_manager import NewtonNerdManager


ContactMode = Literal["fixed_ground", "newton_native"]
"""Contact source used to build neural-model inputs."""

ContactPackingPolicy = Literal[
    "stable_index",
    "penetration_priority",
    "random",
    "force_priority",
    "body_round_robin_pair_atomic",
]
"""Policy for ordering Newton native contacts before packing into fixed slots."""

ContactRepresentation = Literal["flat", "contact_tokens", "raw15_tokens", "active15_tokens"]
"""Contact encoding used by the trained neural model."""


@configclass
class NerdSolverCfg(NewtonSolverCfg):
    """Configuration for NeRD neural solver integration with Newton.

    The config describes how a trained NeRD model should be used as a Newton
    solver. It intentionally keeps runtime objects (Newton model, contacts, and
    the instantiated torch module) out of the dataclass so ``NewtonManager`` can
    construct them after USD parsing and collision setup are complete.
    """

    class_type: type[NewtonNerdManager] | str = "{DIR}.nerd_newton_manager:NewtonNerdManager"
    """Manager class for the NeRD neural solver."""

    solver_type: str = "nerd"
    """Solver type. Can be ``"nerd"``."""

    name: str = "NeuralSolver"
    """Neural solver class name registered in ``isaaclab_neural.solvers.factory``."""

    num_states_history: int = 1
    """Number of historical states/actions passed to ``TransformerNeuralSolver`` models."""

    reset_seq_length: int = 1
    """Step interval used by ``RNNNeuralSolver`` to reset recurrent hidden state."""

    neural_model_path: str | None = None
    """Path to a serialized neural model checkpoint.

    This is a declaration only until ``NewtonManager`` adds NeRD model loading.
    Callers may alternatively inject an already-instantiated torch module during
    solver construction.
    """

    neural_model_cfg: dict | None = None
    """Optional model architecture config used to instantiate the neural model before loading weights."""

    num_contacts_per_env: int = 0
    """Fixed contact slots per environment expected by the trained neural model."""

    contact_mode: ContactMode = "fixed_ground"
    """Contact input mode.

    ``"fixed_ground"`` uses NeRD's original fixed-size abstract contact buffer.
    ``"newton_native"`` uses Newton's ``CollisionPipeline`` plus
    ``NewtonContactAdapter`` to pack dynamic contacts into fixed-size neural
    inputs.
    """

    contact_packing_policy: ContactPackingPolicy = "penetration_priority"
    """Policy for ordering Newton native contacts before packing into fixed slots."""

    contact_representation: ContactRepresentation = "flat"
    """Contact encoding consumed by the trained model."""

    max_contact_tokens: int = 64
    """Maximum padded contact tokens per environment for either token representation."""

    states_frame: Literal["world", "body", "body_translation_only"] = "body"
    """Frame used to express neural-model states."""

    anchor_frame_step: Literal["first", "last", "every"] = "every"
    """Anchor frame step used when ``states_frame`` is body-relative."""

    states_embedding_type: Literal["identical", "sinusoidal"] | None = None
    """State embedding type used by the trained model."""

    prediction_type: Literal["absolute", "relative", "acceleration"] = "relative"
    """Prediction target convention used by the trained model."""

    orientation_prediction_parameterization: Literal["quaternion", "exponential", "naive"] = "quaternion"
    """Quaternion prediction parameterization used for orientation DOFs."""

    min_contact_event_threshold: float | None = None
    """Minimum contact depth threshold used when computing contact masks.

    If ``None``, ``NeuralSolver`` uses its package default.
    """
