# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NeRD neural solver."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .nerd_newton_manager import NewtonNerdManager


ContactMode = Literal["fixed_ground", "newton_native"]
"""Contact source used to build neural-model inputs."""

DEFAULT_NUM_CONTACTS_PER_ENV = 64
"""Default Newton-native contact capacity per environment."""


def normalize_contact_mode(contact_mode: ContactMode | str) -> Literal["newton_native"]:
    """Reject the deprecated contact mode and return the supported mode.

    Args:
        contact_mode: Requested contact source convention.

    Returns:
        The validated ``"newton_native"`` contact mode.

    Raises:
        ValueError: If :paramref:`contact_mode` is deprecated or not recognized.
    """
    if contact_mode == "fixed_ground":
        warnings.warn(
            "'fixed_ground' contact mode is deprecated and unsupported.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise ValueError(
            "'fixed_ground' and 'newton_native' use incompatible contact representations. "
            "Select 'newton_native' and use a matching dataset and checkpoint."
        )
    if contact_mode == "newton_native":
        return "newton_native"
    raise ValueError(f"Unsupported contact mode: {contact_mode}")


ContactPackingPolicy = Literal[
    "stable_index",
    "penetration_priority",
    "random",
    "force_priority",
    "body_round_robin_pair_atomic",
]
"""Policy for ordering Newton native contacts before packing into fixed slots."""

ContactRepresentation = Literal["flat", "contact_tokens"]
"""Contact encoding used by the trained neural model."""

ContactFilter = Literal["none", "solver_active", "solver_active_movable"]
"""Selection applied to Newton native pairs before fixed-capacity packing."""

ContactTokenFrame = Literal["root", "owner"]
"""Frame used to express packed contact-token geometry at train/inference time."""


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

    use_feature_cache: bool = False
    """Whether to cache encoded Transformer inputs during sequential inference."""

    use_kv_cache: bool = False
    """Whether to use incremental RoPE Transformer inference with a per-environment KV cache."""

    use_frame_cache: bool = False
    """Whether ``TransformerNeuralSolver`` preprocesses and encodes only the newest frame each step.

    Earlier frames' encodings are reused and the Transformer still attends over the full window, so predictions
    (including auxiliary force heads) equal uncached inference up to floating-point roundoff. Requires per-frame
    coordinate anchoring (``anchor_frame_step="every"`` or world-frame states) and excludes the feature/KV caches.
    """

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

    num_contacts_per_env: int = DEFAULT_NUM_CONTACTS_PER_ENV
    """Fixed contact slots per environment expected by the trained neural model.

    Set to zero for contact-free systems.
    """

    contact_mode: ContactMode = "newton_native"
    """Contact input mode.

    ``"newton_native"`` uses Newton's ``CollisionPipeline`` plus
    ``NewtonContactAdapter`` to pack dynamic contacts into fixed-size neural inputs.
    The deprecated ``"fixed_ground"`` value emits :class:`DeprecationWarning`
    and is rejected because its contact representation is incompatible.
    """

    contact_packing_policy: ContactPackingPolicy = "penetration_priority"
    """Policy for ordering Newton native contacts before packing into fixed slots."""

    contact_representation: ContactRepresentation = "flat"
    """Contact encoding consumed by the trained model."""

    contact_filter: ContactFilter = "none"
    """Contact selection applied before packing.

    ``"solver_active"`` keeps native pairs with negative signed surface
    separation [m]. ``"solver_active_movable"`` also removes pairs where both
    sides are immovable, using verified body mobility. ``"none"`` preserves all candidates.
    These filters affect neural inputs, not native collision solving.
    """

    max_contact_tokens: int = 64
    """Maximum directed contact tokens per environment when ``contact_representation='contact_tokens'``."""

    contact_token_frame: ContactTokenFrame = "root"
    """Frame used to express packed contact-token geometry.

    ``"root"`` rotates token points, normals, lever arms, and relative
    velocities into the articulation root (base) frame. ``"owner"`` uses each
    token's primary body pose instead. Stored native pairs remain world-frame.
    Force labels stay in the root-frame convention used by the force head.
    """

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

    def __post_init__(self) -> None:
        """Validate the configured contact mode after initialization."""
        self.contact_mode = normalize_contact_mode(self.contact_mode)
        if self.num_contacts_per_env < 0:
            raise ValueError("num_contacts_per_env must be non-negative.")
        if self.contact_filter not in {"none", "solver_active", "solver_active_movable"}:
            raise ValueError(f"Unsupported contact_filter: {self.contact_filter!r}.")
        if self.contact_filter == "solver_active_movable" and self.contact_representation != "contact_tokens":
            raise ValueError("solver_active_movable requires contact-token representation.")
        if self.contact_token_frame not in {"root", "owner"}:
            raise ValueError(f"Unsupported contact_token_frame: {self.contact_token_frame!r}.")
        if self.contact_representation == "contact_tokens":
            if self.num_contacts_per_env == 0:
                raise ValueError("contact_representation='contact_tokens' requires a positive contact capacity.")
            if self.max_contact_tokens <= 0:
                raise ValueError("max_contact_tokens must be positive for contact_representation='contact_tokens'.")
