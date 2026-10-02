# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warnings
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
import warp as wp
from newton import Contacts, Control, JointType, Model, State
from newton.solvers import SolverBase

from isaaclab_neural.contacts.contact_set_schema import CONTACT_FLAT_VALUE_FIELDS
from isaaclab_neural.contacts.newton_contact_adapter import NewtonContactAdapter
from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode
from isaaclab_neural.solvers.kernels import determine_angular_dofs
from isaaclab_neural.solvers.world_layout import resolve_neural_solver_layout
from isaaclab_neural.utils import newton_utils, torch_utils, warp_utils
from isaaclab_neural.utils.newton_utils import base_joint_type

INFERENCE_PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
"""Supported ``inference_precision`` values of online NeRD model inference."""


@contextmanager
def _tf32_matmul() -> Iterator[None]:
    """Temporarily allow TF32 CUDA matmuls and convolutions, restoring the previous flags."""
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def contact_tokens_to_model_frame(
    contact_tokens: torch.Tensor, frame_q: torch.Tensor, translation_only: bool, per_owner: bool
) -> torch.Tensor:
    """Transform world-frame contact-token geometry into the model frame.

    Args:
        contact_tokens: Packed tokens, shape ``[B, T, K, D]``.
        frame_q: Primary-body poses ``[B, T, P, 7]`` when ``per_owner`` (each token uses its owner body), otherwise
            one shared pose per frame ``[B, T, 7]``.
        translation_only: Whether to remove only the frame translation.
        per_owner: Whether ``frame_q`` holds per-primary-body poses.
    """
    from isaaclab_neural.contacts.contact_set_encoder import (
        gather_owner_body_q,
        transform_contact_tokens_to_body_frame,
    )

    if per_owner:
        frame_q = gather_owner_body_q(frame_q, contact_tokens)
    return transform_contact_tokens_to_body_frame(contact_tokens, frame_q, translation_only=translation_only)


def _quat_rotate_inverse_elementwise(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate ``vector`` by the inverse of xyzw ``quat`` with broadcastable elementwise operations only."""
    w = quat[..., 3:4]
    xyz = quat[..., :3]
    return (
        vector * (2.0 * w * w - 1.0)
        - torch.cross(xyz, vector, dim=-1) * w * 2.0
        + xyz * (xyz * vector).sum(dim=-1, keepdim=True) * 2.0
    )


def contact_tokens_to_model_frame_elementwise(
    contact_tokens: torch.Tensor, frame_q: torch.Tensor, translation_only: bool, per_owner: bool
) -> torch.Tensor:
    """Elementwise equivalent of :func:`contact_tokens_to_model_frame` for ``[B, T, K, D]`` tokens.

    The reference rotation forms its dot products with batched matrix products, which run as slow per-row
    matrix-vector kernels over many tokens; this version is fusible by ``torch.compile``. Results are equal up to
    floating-point roundoff.
    """
    from isaaclab_neural.contacts.contact_set_encoder import gather_owner_body_q
    from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_VALID_INDEX

    if per_owner:
        frame_q = gather_owner_body_q(frame_q, contact_tokens)
    else:
        frame_q = frame_q.unsqueeze(-2).expand(*contact_tokens.shape[:-1], 7)
    position, quat = frame_q[..., :3], frame_q[..., 3:7]
    points, normals, levers = contact_tokens[..., 4:7], contact_tokens[..., 7:10], contact_tokens[..., 10:13]
    gap, velocities = contact_tokens[..., 13:14], contact_tokens[..., 14:17]
    if translation_only:
        points = points - position
    else:
        points = _quat_rotate_inverse_elementwise(quat, points - position)
        normals = _quat_rotate_inverse_elementwise(quat, normals)
        levers = _quat_rotate_inverse_elementwise(quat, levers)
        velocities = _quat_rotate_inverse_elementwise(quat, velocities)
    transformed = torch.cat((contact_tokens[..., :4], points, normals, levers, gap, velocities), dim=-1)
    return transformed * (contact_tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5).unsqueeze(-1)


if TYPE_CHECKING:
    from isaaclab_neural.models.models import ModelMixedInputOutput

MIN_CONTACT_EVENT_THRESHOLD = 0.12


class NeuralSolver(SolverBase):
    """Integrator that uses a neural network to predict the next generalized state.

    Each Newton world is one environment. A world may contain multiple
    articulations when every world is a replica with a contiguous joint layout.
    The concatenated ``joint_q`` / ``joint_qd`` of that world is the model state.
    """

    def __init__(
        self,
        name="NeuralSolver",
        model: Model | None = None,
        contacts: Contacts | None = None,
        neural_model: torch.nn.Module | None = None,
        num_contacts_per_env: int = 0,
        states_frame: str | None = "body",
        anchor_frame_step: str | None = "every",
        states_embedding_type: str | None = None,
        prediction_type: str = "relative",
        orientation_prediction_parameterization: str = "quaternion",
        min_contact_event_threshold: float | None = None,
        contact_mode: Literal["fixed_ground", "newton_native"] = "newton_native",
        contact_adapter: NewtonContactAdapter | None = None,
        contact_representation: str = "flat",
        max_contact_tokens: int = 0,
        contact_filter: str = "none",
        contact_token_frame: str = "root",
        inference_precision: str = "fp32",
        inference_compile: bool = False,
        step_mode: str = "physics_frame",
        joint_target_input: bool = False,
    ):
        """
        Args:
            name (str, optional): Name used to identify this solver.
            model (Model): The Newton model storing static simulation metadata.
            contacts (Contacts): Optional contact buffer used to build neural-model inputs.
            neural_model (torch.nn.Module, optional): Neural network used for prediction.
            num_contacts_per_env (int, optional): Contact capacity per environment. Set to
                zero for contact-free systems.
            states_frame (str, optional): Frame used to express states; one of "world",
                "body", or "body_translation_only". Defaults to "body".
            anchor_frame_step (str, optional): Anchor frame step for body-frame states; one
                of "first", "last", or "every". Defaults to "every".
            states_embedding_type (Optional[str], optional): State embedding type; one of
                None, "identical", or "sinusoidal". Defaults to None.
            prediction_type (str, optional): Prediction target type; one of "absolute",
                "relative", or "acceleration". Defaults to "relative".
            orientation_prediction_parameterization (str, optional): Quaternion prediction
                parameterization; one of "quaternion", "exponential", or "naive".
            min_contact_event_threshold (float, optional): Minimum depth threshold used
                when computing contact masks. Defaults to MIN_CONTACT_EVENT_THRESHOLD.
            contact_mode: Contact source convention. Deprecated ``fixed_ground``
                inputs warn and are rejected as incompatible.
            contact_adapter: Adapter used by ``newton_native`` mode.
            contact_filter: Native-pair selection applied before packing.
            contact_token_frame: Frame for packed token geometry; ``"root"`` or
                ``"owner"``. Defaults to ``"root"``.
            inference_precision: Matmul precision of online model inference on CUDA; one of
                ``"fp32"``, ``"tf32"``, ``"fp16"`` or ``"bf16"`` (autocast). Defaults to ``"fp32"``.
            inference_compile: Compile fixed-shape steady-state inference steps with ``torch.compile`` where the
                solver supports it (frame-cached transformers). Defaults to False.
            step_mode: Time span of one prediction, ``"physics_frame"`` or ``"control_step"`` (see
                :attr:`NerdSolverCfg.step_mode`). Defaults to ``"physics_frame"``.
            joint_target_input: Expose Newton joint position targets to the model as ``joint_target_offset``.
                Defaults to False.
        """
        self.solver_name = name
        if model is None:
            raise ValueError("NeuralSolver requires a Newton model.")
        super().__init__(model)
        self.torch_device = wp.device_to_torch(model.device)
        self.neural_model: torch.nn.Module | None = neural_model
        if neural_model is not None:
            neural_model.to(self.torch_device)

        if min_contact_event_threshold is not None:
            self.min_contact_event_threshold = min_contact_event_threshold
        else:
            self.min_contact_event_threshold = MIN_CONTACT_EVENT_THRESHOLD

        layout = resolve_neural_solver_layout(model)
        self.dof_q_per_env = layout.dof_q_per_env
        self.dof_qd_per_env = layout.dof_qd_per_env
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.num_envs = layout.num_envs
        self.num_joints_per_env = layout.num_joints_per_env
        self.num_bodies_per_env = layout.num_bodies_per_env
        self.joint_f_dim = layout.joint_f_dim
        self.articulations_per_env = layout.articulations_per_env
        self.state_world_ids = torch.as_tensor(layout.state_world_ids, device=self.torch_device, dtype=torch.long)
        if sorted(self.state_world_ids.tolist()) != list(range(self.num_envs)):
            raise ValueError(f"Generalized state rows do not map one-to-one to Newton worlds: {self.state_world_ids}.")
        self.root_body_ids = torch.as_tensor(layout.root_body_ids, device=self.torch_device, dtype=torch.long)
        if torch.unique(self.root_body_ids).numel() != self.num_envs:
            raise ValueError(f"Articulation root body ids are not unique: {self.root_body_ids}.")
        if model.body_world is None:
            self.root_world_ids = self.state_world_ids.clone()
        else:
            body_world = torch.as_tensor(model.body_world.numpy(), device=self.torch_device, dtype=torch.long)
            self.root_world_ids = body_world.index_select(0, self.root_body_ids)
        if not torch.equal(self.root_world_ids, self.state_world_ids):
            raise ValueError(
                "Generalized-state and root-body rows map to different Newton worlds: "
                f"state={self.state_world_ids}, root={self.root_world_ids}."
            )
        self.num_contacts_per_env = num_contacts_per_env
        self.contact_mode = normalize_contact_mode(contact_mode)
        self.contact_representation = contact_representation
        self.max_contact_tokens = int(max_contact_tokens)
        if contact_filter not in {"none", "solver_active", "solver_active_movable"}:
            raise ValueError(f"Unsupported contact_filter: {contact_filter!r}.")
        self.contact_filter = contact_filter
        if contact_token_frame not in {"root", "owner"}:
            raise ValueError(f"Unsupported contact_token_frame: {contact_token_frame!r}.")
        self.contact_token_frame = contact_token_frame
        if inference_precision not in INFERENCE_PRECISIONS:
            raise ValueError(f"inference_precision must be one of {INFERENCE_PRECISIONS}, got {inference_precision!r}.")
        self.inference_precision = inference_precision
        self.inference_compile = bool(inference_compile)
        if step_mode not in ("physics_frame", "control_step"):
            raise ValueError(f"step_mode must be 'physics_frame' or 'control_step', got {step_mode!r}.")
        self.step_mode = step_mode
        self.joint_target_input = bool(joint_target_input)
        self.contact_adapter = contact_adapter
        self.primary_body_q = getattr(contact_adapter, "primary_body_q", None)
        adapter_contact_filter = getattr(contact_adapter, "contact_filter", "none")
        if contact_adapter is not None and adapter_contact_filter != self.contact_filter:
            raise ValueError(
                "NeuralSolver and NewtonContactAdapter must use the same contact_filter: "
                f"solver={self.contact_filter!r}, adapter={adapter_contact_filter!r}."
            )
        if self.num_contacts_per_env < 0:
            raise ValueError("num_contacts_per_env must be non-negative.")
        if self.contact_representation == "contact_tokens":
            if self.num_contacts_per_env == 0:
                raise ValueError("contact_representation='contact_tokens' requires a positive contact capacity.")
            if self.max_contact_tokens <= 0:
                raise ValueError("max_contact_tokens must be positive for contact_representation='contact_tokens'.")
        if self.num_contacts_per_env > 0 and self.contact_adapter is None:
            raise ValueError("NeuralSolver contact_mode='newton_native' requires a NewtonContactAdapter.")
        self.num_contact_bodies_per_env = getattr(
            self.contact_adapter, "num_primary_bodies_per_env", self.num_bodies_per_env
        )

        # initialize model input variables
        self.root_body_q = torch.empty((self.num_envs, 7), device=self.torch_device)
        self.root_body_qd = torch.empty((self.num_envs, 6), device=self.torch_device)
        self.states = torch.empty((self.num_envs, self.state_dim), device=self.torch_device)
        self.joint_f = torch.empty((self.num_envs, self.joint_f_dim), device=self.torch_device)
        self.joint_target_q = torch.zeros((self.num_envs, self.dof_q_per_env), device=self.torch_device)
        self.contact_inputs: dict[str, torch.Tensor] = (
            {} if self.contact_adapter is None else self.contact_adapter.to_neural_inputs()
        )

        self.gravity_dir = torch.zeros((self.num_envs, 3), device=self.torch_device)
        self.gravity_dir[:, self.model.up_axis] = -1.0

        self._build_dof_types()

        assert states_frame in ["world", "body", "body_translation_only"]
        self.states_frame = states_frame
        if states_frame == "body" or states_frame == "body_translation_only":
            assert anchor_frame_step in ["first", "last", "every"]
            self.anchor_frame_step = anchor_frame_step

        assert states_embedding_type in [None, "identical", "sinusoidal"]
        self.states_embedding_type = states_embedding_type
        self._init_state_embedding()

        assert prediction_type in ["absolute", "relative", "acceleration"]
        self.prediction_type = prediction_type

        assert orientation_prediction_parameterization in ["quaternion", "exponential", "naive"]
        self.orientation_prediction_parameterization = orientation_prediction_parameterization

        self._init_prediction()
        self._configure_auxiliary_prediction_cache()

    @property
    def contacts(self) -> Any:
        """Deprecated NeRD feature alias; use the typed ``contact_inputs`` mapping.

        This compatibility boundary retains the old dictionary API without storing
        neural features in Newton's native contact-buffer property.
        """
        warnings.warn(
            "NeuralSolver.contacts is deprecated; use NeuralSolver.contact_inputs for neural contact features.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.contact_inputs

    @contacts.setter
    def contacts(self, value: Any) -> None:
        warnings.warn(
            "NeuralSolver.contacts is deprecated; use NeuralSolver.contact_inputs for neural contact features.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(value, dict):
            raise TypeError("NeRD contact inputs must be a dictionary; pass native Newton contacts to step().")
        self.contact_inputs = value

    def _build_dof_types(self):
        joint_q_start = self.model.joint_q_start
        joint_qd_start = self.model.joint_qd_start
        joint_types = self.model.joint_type
        if joint_q_start is None or joint_qd_start is None or joint_types is None:
            raise ValueError("NeuralSolver requires Newton joint position, velocity, and type metadata.")
        # compute information about the joint dofs
        self.joint_q_end_wp = wp.empty(self.num_joints_per_env, dtype=int, device=self.device)
        self.is_angular_dof_wp = wp.empty(self.dof_q_per_env, dtype=bool, device=self.device)
        self.is_continuous_dof_wp = wp.zeros(self.state_dim, dtype=bool, device=self.device)
        # determine which dofs are angular and continuous
        wp.launch(
            determine_angular_dofs,
            dim=self.num_joints_per_env,
            inputs=[
                joint_types,
                joint_q_start,
                joint_qd_start,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
            ],
            outputs=[self.joint_q_end_wp, self.is_angular_dof_wp, self.is_continuous_dof_wp],
            device=self.model.device,
        )

        self.joint_q_start = joint_q_start.numpy()[: self.num_joints_per_env]
        self.joint_qd_start = joint_qd_start.numpy()[: self.num_joints_per_env]
        self.joint_q_end = self.joint_q_end_wp.numpy()
        self.joint_types = joint_types.numpy()[: self.num_joints_per_env]
        self.free_joint_indices = np.flatnonzero(self.joint_types == JointType.FREE)
        self.is_angular_dof = self.is_angular_dof_wp.numpy()
        self.is_continuous_dof = self.is_continuous_dof_wp.numpy()
        # Type of the first non-FIXED joint — i.e., the "base" joint after
        # skipping any leading FIXED root (e.g. Cartpole's world-attach).
        self.base_joint_type = base_joint_type(self.joint_types)

    """Compute the dimension of the input state embedding. """

    def _init_state_embedding(self):
        if self.states_embedding_type is None or self.states_embedding_type == "identical":
            self.state_embedding_dim = self.state_dim
        elif self.states_embedding_type == "sinusoidal":
            self.state_embedding_dim = self.state_dim + (self.is_angular_dof).sum().item()
        else:
            raise NotImplementedError

        self.states_embedding = torch.zeros((self.num_envs, self.state_embedding_dim), device=self.torch_device)

    """Compute the dimension of the prediction output."""

    def _init_prediction(self):
        if self.prediction_type == "absolute" or self.prediction_type == "relative":
            num_regular_dofs, num_sperical_joints = 0, 0
            for i in range(self.num_joints_per_env):
                if self.joint_types[i] == JointType.FREE:
                    num_regular_dofs += 3
                    num_sperical_joints += 1
                elif self.joint_types[i] == JointType.BALL:
                    num_sperical_joints += 1
                else:
                    num_regular_dofs += self.joint_q_end[i] - self.joint_q_start[i]
            if self.orientation_prediction_parameterization == "quaternion":
                self.prediction_dim = num_regular_dofs + num_sperical_joints * 4 + self.dof_qd_per_env
            elif self.orientation_prediction_parameterization == "exponential":
                self.prediction_dim = num_regular_dofs + num_sperical_joints * 3 + self.dof_qd_per_env
            elif self.orientation_prediction_parameterization == "naive":
                self.prediction_dim = num_regular_dofs + num_sperical_joints * 4 + self.dof_qd_per_env
            else:
                raise NotImplementedError
        elif self.prediction_type == "acceleration":
            self.prediction_dim = self.dof_qd_per_env
        else:
            raise NotImplementedError

    def set_neural_solver_model(self, neural_solver_model: torch.nn.Module) -> None:
        self.neural_model = neural_solver_model
        neural_solver_model.to(self.torch_device)
        self._configure_auxiliary_prediction_cache()

    def _require_neural_model(self) -> torch.nn.Module:
        neural_model = self.neural_model
        if neural_model is None:
            raise RuntimeError("Attach a neural model with set_neural_solver_model() before training or inference.")
        return neural_model

    def _configure_auxiliary_prediction_cache(self):
        """Create the online force cache only when the neural head is enabled."""
        predicts_forces = bool(
            self.neural_model is not None and getattr(self.neural_model, "predict_body_net_contact_forces", False)
        )
        if predicts_forces:
            model_num_bodies = getattr(self.neural_model, "num_bodies", None)
            if model_num_bodies != self.num_contact_bodies_per_env:
                raise ValueError(
                    "Neural model force-head body count does not match solver metadata: "
                    f"model={model_num_bodies}, solver={self.num_contact_bodies_per_env}."
                )
            self.predicted_body_net_contact_forces_b = torch.zeros(
                (self.num_envs, self.num_contact_bodies_per_env, 3),
                device=self.torch_device,
            )
        else:
            self.predicted_body_net_contact_forces_b = None
        if getattr(self.neural_model, "predict_finger_object_normal_forces", False) and (
            getattr(self, "use_kv_cache", False) or getattr(self, "use_feature_cache", False)
        ):
            raise ValueError("Filtered finger-force prediction requires uncached auxiliary inference.")
        self.predicted_finger_object_normal_forces_b = (
            torch.zeros((self.num_envs, 2, 3), device=self.torch_device)
            if getattr(self.neural_model, "predict_finger_object_normal_forces", False)
            else None
        )

        robot_enabled = getattr(self.neural_model, "predict_robot_net_normal_forces", False)
        if robot_enabled and (getattr(self, "use_kv_cache", False) or getattr(self, "use_feature_cache", False)):
            raise ValueError("Robot normal-force prediction requires uncached auxiliary inference.")
        self.predicted_robot_net_normal_forces_w = (
            torch.zeros((self.num_envs, 7, 3), device=self.torch_device) if robot_enabled else None
        )

    def train(self):
        self._require_neural_model().train()

    def eval(self):
        self._require_neural_model().eval()

    def reset(self, env_ids=None, world_mask=None, flags=None):
        """Reset solver auxiliary state.

        Isaac Lab 3 calls :meth:`newton.solvers.SolverBase.reset` as
        ``reset(state, world_mask=..., flags=0)`` from
        :meth:`~isaaclab_newton.physics.newton_manager.NewtonManager._reset_solver_internals`.
        That path has no NeRD internals to clear. History and force-head state
        are reset with ``env_ids``.

        Args:
            env_ids: Optional environment ids whose auxiliary predictions
                should be cleared. Clears all environments when omitted.
                Newton-style calls pass the Newton ``State`` here.
            world_mask: Unused Newton SolverBase reset mask.
            flags: Unused Newton SolverBase reset flags.
        """
        if world_mask is not None or flags is not None:
            return
        for name in (
            "predicted_body_net_contact_forces_b",
            "predicted_finger_object_normal_forces_b",
            "predicted_robot_net_normal_forces_w",
        ):
            predicted_forces = getattr(self, name, None)
            if predicted_forces is None:
                continue
            if env_ids is None:
                predicted_forces.zero_()
            else:
                indices = torch.as_tensor(env_ids, device=predicted_forces.device, dtype=torch.long)
                predicted_forces[indices] = 0.0

    def sync_from_newton(
        self,
        newton_states: State,
        contacts: Contacts | None,
        joint_f,
        *,
        update_history: bool = True,
        joint_target_q=None,
    ) -> None:
        """Synchronize cached neural-model inputs from the current Newton state.

        Args:
            newton_states: Newton state to read generalized and body states from.
            contacts: Contact buffer used to build neural-model inputs.
            joint_f: Joint force buffer to expose to the neural model.
            update_history: Whether stateful solvers should treat this as a
                history update. The base solver has no history, so this flag is
                accepted for subclass consistency.
            joint_target_q: Newton ``control.joint_target_q`` buffer, required when ``joint_target_input`` is set.
        """
        del update_history
        self._update_states(newton_states, contacts, joint_f, joint_target_q=joint_target_q)

    def translate_world_inputs(
        self, inputs: dict[str, torch.Tensor], translation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Translate raw trajectories and contact geometry between replicated worlds.

        Joint angles, free-body orientations, velocities, forces, contact identities,
        and padding are preserved. Static scene geometry is not moved. Use this when
        assigning recorded robot/object trajectories to a different replica's table.

        Args:
            inputs: Raw world-frame states and optional contact/history fields with
                leading batch dimension. Unmodified fields are shared with the input.
            translation: Destination minus source world origin [m], shape ``[B, 3]``.

        Returns:
            Inputs with all free-joint positions and world-space points translated,
            without mutating the recorded tensors.
        """
        if translation.shape != (inputs["states"].shape[0], 3):
            raise ValueError("World translation must have shape [batch_size, 3].")
        result = dict(inputs)
        for key in ("states", "next_states"):
            if key not in inputs:
                continue
            value = inputs[key].clone()
            shift = translation.to(value).reshape(value.shape[0], *([1] * (value.ndim - 2)), 3)
            for joint_id in self.free_joint_indices:
                q_start = int(self.joint_q_start[joint_id])
                value[..., q_start : q_start + 3] += shift
            result[key] = value
        for key in ("root_body_q", "primary_body_q", "env_origin"):
            if key not in inputs:
                continue
            value = inputs[key].clone()
            value[..., :3] += translation.to(value).reshape(value.shape[0], *([1] * (value.ndim - 2)), 3)
            result[key] = value
        for key in ("contact_points_0", "contact_points_1"):
            if key not in inputs:
                continue
            points = inputs[key].unflatten(-1, (-1, 3))
            shifted = points + translation.to(points).reshape(points.shape[0], *([1] * (points.ndim - 2)), 3)
            if "contact_masks" in inputs:
                shifted = torch.where(inputs["contact_masks"].unsqueeze(-1), shifted, points)
            result[key] = shifted.flatten(-2)
        if "contact_tokens" in inputs:
            tokens = inputs["contact_tokens"].clone()
            shift = translation.to(tokens).reshape(tokens.shape[0], *([1] * (tokens.ndim - 2)), 3)
            tokens[..., 4:7] = torch.where(tokens[..., :1] > 0.5, tokens[..., 4:7] + shift, tokens[..., 4:7])
            result["contact_tokens"] = tokens
        return result

    def _before_model_forward(self):
        pass

    def _forward_neural_model(self, model_inputs):
        """Predict state and update any enabled diagnostic auxiliary caches."""
        neural_model = self._require_neural_model()
        finger_forces = getattr(self, "predicted_finger_object_normal_forces_b", None)
        robot_forces = getattr(self, "predicted_robot_net_normal_forces_w", None)
        if self.predicted_body_net_contact_forces_b is None and finger_forces is None and robot_forces is None:
            return neural_model.forward(model_inputs, single_step=True)

        forward_with_aux = getattr(neural_model, "forward_with_aux", None)
        if not callable(forward_with_aux):
            raise TypeError("Auxiliary force prediction requires a callable forward_with_aux() on the neural model.")
        model_output = cast("ModelMixedInputOutput", forward_with_aux(model_inputs, single_step=True))
        self._publish_auxiliary_predictions(model_output)
        return model_output.state_prediction

    def _publish_auxiliary_predictions(self, model_output: "ModelMixedInputOutput") -> None:
        """Copy the newest-step auxiliary force predictions into the solver's online caches."""
        finger_forces = getattr(self, "predicted_finger_object_normal_forces_b", None)
        robot_forces = getattr(self, "predicted_robot_net_normal_forces_w", None)
        if self.predicted_body_net_contact_forces_b is not None:
            predicted_forces = model_output.auxiliary_predictions["body_net_contact_forces_b"]
            self.predicted_body_net_contact_forces_b.copy_(predicted_forces[:, -1])
        if finger_forces is not None:
            prediction = model_output.auxiliary_predictions["finger_object_normal_forces_b"][:, -1]
            if prediction.shape != finger_forces.shape or not torch.isfinite(prediction).all():
                raise ValueError("Filtered finger-force prediction must be finite [B, 2, 3].")
            finger_forces.copy_(prediction)
        if robot_forces is not None:
            prediction = model_output.auxiliary_predictions["robot_net_normal_forces_w"][:, -1]
            if prediction.shape != robot_forces.shape or not torch.isfinite(prediction).all():
                raise ValueError("Robot normal-force prediction must be finite [B, 7, 3].")
            robot_forces.copy_(prediction)

    def _inference_precision_context(self):
        """Return the reduced-precision context for online model inference (no-op for FP32 or CPU)."""
        precision = getattr(self, "inference_precision", "fp32")
        if precision == "fp32" or not str(self.torch_device).startswith("cuda"):
            return nullcontext()
        if precision == "tf32":
            return _tf32_matmul()
        return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)

    def _model_forward(self, model_inputs):
        """Run the neural-model inference path for the current solver."""
        return self._forward_neural_model(model_inputs)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.01,
    ):
        from isaaclab_neural.utils import step_profile

        assert self.neural_model is not None, (
            "Cannot simulate via neural integrator as a neural model has not been setup yet."
        )
        if control is None:
            raise RuntimeError("Newton control is not initialized; cannot run the NeRD solver.")
        with step_profile.section("solver_update_states"):
            self._update_states(state_in, contacts, control.joint_f, joint_target_q=control.joint_target_q)

        # Keep torch inference on Warp's CUDA stream so state reads, model
        # execution, and state writes preserve ordering without extra syncs.
        # NOTE[Jie]: need to remove no_grad if we want the differentiability
        if str(self.torch_device).startswith("cuda"):
            stream_context = torch.cuda.stream(wp.stream_to_torch(self.device))
        else:
            stream_context = nullcontext()

        with torch.no_grad(), stream_context:
            with step_profile.section("model_input_prepare"):
                self._before_model_forward()
                model_inputs = self.get_neural_model_inputs()
            # compute the prediction using neural model, shape (num_envs, 1, dim)
            with step_profile.section("model_forward"), self._inference_precision_context():
                prediction = self._model_forward(model_inputs).float()

            with step_profile.section("prediction_postprocess"):
                cur_states = model_inputs["states"][:, -1, :]
                next_states = self.convert_prediction_to_next_states(cur_states, prediction.squeeze(1), dt)

                next_states_world = self.convert_states_back_to_world(model_inputs["root_body_q"], next_states)

                self.wrap2PI(next_states_world)

            with step_profile.section("state_writeback"):
                self._assign_states_from_torch(state_out, next_states_world)

        # update maximal coordinates
        with step_profile.section("eval_fk"):
            newton_utils.eval_fk(self.model, state_out)

    """
    Update the states, joint_f, and contacts in neural solver from a Newton state.
    """

    def _update_states(self, newton_states: State, contacts: Contacts | None, joint_f, *, joint_target_q=None):
        from isaaclab_neural.utils import step_profile

        if self.contact_adapter is not None and contacts is None:
            raise RuntimeError("Newton contacts are not initialized; cannot update NeRD contact inputs.")
        body_q = newton_states.body_q
        body_qd = newton_states.body_qd
        if body_q is None or body_qd is None:
            raise ValueError("NeuralSolver requires Newton body pose and velocity buffers.")
        self._acquire_states_to_torch(newton_states, self.states)
        self.wrap2PI(self.states)
        self.root_body_q = wp.to_torch(body_q).index_select(0, self.root_body_ids)
        self.root_body_qd = wp.to_torch(body_qd).index_select(0, self.root_body_ids)
        if self.joint_f_dim > 0:
            self.joint_f = wp.to_torch(joint_f).view(self.num_envs, self.joint_f_dim)
        if getattr(self, "joint_target_input", False):
            if joint_target_q is None:
                raise RuntimeError("joint_target_input requires Newton control.joint_target_q.")
            self.joint_target_q = wp.to_torch(joint_target_q).view(self.num_envs, self.dof_q_per_env)
        reuse_contact_inputs = getattr(self, "_reuse_contact_inputs", False)
        self._reuse_contact_inputs = False
        if self.contact_adapter is not None and not reuse_contact_inputs:
            assert contacts is not None
            with step_profile.section("adapter_update"):
                self.contact_adapter.update(contacts, newton_states)
            with step_profile.section("to_neural_inputs"):
                self.contact_inputs = self.contact_adapter.to_neural_inputs()
            self.primary_body_q = getattr(self.contact_adapter, "primary_body_q", None)

    def reuse_contact_inputs_once(self) -> None:
        """Keep the current contact inputs for the next state update instead of re-packing contacts.

        The caller guarantees that the contacts and body state are unchanged since the last adapter update.
        """
        self._reuse_contact_inputs = True

    def get_contact_masks(
        self,
        contact_depths,  # (num_envs, (T), num_contacts_per_env)
        contact_thickness0,  # (num_envs, (T), num_contacts_per_env)
        contact_thickness1,  # (num_envs, (T), num_contacts_per_env)
    ):
        del contact_depths, contact_thickness0, contact_thickness1
        warnings.warn(
            "NeuralSolver.get_contact_masks() is deprecated with fixed_ground contact processing.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise ValueError(
            "Newton-native inputs require explicit contact_masks. Masks cannot be reconstructed "
            "from signed surface separation because inactive padding and touching contacts can both be zero."
        )

    def _contact_token_transform(self):
        """Return the function mapping world-frame contact tokens into the model frame (see
        :func:`contact_tokens_to_model_frame`); solvers may substitute a compiled equivalent."""
        return contact_tokens_to_model_frame

    def process_neural_model_inputs(self, model_inputs):
        # Joint targets enter the model relative to the current joint coordinates (joint space, frame-independent).
        if "joint_target_q" in model_inputs:
            model_inputs["joint_target_offset"] = (
                model_inputs.pop("joint_target_q") - model_inputs["states"][..., : self.dof_q_per_env]
            )

        # convert frame
        (
            model_inputs["states"],
            model_inputs["next_states"],
            contact_points_0,
            contact_points_1,
            contact_normals,
            gravity_dir,
        ) = self.convert_coordinate_frame(
            model_inputs["root_body_q"],
            model_inputs["states"],
            model_inputs.get("next_states", None),
            model_inputs.get("contact_points_0", None),
            model_inputs.get("contact_points_1", None),
            model_inputs.get("contact_normals", None),
            model_inputs.get("gravity_dir", None),
        )
        # Token mode has no flat contact fields; do not inject None placeholders that
        # would later be mistaken for newton_native masked-contact RMS inputs.
        for key, value in (
            ("contact_points_0", contact_points_0),
            ("contact_points_1", contact_points_1),
            ("contact_normals", contact_normals),
            ("gravity_dir", gravity_dir),
        ):
            if value is not None:
                model_inputs[key] = value
            else:
                model_inputs.pop(key, None)

        if "contact_tokens" in model_inputs:
            translation_only = self.states_frame == "body_translation_only"
            if self.contact_token_frame == "owner":
                primary_body_q = model_inputs.get("primary_body_q")
                if primary_body_q is None:
                    raise ValueError(
                        "contact_token_frame='owner' requires primary_body_q poses "
                        "[B, T, num_primary_bodies, 7] recorded at collection time."
                    )
                model_inputs["contact_tokens"] = self._contact_token_transform()(
                    model_inputs["contact_tokens"], primary_body_q, translation_only, True
                )
            elif self.states_frame != "world":
                root_body_q = model_inputs["root_body_q"]
                batch, time = root_body_q.shape[:2]
                if self.anchor_frame_step == "first":
                    token_frame_q = root_body_q[:, 0:1].expand(batch, time, 7)
                elif self.anchor_frame_step == "last":
                    token_frame_q = root_body_q[:, -1:].expand(batch, time, 7)
                elif self.anchor_frame_step == "every":
                    token_frame_q = root_body_q
                else:
                    raise NotImplementedError(f"Unsupported anchor_frame_step: {self.anchor_frame_step!r}.")
                model_inputs["contact_tokens"] = self._contact_token_transform()(
                    model_inputs["contact_tokens"], token_frame_q, translation_only, False
                )
        model_inputs.pop("primary_body_q", None)

        # post processing
        self.wrap2PI(model_inputs["states"])
        if model_inputs["next_states"] is not None:
            self.wrap2PI(model_inputs["next_states"])

        if "states_embedding" in model_inputs:
            self.embed_states(model_inputs["states"], model_inputs["states_embedding"])
        else:
            model_inputs["states_embedding"] = self.embed_states(model_inputs["states"])

        # Apply contact mask: zero features for inactive anchors.
        if model_inputs.get("contact_points_1") is not None and "contact_masks" in model_inputs:
            mask = model_inputs["contact_masks"].unsqueeze(-1)  # (B, T, C, 1) bool
            for key in CONTACT_FLAT_VALUE_FIELDS:
                if key in model_inputs:
                    shape = model_inputs[key].shape
                    model_inputs[key] = torch.where(
                        mask,
                        model_inputs[key].view(shape[0], shape[1], self.num_contacts_per_env, -1),
                        0.0,
                    ).view(shape)

        return model_inputs

    """
    Prepare the inputs for the neural model inference.
    """

    def get_raw_neural_model_inputs(self):
        # assemble the model inputs in world frame without preprocessing. Dataset
        # generation writes these raw tensors so training can apply preprocessing
        # exactly once, matching online inference.
        model_inputs = {
            "root_body_q": self.root_body_q,
            "root_body_qd": self.root_body_qd,
            "states": self.states,
            "joint_f": self.joint_f,
            "gravity_dir": self.gravity_dir,
            **self.contact_inputs,
        }
        if self.primary_body_q is not None:
            model_inputs["primary_body_q"] = self.primary_body_q
        if getattr(self, "joint_target_input", False):
            model_inputs["joint_target_q"] = self.joint_target_q
        for k in model_inputs.keys():
            model_inputs[k] = model_inputs[k].unsqueeze(1)  # (num_envs, T, dim)

        return model_inputs

    def get_neural_model_inputs(self):
        # assemble the model inputs in world frame
        model_inputs = {
            "root_body_q": self.root_body_q,
            "states": self.states,
            "states_embedding": self.states_embedding,
            "joint_f": self.joint_f,
            "gravity_dir": self.gravity_dir,
            **self.contact_inputs,
        }
        if self.contact_token_frame == "owner" and self.primary_body_q is not None:
            model_inputs["primary_body_q"] = self.primary_body_q
        if getattr(self, "joint_target_input", False):
            model_inputs["joint_target_q"] = self.joint_target_q
        for k in model_inputs.keys():
            model_inputs[k] = model_inputs[k].unsqueeze(1)  # (num_envs, T, dim)

        processed_model_inputs = self.process_neural_model_inputs(model_inputs)

        return processed_model_inputs

    """
    Fix continuous angular dofs in the states vector (in-place operation).
    """

    def wrap2PI(self, states, is_continuous_dof=None):
        if is_continuous_dof is None:
            is_continuous_dof = self.is_continuous_dof
        if not is_continuous_dof.any():
            return
        assert states.shape[-1] == is_continuous_dof.shape[0]
        wrap_delta = torch.floor((states[..., is_continuous_dof] + np.pi) / (2 * np.pi)) * (2 * np.pi)
        states[..., is_continuous_dof] -= wrap_delta

    def wrap2PI_differentiable(self, states, is_continuous_dof=None):
        """
        Differentiable angle wrapping using atan2.
        Returns a NEW tensor (out-of-place operation for proper gradient flow).

        For continuous DOFs (joints that rotate continuously), wrap angles to [-pi, pi]
        using differentiable operations.

        Args:
            states: Tensor of shape (..., state_dim) containing positions and velocities
            is_continuous_dof: Boolean mask for continuous DOFs, defaults to self.is_continuous_dof

        Returns:
            New tensor with wrapped angles (gradients flow correctly)
        """
        if is_continuous_dof is None:
            is_continuous_dof = self.is_continuous_dof

        # Early return if no continuous DOFs
        if not is_continuous_dof.any():
            return states.clone()

        assert states.shape[-1] == is_continuous_dof.shape[0]

        # Create a copy of the input tensor
        wrapped_states = states.clone()

        # Extract continuous DOF positions
        continuous_positions = states[..., is_continuous_dof]

        # Differentiable wrapping: atan2(sin(x), cos(x))
        wrapped_angles = torch.atan2(torch.sin(continuous_positions), torch.cos(continuous_positions))

        # Assign to the copy (safe out-of-place operation)
        wrapped_states[..., is_continuous_dof] = wrapped_angles

        return wrapped_states  # Return new tensor

    def _acquire_states_to_torch(self, newton_states: State, torch_states: torch.Tensor):
        wp.launch(
            warp_utils.acquire_states,
            dim=self.num_envs,
            inputs=[
                newton_states.joint_q,
                newton_states.joint_qd,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[wp.from_torch(torch_states)],
            device=self.device,
        )

    def _assign_states_from_torch(self, newton_state: State, torch_states: torch.Tensor):
        wp.launch(
            warp_utils.assign_states,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(torch_states),
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[newton_state.joint_q, newton_state.joint_qd],
            device=self.device,
        )

    """
    Converts the prediction tensor to the next states tensor according to prediction_type.

    Args:
        states (torch.Tensor): The current states tensor (num_envs, state_dim).
        prediction (torch.Tensor): The prediction tensor (num_envs, pred_dim).

    Returns:
        torch.Tensor: The next states tensor (num_envs, state_dim).

    Raises:
        NotImplementedError: If the prediction type is not supported.
    """

    def convert_prediction_to_next_states(self, states, prediction, dt=None):
        next_states = torch.empty_like(states)

        if self.prediction_type in ["absolute", "relative"]:
            """ full state prediction: absolute or relative """
            prediction_dof_offset = 0
            # Consecutive regular joints with contiguous coordinates are converted as one elementwise run
            # (identical per-element arithmetic, far fewer kernel launches for many scalar joints).
            run_start = run_end = None

            def flush_run():
                nonlocal prediction_dof_offset, run_start, run_end
                if run_start is not None:
                    prediction_dof_offset += self._convert_prediction_to_next_states_regular_dofs(
                        states[..., run_start:run_end],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., run_start:run_end],
                    )
                    run_start = run_end = None

            # Compute position components of the next states for each joint individually
            for joint_id in range(self.num_joints_per_env):
                joint_dof_start = self.joint_q_start[joint_id]
                joint_type = self.joint_types[joint_id]
                if joint_type != JointType.FREE and joint_type != JointType.BALL:
                    if run_start is not None and joint_dof_start != run_end:
                        flush_run()
                    if run_start is None:
                        run_start = joint_dof_start
                    run_end = self.joint_q_end[joint_id]
                    continue
                flush_run()
                if joint_type == JointType.FREE:
                    # position dofs
                    prediction_dof_offset += self._convert_prediction_to_next_states_regular_dofs(
                        states[..., joint_dof_start : joint_dof_start + 3],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start : joint_dof_start + 3],
                    )
                    # 3d orientation dofs
                    prediction_dof_offset += self._convert_prediction_to_next_states_orientation_dofs(
                        states[..., joint_dof_start + 3 : joint_dof_start + 7],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start + 3 : joint_dof_start + 7],
                    )
                else:
                    prediction_dof_offset += self._convert_prediction_to_next_states_orientation_dofs(
                        states[..., joint_dof_start : joint_dof_start + 4],
                        prediction[..., prediction_dof_offset:],
                        next_states[..., joint_dof_start : joint_dof_start + 4],
                    )
            flush_run()

            # Compute velocity components of the next states
            if self.prediction_type == "absolute":
                next_states[..., self.dof_q_per_env :].copy_(prediction[..., prediction_dof_offset:])
            elif self.prediction_type == "relative":
                next_states[..., self.dof_q_per_env :] = (
                    states[..., self.dof_q_per_env :] + prediction[..., prediction_dof_offset:]
                )
            else:
                raise NotImplementedError
        elif self.prediction_type == "acceleration":
            # NOTE: Acceleration prediction is partially implemented for target statistics
            # in convert_next_states_to_prediction(), but converting model outputs back
            # into next states still needs completion before this mode can be used.
            raise NotImplementedError
        else:
            raise NotImplementedError

        return next_states

    """
    Converts the prediction to the next states for regular degrees-of-freedom.

    Params:
        states: The current states.
        prediction: The prediction.
        next_states: The next states.

    Return:
        The number of corresponding degrees-of-freedom of prediction.

    Assume the dofs in the states and next_states are all regular dofs.
    Assume prediction is index from zero, prediction.shape[-1] might be longer
        than states.shape[-1], but only the first prediction_dims will be used.
    """

    def _convert_prediction_to_next_states_regular_dofs(self, states, prediction, next_states):
        assert states.shape[-1] == next_states.shape[-1]
        dofs = states.shape[-1]
        if self.prediction_type == "absolute":
            next_states.copy_(prediction[..., :dofs])
            return dofs
        elif self.prediction_type == "relative":
            next_states.copy_(states + prediction[..., :dofs])
            return dofs
        else:
            raise NotImplementedError

    """
    Converts the prediction to the next states for orientation degrees-of-freedom.

    Params:
        states: The current states.
        prediction: The prediction.
        next_states: The next states.

    Return:
        The number of corresponding degrees-of-freedom of prediction.

    Assume the dofs in the states and next_states are all regular dofs.
    Assume prediction is index from zero, prediction.shape[-1] might be longer
        than states.shape[-1], but only the first prediction_dim will be used.
    """

    def _convert_prediction_to_next_states_orientation_dofs(self, states, prediction, next_states):
        assert states.shape[-1] == 4 and next_states.shape[-1] == 4

        # Parse the prediction into quaternion
        prediction_dofs = None
        if (
            self.orientation_prediction_parameterization == "naive"
            or self.orientation_prediction_parameterization == "quaternion"
        ):
            predicted_quaternion = prediction[..., :4]
            prediction_dofs = 4
        elif self.orientation_prediction_parameterization == "exponential":
            predicted_quaternion = torch_utils.exponential_coord_to_quat(prediction[..., :3])
            prediction_dofs = 3
        else:
            raise NotImplementedError

        # Apply quaternion/delta quaternion to the states to acquire next_states
        if self.prediction_type == "absolute":
            raw_next_quaternion = predicted_quaternion
        elif self.prediction_type == "relative":
            if self.orientation_prediction_parameterization == "naive":
                raw_next_quaternion = states + predicted_quaternion
            else:
                # raw_next_quaternion = torch_utils.quat_mul(states, predicted_quaternion)
                raw_next_quaternion = torch_utils.quat_mul(predicted_quaternion, states)
        else:
            raise NotImplementedError

        # Normalize the next_states quaternion
        next_states.copy_(torch_utils.normalize(raw_next_quaternion))

        return prediction_dofs

    def compute_acceleration_from_pos(self, states, next_states, dt):
        # WARNING: This helper still assumes the old spatial-twist free-root layout
        # ([omega, nu]) even though current Newton states use [lin_vel, ang_vel].
        # Keep acceleration prediction disabled until this is updated.
        acceleration = torch.empty(
            (*states.shape[:-1], self.dof_qd_per_env), dtype=states.dtype, device=self.torch_device
        )

        vel = states[..., self.dof_q_per_env :]

        acc_dof_offset = 0
        for joint_id in range(self.num_joints_per_env):
            joint_dof_start = self.joint_q_start[joint_id]
            if self.joint_types[joint_id] == JointType.FREE:
                p0 = states[..., joint_dof_start : joint_dof_start + 3]
                q0 = states[..., joint_dof_start + 3 : joint_dof_start + 7]
                p1 = next_states[..., joint_dof_start : joint_dof_start + 3]
                q1 = next_states[..., joint_dof_start + 3 : joint_dof_start + 7]
                omega_0 = vel[..., acc_dof_offset : acc_dof_offset + 3]
                nu_0 = vel[..., acc_dof_offset + 3 : acc_dof_offset + 6]

                delta_q = torch_utils.delta_quat(q0, q1)
                omega_1 = torch_utils.quat_to_exponential_coord(delta_q) / dt
                nu_1 = (p1 - p0 - torch.cross(omega_1, p0, dim=-1) * dt) / dt
                acceleration[..., acc_dof_offset : acc_dof_offset + 3] = (omega_1 - omega_0) / dt
                acceleration[..., acc_dof_offset + 3 : acc_dof_offset + 6] = (nu_1 - nu_0) / dt
                acc_dof_offset += 6
            elif self.joint_types[joint_id] == JointType.BALL:
                raise NotImplementedError
            else:
                joint_dof_end = self.joint_q_end[joint_id]
                joint_dofs = joint_dof_end - joint_dof_start
                delta_states = (
                    next_states[..., joint_dof_start:joint_dof_end] - states[..., joint_dof_start:joint_dof_end]
                )
                self.wrap2PI(delta_states, self.is_continuous_dof[joint_dof_start:joint_dof_end])
                acceleration[..., acc_dof_offset : acc_dof_offset + joint_dofs] = (
                    delta_states - vel[..., acc_dof_offset : acc_dof_offset + joint_dofs] * dt
                ) / (dt * dt)
                acc_dof_offset += joint_dofs

        return acceleration

    def convert_next_states_to_prediction(
        self,
        states,  # (B, (T), dof_states)
        next_states,  # (B, (T), dof_states)
        dt=None,
        prediction_type=None,
    ):
        prediction = torch.empty(
            (*states.shape[:-1], self.prediction_dim), dtype=states.dtype, device=self.torch_device
        )

        if prediction_type is None:
            prediction_type = self.prediction_type

        if prediction_type in ["absolute", "relative"]:
            prediction_dof_offset = 0

            # Compute position components of the prediction for each joint individually
            for joint_id in range(self.num_joints_per_env):
                joint_dof_start = self.joint_q_start[joint_id]
                if self.joint_types[joint_id] == JointType.FREE:
                    prediction_dof_offset += self._convert_next_states_to_prediction_regular_dofs(
                        states[..., joint_dof_start : joint_dof_start + 3],
                        next_states[..., joint_dof_start : joint_dof_start + 3],
                        self.is_continuous_dof[joint_dof_start : joint_dof_start + 3],
                        prediction[..., prediction_dof_offset:],
                        prediction_type,
                    )
                    prediction_dof_offset += self._convert_next_states_to_prediction_orientation_dofs(
                        states[..., joint_dof_start + 3 : joint_dof_start + 7],
                        next_states[..., joint_dof_start + 3 : joint_dof_start + 7],
                        prediction[..., prediction_dof_offset:],
                        prediction_type,
                    )
                elif self.joint_types[joint_id] == JointType.BALL:
                    prediction_dof_offset += self._convert_next_states_to_prediction_orientation_dofs(
                        states[..., joint_dof_start : joint_dof_start + 4],
                        next_states[..., joint_dof_start : joint_dof_start + 4],
                        prediction[..., prediction_dof_offset:],
                        prediction_type,
                    )
                else:
                    joint_dof_end = self.joint_q_end[joint_id]
                    prediction_dof_offset += self._convert_next_states_to_prediction_regular_dofs(
                        states[..., joint_dof_start:joint_dof_end],
                        next_states[..., joint_dof_start:joint_dof_end],
                        self.is_continuous_dof[joint_dof_start:joint_dof_end],
                        prediction[..., prediction_dof_offset:],
                        prediction_type,
                    )

            # Compute velocity components of the prediction
            if prediction_type == "absolute":
                prediction[..., prediction_dof_offset:].copy_(next_states[..., self.dof_q_per_env :])
            elif prediction_type == "relative":
                prediction[..., prediction_dof_offset:] = (
                    next_states[..., self.dof_q_per_env :] - states[..., self.dof_q_per_env :]
                )
            else:
                raise NotImplementedError
        elif prediction_type == "acceleration":
            # NOTE: For acceleration prediction, we only use velocity to compute the acceleration
            # The converted acceleration is only used for computing the target mean/std,
            # not in loss computation.
            prediction.copy_((next_states[..., self.dof_q_per_env :] - states[..., self.dof_q_per_env :]) / dt)
        else:
            raise NotImplementedError

        return prediction

    """
    Converts the next states to the prediction for regular degrees-of-freedom.

    Params:
        states: The current states.
        next_states: The next states.
        is_continuous_dof: Indicates whether the degrees-of-freedom are continuous.
        prediction: The prediction.

    Return:
        The number of degrees-of-freedom of converted prediction.

    Assume the dofs in the states and next_states are all regular dofs.
    Assume is_continuous_dof is index from zero and has the same length as states.shape[-1].
    Assume prediction is index from zero, prediction.shape[-1] might be longer
        than states.shape[-1], but only the first prediction_dim will be used.
    The converted prediction is saved in predction[0:prediction_dim]
    """

    def _convert_next_states_to_prediction_regular_dofs(
        self, states, next_states, is_continuous_dof, prediction, prediction_type=None
    ):
        if prediction_type is None:
            prediction_type = self.prediction_type

        assert states.shape[-1] == next_states.shape[-1]
        dofs = states.shape[-1]

        if prediction_type == "absolute":
            prediction[..., :dofs].copy_(next_states)
            return dofs
        elif prediction_type == "relative":
            prediction[..., :dofs].copy_(next_states - states)
            self.wrap2PI(prediction[..., :dofs], is_continuous_dof)
            return dofs
        else:
            raise NotImplementedError

    """
    Converts the next states to the prediction for orientation degrees-of-freedom.

    Params:
        states: The current states of shape (..., 4).
        next_states: The next states of shape (..., 4).
        prediction: The prediction.

    Return:
        The number of degrees-of-freedom of converted prediction.

    Assume the dofs in the states and next_states are in correct sizes.
    Assume prediction is index from zero.
    The converted prediction is saved in predction[0:prediction_dim]
    """

    def _convert_next_states_to_prediction_orientation_dofs(
        self, states, next_states, prediction, prediction_type=None
    ):
        if prediction_type is None:
            prediction_type = self.prediction_type

        assert states.shape[-1] == 4 and next_states.shape[-1] == 4

        if prediction_type == "absolute":
            target_quaternion = next_states
        elif prediction_type == "relative":
            if self.orientation_prediction_parameterization == "naive":
                target_quaternion = next_states - states
            else:
                target_quaternion = torch_utils.delta_quat(states, next_states, frame="world")

        if (
            self.orientation_prediction_parameterization == "naive"
            or self.orientation_prediction_parameterization == "quaternion"
        ):
            prediction[..., :4].copy_(target_quaternion)
            return 4
        elif self.orientation_prediction_parameterization == "exponential":
            prediction[..., :3].copy_(torch_utils.quat_to_exponential_coord(target_quaternion))
            return 3
        else:
            raise NotImplementedError

    def _convert_contacts_w2b(
        self,
        root_body_q,  # (B, T, num_contacts, 7)
        contact_points_0,  # (B, T, num_contacts * 3)
        contact_points_1,  # (B, T, num_contacts * 3)
        contact_normals,  # (B, T, num_contacts * 3)
        translation_only,
    ):
        shape = contact_points_1.shape
        root_body_q = root_body_q.reshape(-1, 7)
        if contact_points_0 is not None:
            contact_points_0 = contact_points_0.reshape(-1, 3)
        contact_points_1 = contact_points_1.reshape(-1, 3)
        contact_normals = contact_normals.reshape(-1, 3)

        body_frame_pos = root_body_q[:, :3]
        if translation_only:
            body_frame_quat = torch.zeros_like(root_body_q[:, 3:7])
            body_frame_quat[:, 3] = 1.0
        else:
            body_frame_quat = root_body_q[:, 3:7]

        if contact_points_0 is not None:
            assert contact_points_0.shape[0] == root_body_q.shape[0]
            contact_points_0_body = torch_utils.transform_point_inverse(
                body_frame_pos, body_frame_quat, contact_points_0
            ).view(*shape)
        else:
            contact_points_0_body = None

        assert contact_points_1.shape[0] == root_body_q.shape[0]
        contact_points_1_body = torch_utils.transform_point_inverse(
            body_frame_pos, body_frame_quat, contact_points_1
        ).view(*shape)

        assert contact_normals.shape[0] == root_body_q.shape[0]
        if translation_only:
            contact_normals_body = contact_normals.view(*shape)
        else:
            contact_normals_body = torch_utils.quat_rotate_inverse(body_frame_quat, contact_normals).view(*shape)

        return contact_points_0_body, contact_points_1_body, contact_normals_body

    """
    Convert the states from world frame to body frame defined by root_body_q.
    Only convert the root joint states if applicable.
    """

    def _convert_states_w2b(
        self,
        root_body_q,  # (B, T, 7)
        states,  # (B, T, dof_states)
        translation_only,
    ):
        shape = states.shape
        root_body_q = root_body_q.reshape(-1, 7)
        states = states.reshape(-1, self.state_dim)

        body_frame_pos = root_body_q[:, :3]
        if translation_only:
            body_frame_quat = torch.zeros_like(root_body_q[:, 3:7])
            body_frame_quat[:, 3] = 1.0
        else:
            body_frame_quat = root_body_q[:, 3:7]

        assert states.shape[0] == root_body_q.shape[0]

        states_body = states.clone()
        for joint_id in self.free_joint_indices:
            q_start = int(self.joint_q_start[joint_id])
            v_start = self.dof_q_per_env + int(self.joint_qd_start[joint_id])
            # velocity representation is [lin_vel, ang_vel]
            (pos_body, quat_body, lin_vel_body, ang_vel_body) = torch_utils.convert_states_w2b(
                body_frame_pos,
                body_frame_quat,
                p=states[:, q_start : q_start + 3],
                quat=states[:, q_start + 3 : q_start + 7],
                lin_vel=states[:, v_start : v_start + 3],
                ang_vel=states[:, v_start + 3 : v_start + 6],
            )
            states_body[:, q_start : q_start + 7] = torch.cat((pos_body, torch_utils.normalize(quat_body)), dim=-1)
            states_body[:, v_start : v_start + 6] = torch.cat((lin_vel_body, ang_vel_body), dim=-1)

        return states_body.view(*shape)

    def _convert_gravity_w2b(
        self,
        root_body_q,  # (B, T, 7)
        gravity_dir,  # (B, T, 3)
        translation_only,
    ):
        if translation_only:
            return gravity_dir

        shape = gravity_dir.shape
        root_body_q = root_body_q.reshape(-1, 7)
        gravity_dir = gravity_dir.reshape(-1, 3)

        body_frame_quat = root_body_q[:, 3:7]

        assert gravity_dir.shape[0] == body_frame_quat.shape[0]
        gravity_dir_body = torch_utils.quat_rotate_inverse(body_frame_quat, gravity_dir).view(*shape)

        return gravity_dir_body

    def convert_coordinate_frame(
        self,
        root_body_q,  # (B, T, 7)
        states,  # (B, T, dof_states)
        next_states,  # (B, T, dof_states), can be None
        contact_points_0,  # (B, T, num_contacts * 3)
        contact_points_1,  # (B, T, num_contacts * 3)
        contact_normals,  # (B, T, num_contacts * 3)
        gravity_dir,  # (B, T, 3)
    ):
        assert len(states.shape) == 3

        if self.states_frame == "world":
            return states, next_states, contact_points_0, contact_points_1, contact_normals, gravity_dir
        elif self.states_frame == "body" or self.states_frame == "body_translation_only":
            B, T = states.shape[0], states.shape[1]

            if self.anchor_frame_step == "first":
                anchor_frame_body_q = root_body_q[:, 0:1, :].expand(B, T, 7)
            elif self.anchor_frame_step == "last":
                anchor_frame_body_q = root_body_q[:, -1:, :].expand(B, T, 7)
            elif self.anchor_frame_step == "every":
                anchor_frame_body_q = root_body_q
            else:
                raise NotImplementedError

            # convert contacts
            if contact_points_1 is not None:
                contact_points_0_body, contact_points_1_body, contact_normals_body = self._convert_contacts_w2b(
                    anchor_frame_body_q.view(B, T, 1, 7).expand(B, T, self.num_contacts_per_env, 7),
                    contact_points_0,
                    contact_points_1,
                    contact_normals,
                    translation_only=(self.states_frame == "body_translation_only"),
                )
            else:
                contact_points_0_body = None
                contact_points_1_body = None
                contact_normals_body = None

            # convert states
            states_body = self._convert_states_w2b(
                anchor_frame_body_q, states, translation_only=(self.states_frame == "body_translation_only")
            )
            if next_states is not None:
                next_states_body = self._convert_states_w2b(
                    anchor_frame_body_q, next_states, translation_only=(self.states_frame == "body_translation_only")
                )
            else:
                next_states_body = None

            # convert gravity
            if gravity_dir is not None:
                gravity_dir_body = self._convert_gravity_w2b(
                    anchor_frame_body_q, gravity_dir, translation_only=(self.states_frame == "body_translation_only")
                )
            else:
                gravity_dir_body = None

            return (
                states_body,
                next_states_body,
                contact_points_0_body,
                contact_points_1_body,
                contact_normals_body,
                gravity_dir_body,
            )
        else:
            raise NotImplementedError

    def convert_states_back_to_world(
        self,
        root_body_q,  # (B, T, 7)
        states,  # (B, dof_states)
    ):
        if self.states_frame == "world":
            return states.clone()
        elif self.states_frame == "body" or self.states_frame == "body_translation_only":
            if self.anchor_frame_step == "first":
                anchor_step = 0
            elif self.anchor_frame_step == "last" or self.anchor_frame_step == "every":
                anchor_step = -1
            else:
                raise NotImplementedError

            shape = states.shape

            anchor_frame_q = root_body_q[:, anchor_step, :]

            anchor_frame_pos = anchor_frame_q[:, :3]
            if self.states_frame == "body":
                anchor_frame_quat = anchor_frame_q[:, 3:7]
            elif self.states_frame == "body_translation_only":
                anchor_frame_quat = torch.zeros_like(anchor_frame_q[:, 3:7])
                anchor_frame_quat[:, 3] = 1.0

            assert states.shape[0] == anchor_frame_q.shape[0]

            states_world = states.clone()
            # only need to convert the states of the FREE base joint
            for joint_id in self.free_joint_indices:
                q_start = int(self.joint_q_start[joint_id])
                v_start = self.dof_q_per_env + int(self.joint_qd_start[joint_id])
                # velocity representation is [lin_vel, ang_vel]
                (pos_world, quat_world, lin_vel_world, ang_vel_world) = torch_utils.convert_states_b2w(
                    anchor_frame_pos,
                    anchor_frame_quat,
                    p=states[:, q_start : q_start + 3],
                    quat=states[:, q_start + 3 : q_start + 7],
                    lin_vel=states[:, v_start : v_start + 3],
                    ang_vel=states[:, v_start + 3 : v_start + 6],
                )
                states_world[:, q_start : q_start + 7] = torch.cat(
                    (pos_world, torch_utils.normalize(quat_world)), dim=-1
                )
                states_world[:, v_start : v_start + 6] = torch.cat((lin_vel_world, ang_vel_world), dim=-1)

            return states_world.view(*shape)
        raise NotImplementedError(f"Unsupported states frame: {self.states_frame}")

    """
    Embeds the given states into a new representation based on states_embedding_type.

    Args:
        states (torch.Tensor): The input states to be embedded.
        states_embedding (torch.Tensor, optional): The tensor to store the embedded states.
            If None, a new tensor will be created.

    Returns:
        torch.Tensor: The embedded states.

    Raises:
        NotImplementedError: If the states_embedding_type is not supported.
    """

    def embed_states(self, states, states_embedding=None):
        if self.states_embedding_type is None or self.states_embedding_type == "identical":
            if states_embedding is not None:
                states_embedding.copy_(states)
                return states_embedding
            return states.clone()
        elif self.states_embedding_type == "sinusoidal":
            if states_embedding is None:
                states_embedding = torch.zeros((*states.shape[:-1], self.state_embedding_dim), device=states.device)
            idx = 0
            for dof_idx in range(len(self.is_angular_dof)):
                if not self.is_angular_dof[dof_idx]:
                    states_embedding[..., idx] = states[..., dof_idx].clone()
                    idx += 1
                else:
                    states_embedding[..., idx] = torch.sin(states[..., dof_idx])
                    states_embedding[..., idx + 1] = torch.cos(states[..., dof_idx])
                    idx += 2
            states_embedding[..., idx:] = states[..., self.dof_q_per_env :].clone()
            return states_embedding
        else:
            raise NotImplementedError
