import torch
import warp as wp
import newton

from typing import Literal

from isaaclab_neural.contacts.packing import ContactPackingPolicy, get_contact_order
from isaaclab_neural.utils import torch_utils


class NewtonContactAdapter:
    """
    Adapter from Newton's native dynamic contacts to NeRD's fixed-size contact input.

    Newton's collision pipeline emits a variable number of contacts each frame.
    NeRD models expect a fixed `(num_envs, num_contacts_per_env, ...)` layout,
    so this class packs the native contacts into fixed buffers with masks.
    """

    def __init__(
        self,
        model: newton.Model,
        num_contacts_per_env: int,
        device: str | None = None,
        packing_policy: ContactPackingPolicy = "stable_index",
    ):
        self.model = model
        self.num_envs = int(model.world_count)
        self.num_contacts_per_env = int(num_contacts_per_env)
        self.num_total_contacts = self.num_envs * self.num_contacts_per_env
        self.packing_policy = packing_policy

        if device is None:
            self.device = wp.device_to_torch(model.device)
        else:
            self.device = torch.device(device)

        self.shape_body = model.shape_body.numpy()
        self.bodies_per_env = int(model.body_count // model.world_count)
        self.body_world = (
            model.body_world.numpy() if hasattr(model, "body_world") else None
        )

        self.contact_masks = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.bool,
            device=self.device,
        )
        self.contact_normals = torch.zeros(
            (self.num_envs, self.num_contacts_per_env, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_depths = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_thicknesses_0 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_thicknesses_1 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_points_0 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_points_1 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env, 3),
            dtype=torch.float32,
            device=self.device,
        )

    def reset_buffers(self) -> None:
        """Clear all fixed-size contact buffers before packing a new frame."""
        self.contact_masks.zero_()
        self.contact_normals.zero_()
        self.contact_depths.zero_()
        self.contact_thicknesses_0.zero_()
        self.contact_thicknesses_1.zero_()
        self.contact_points_0.zero_()
        self.contact_points_1.zero_()

    def update(
        self,
        contacts: newton.Contacts,
        state: newton.State | None = None,
    ) -> None:
        """Pack Newton native contacts into fixed NeRD contact buffers."""
        self.reset_buffers()

        contact_count = self._contact_count(contacts)
        if contact_count == 0:
            return

        raw = self._read_raw_contacts(contacts, contact_count, state)
        order = get_contact_order(raw, self.packing_policy)
        write_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )

        for contact_idx in order.tolist():
            shape0 = int(raw["shape0"][contact_idx].item())
            shape1 = int(raw["shape1"][contact_idx].item())
            env_id = self._contact_env_id(shape0, shape1)
            if env_id < 0 or env_id >= self.num_envs:
                continue

            slot_id = int(write_counts[env_id].item())
            if slot_id >= self.num_contacts_per_env:
                continue

            self.contact_masks[env_id, slot_id] = True
            self.contact_normals[env_id, slot_id].copy_(raw["normal"][contact_idx])
            self.contact_depths[env_id, slot_id] = raw["depth"][contact_idx]
            self.contact_thicknesses_0[env_id, slot_id] = raw["thickness0"][contact_idx]
            self.contact_thicknesses_1[env_id, slot_id] = raw["thickness1"][contact_idx]
            self.contact_points_0[env_id, slot_id].copy_(raw["point0"][contact_idx])
            self.contact_points_1[env_id, slot_id].copy_(raw["point1_world"][contact_idx])

            write_counts[env_id] += 1

    def to_neural_inputs(self) -> dict[str, torch.Tensor]:
        """Return cloned tensors with the same shape convention as fixed-ground contacts."""
        B = self.num_envs
        C = self.num_contacts_per_env
        return {
            "contact_masks": self.contact_masks.clone(),
            "contact_normals": self.contact_normals.reshape(B, C * 3).clone(),
            "contact_depths": self.contact_depths.clone(),
            "contact_thicknesses_0": self.contact_thicknesses_0.clone(),
            "contact_thicknesses_1": self.contact_thicknesses_1.clone(),
            "contact_points_0": self.contact_points_0.reshape(B, C * 3).clone(),
            "contact_points_1": self.contact_points_1.reshape(B, C * 3).clone(),
        }

    def fingerprint(self) -> dict:
        """Return metadata that must match between data collection and runtime."""
        return {
            "contact_mode": "newton_native",
            "num_contacts_per_env": self.num_contacts_per_env,
            "packing_policy": self.packing_policy,
            "thickness_rule": "native_margin",
            "depth_rule": "support_distance_along_normal",
            "contact_points_1_frame": "world",
            "adapter_version": 3,
        }

    def _contact_count(self, contacts: newton.Contacts) -> int:
        if contacts.rigid_contact_count is None:
            return 0
        count = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
        return min(count, int(contacts.rigid_contact_max))

    def _read_raw_contacts(
        self,
        contacts: newton.Contacts,
        contact_count: int,
        state: newton.State | None,
    ) -> dict[str, torch.Tensor]:
        shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:contact_count].to(self.device)
        shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:contact_count].to(self.device)
        point0 = wp.to_torch(contacts.rigid_contact_point0)[:contact_count].to(self.device)
        point1 = wp.to_torch(contacts.rigid_contact_point1)[:contact_count].to(self.device)
        normal = wp.to_torch(contacts.rigid_contact_normal)[:contact_count].to(self.device)

        thickness0 = self._read_thickness(contacts, contact_count, side=0)
        thickness1 = self._read_thickness(contacts, contact_count, side=1)
        depth, point0_world, point1_world = self._read_depth_and_world_points(
            contact_count,
            shape0,
            shape1,
            point0,
            point1,
            normal,
            state,
        )

        return {
            "shape0": shape0,
            "shape1": shape1,
            "point0": point0,
            "point1": point1,
            "point0_world": point0_world,
            "point1_world": point1_world,
            "normal": normal,
            "depth": depth,
            "thickness0": thickness0,
            "thickness1": thickness1,
        }

    def _read_thickness(
        self,
        contacts: newton.Contacts,
        contact_count: int,
        side: Literal[0, 1],
    ) -> torch.Tensor:
        margin_name = f"rigid_contact_margin{side}"
        margins = getattr(contacts, margin_name)
        if margins is None:
            raise ValueError(
                f"NewtonContactAdapter requires {margin_name} for thickness. "
                "Please pass contacts generated by Newton's collision pipeline."
            )

        return wp.to_torch(margins)[:contact_count].to(self.device)

    def _read_depth_and_world_points(
        self,
        contact_count: int,
        shape0: torch.Tensor,
        shape1: torch.Tensor,
        point0: torch.Tensor,
        point1: torch.Tensor,
        normal: torch.Tensor,
        state: newton.State | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            raise ValueError(
                "NewtonContactAdapter computes depth from geometry. "
                "Pass the current Newton State to NewtonContactAdapter.update() "
                "so world-space depth can be computed from point0/point1/normal."
            )

        point0_world = self._points_to_world(shape0, point0, state)
        point1_world = self._points_to_world(shape1, point1, state)
        depth = torch.sum(normal * (point1_world - point0_world), dim=-1)
        return depth, point0_world, point1_world

    def _points_to_world(
        self,
        shapes: torch.Tensor,
        points: torch.Tensor,
        state: newton.State,
    ) -> torch.Tensor:
        body_q = wp.to_torch(state.body_q).to(self.device)
        points_world = points.clone()

        body_ids = torch.empty(shapes.shape[0], dtype=torch.long, device=self.device)
        for i, shape_id in enumerate(shapes.tolist()):
            if shape_id < 0 or shape_id >= len(self.shape_body):
                body_ids[i] = -1
            else:
                body_ids[i] = int(self.shape_body[int(shape_id)])

        dynamic_mask = body_ids >= 0
        if dynamic_mask.any():
            q = body_q[body_ids[dynamic_mask]]
            points_world[dynamic_mask] = torch_utils.transform_point(
                q[:, :3],
                q[:, 3:7],
                points[dynamic_mask],
            )

        return points_world

    def _contact_env_id(self, shape0: int, shape1: int) -> int:
        env_id = self._shape_to_env_id(shape0)
        if env_id >= 0:
            return env_id
        return self._shape_to_env_id(shape1)

    def _shape_to_env_id(self, shape_id: int) -> int:
        if shape_id < 0 or shape_id >= len(self.shape_body):
            return -1

        body_id = int(self.shape_body[shape_id])
        if body_id < 0:
            return -1

        if self.body_world is not None:
            return int(self.body_world[body_id])

        return body_id // self.bodies_per_env