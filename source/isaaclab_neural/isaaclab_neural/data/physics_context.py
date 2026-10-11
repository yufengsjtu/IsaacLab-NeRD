# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-world Newton material and inertial context for branch replay and isolated model fitting."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from newton import Model

WORLD_PHYSICS_FIELDS = (
    "body_mass",
    "body_inv_mass",
    "body_inertia",
    "body_inv_inertia",
    "body_com",
    "shape_material_mu",
    "shape_material_restitution",
    "shape_material_ke",
    "shape_material_kd",
)


class WorldPhysicsContext:
    """Bind equal-layout worlds without treating one randomized world as a universal physical protocol.

    Snapshots preserve mass [kg], inverse mass [1/kg], inertia [kg*m^2], inverse inertia [1/(kg*m^2)],
    COM [m], dimensionless friction/restitution, contact stiffness [N/m] and damping [N*s/m].
    Shared world-minus-one ground properties are never overwritten.
    """

    def __init__(self, model: Model, num_envs: int):
        self.model, self.num_envs = model, num_envs
        self.rows = {kind: self._world_rows(wp.to_torch(getattr(model, f"{kind}_world"))) for kind in ("body", "shape")}
        self.values = {name: wp.to_torch(getattr(model, name)) for name in WORLD_PHYSICS_FIELDS}
        self.device = self.rows["body"].device

    @classmethod
    def isolated(cls, model: Model, num_envs: int) -> WorldPhysicsContext:
        """Clone mutable physical arrays while sharing immutable topology and geometry."""
        isolated = copy.copy(model)
        for name in WORLD_PHYSICS_FIELDS:
            setattr(isolated, name, wp.clone(getattr(model, name)))
        return cls(isolated, num_envs)

    def capture(self, env_ids: torch.Tensor, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        """Copy selected worlds' realized SI parameters, without retaining tensor backing-storage views."""
        ids = env_ids.to(device=self.device, dtype=torch.long)
        result = {}
        for name, value in self.values.items():
            rows = self.rows[name.split("_", 1)[0]][ids]
            selected = value[rows].detach()
            result[name] = selected.cpu().clone() if cpu else selected.clone()
        return result

    @torch.no_grad()
    def restore(self, env_ids: torch.Tensor, parameters: dict[str, torch.Tensor]) -> None:
        """Restore selected worlds from captured SI parameters, rejecting incomplete or non-finite snapshots.

        Callers restoring a live simulator must subsequently notify its physics manager of body-inertial
        and shape-property changes. Isolated collision models read these arrays directly.
        """
        if set(parameters) != set(self.values):
            raise ValueError("Per-world physical parameter fields differ from the recorded schema.")
        ids = env_ids.to(device=self.device, dtype=torch.long)
        if ids.ndim != 1 or ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("Physical parameter restoration requires unique valid world indices.")
        updates = {}
        for name, value in self.values.items():
            rows = self.rows[name.split("_", 1)[0]][ids]
            source = parameters[name].to(value)
            if source.shape != (*rows.shape, *value.shape[1:]) or not bool(torch.isfinite(source).all()):
                raise ValueError(f"Invalid physical parameter shape or value for {name}.")
            updates[name] = (rows, source)
        for name, (rows, source) in updates.items():
            self.values[name][rows] = source

    def _world_rows(self, owners: torch.Tensor) -> torch.Tensor:
        indices = (owners >= 0).nonzero().flatten()
        selected = owners[indices].long()
        counts = torch.bincount(selected, minlength=self.num_envs)
        if len(counts) != self.num_envs or not bool((counts > 0).all() & (counts == counts[0]).all()):
            raise ValueError("Per-world physical context requires the same body and shape layout in every world.")
        return indices[torch.argsort(selected, stable=True)].reshape(self.num_envs, int(counts[0]))
