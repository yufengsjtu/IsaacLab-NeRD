# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version-checked eager MJWarp deterministic workspaces for Keyboard collection."""

from __future__ import annotations

import hashlib
import inspect
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

PROFILE = "mjwarp_3_12_eager_v1"
_SOURCE_HASHES = {
    "mujoco_warp._src.constraint": "001a2e7630ca05a61b406daf0bca2f769f8a580f7134629d5b45f37bd8f473d1",
    "mujoco_warp._src.solver": "509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3",
    "warp._src.deterministic": "782512e59e33d43d41f4c7dddeb2a62fce8371fa4d3a88a2a20e66a5497eec31",
}
_FIXED_FACTORIES = {
    "mujoco_warp._src.constraint": {
        "_friction_dof",
        "_friction_tendon",
        "_limit_slide_hinge",
        "_limit_ball",
        "_limit_tendon",
        "_efc_contact_init",
    },
    "mujoco_warp._src.solver": {"_update_constraint_efc", "_update_gradient_grad"},
}
_SCOPE_LOCK = threading.Lock()


def workspace_contract() -> dict[str, Any]:
    """Return the exact audited workspace profile recorded in data provenance."""
    return {"profile": PROFILE, "source_sha256": dict(_SOURCE_HASHES)}


def hessian_record_bound(
    row_nnz: np.ndarray, indices: np.ndarray, counts: np.ndarray, stride: int, lanes: int, max_dofs: int
) -> int:
    """Bound per-thread triangular scatter writes over the complete active row set.

    Unused row slots are ignored; invalid active rows or capacity overflow raise
    rather than being clipped. This bounds scratch records, not physical contacts.
    """
    if (
        stride <= 0
        or lanes <= 0
        or max_dofs <= 0
        or row_nnz.ndim != 2
        or indices.ndim != 2
        or counts.ndim != 1
        or row_nnz.shape[0] != counts.size
        or indices.shape[0] != counts.size
        or np.any(counts < 0)
        or np.any(counts > indices.shape[1])
        or any(not np.issubdtype(value.dtype, np.integer) for value in (row_nnz, indices, counts))
    ):
        raise ValueError("Invalid deterministic Hessian row layout or capacity overflow.")
    worlds, slots = np.nonzero(np.arange(indices.shape[1])[None, :] < counts[:, None])
    selected = indices[worlds, slots]
    if np.any(selected < 0) or np.any(selected >= row_nnz.shape[1]):
        raise ValueError("Invalid active deterministic Hessian row index.")
    widths = row_nnz[worlds, selected]
    width = int(widths.max(initial=0))
    if np.any(widths < 0) or width > max_dofs:
        raise ValueError("Invalid deterministic Hessian row support.")
    visits = (int(counts.max(initial=0)) + stride - 1) // stride
    triangles = width * (width + 1) // 2
    return max(1, visits * ((triangles + lanes - 1) // lanes))


def _record_bound(kernel: Any, values: list[Any], bounds: Any) -> int | None:
    module = kernel.func.__module__
    factory, separator, suffix = kernel.func.__qualname__.partition(".")
    if not separator or suffix != "<locals>.kernel":
        return None
    if factory in _FIXED_FACTORIES.get(module, ()):
        return 0
    if module != "mujoco_warp._src.solver":
        return None
    if factory not in (
        "_update_constraint_init_qfrc_constraint_sparse",
        "_JTDACJ_sparse",
        "_update_gradient_h_incremental_sparse",
    ):
        return None
    arguments = dict(zip((argument.label for argument in kernel.adj.args), values, strict=True))
    if factory == "_update_constraint_init_qfrc_constraint_sparse":
        return int(arguments["qfrc_constraint_out"].shape[1])
    if len(bounds.shape) != 3:
        raise ValueError("Unexpected deterministic Hessian launch dimensions.")
    if factory == "_JTDACJ_sparse":
        rows, counts, stride, output = ("efc_jtdaj_adr_in", "efc_jtdaj_nblock_in", "groups_per_world", "h_out")
    else:
        rows, counts, stride, output = ("quad_changed_ids_in", "quad_changed_count_in", "slots_per_world", "ctx_h_out")
    return hessian_record_bound(
        arguments["efc_J_rownnz_in"].numpy(),
        arguments[rows].numpy(),
        arguments[counts].numpy(),
        int(arguments[stride]),
        int(bounds.shape[2]),
        int(arguments[output].shape[1]),
    )


@contextmanager
def keyboard_deterministic_workspace(enabled: bool) -> Iterator[dict[str, Any] | None]:
    """Scope an audited eager-only scratch allocation profile to this simulation thread.

    Other kernels retain their original bounds. Compiled metadata, physics inputs,
    contact capacities and overflow checks are unchanged. Unsupported library
    sources fail before any simulation rather than receiving guessed bounds.
    """
    if not enabled:
        yield None
        return
    from mujoco_warp._src import constraint, solver
    from warp._src import context, deterministic

    sources = {
        module.__name__: hashlib.sha256(inspect.getsource(module).encode()).hexdigest()
        for module in (constraint, solver, deterministic)
    }
    if sources != _SOURCE_HASHES:
        raise RuntimeError("Unsupported MJWarp/Warp sources for the Keyboard deterministic workspace profile.")
    if not _SCOPE_LOCK.acquire(blocking=False):
        raise RuntimeError("Keyboard deterministic workspace scopes cannot overlap.")
    original = deterministic.launch_deterministic
    owner = threading.get_ident()

    def launch(kernel, hooks, user_params, bounds, device, stream, max_blocks, block_dim, det_meta, fwd_args, **kwargs):
        if threading.get_ident() == owner and not kwargs.get("adjoint", False):
            if context.runtime.captures:
                raise RuntimeError("Keyboard deterministic workspace profile requires eager execution.")
            bound = _record_bound(kernel, fwd_args, bounds)
            if bound is not None:
                det_meta = replace(det_meta, max_records=bound)
        return original(
            kernel, hooks, user_params, bounds, device, stream, max_blocks, block_dim, det_meta, fwd_args, **kwargs
        )

    deterministic.launch_deterministic = launch
    try:
        yield {"profile": PROFILE, "source_sha256": sources}
    finally:
        deterministic.launch_deterministic = original
        _SCOPE_LOCK.release()
