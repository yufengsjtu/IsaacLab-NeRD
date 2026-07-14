# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD stage post-processing helpers for NeRD environments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pxr import Usd

_MATERIAL_BINDING_RELATIONSHIP = "material:binding"


def _has_material_binding(prim: Usd.Prim) -> bool:
    """Return whether *prim* authors any ``material:binding`` relationship."""
    for relationship in prim.GetRelationships():
        name = relationship.GetName()
        if name == _MATERIAL_BINDING_RELATIONSHIP or name.startswith(_MATERIAL_BINDING_RELATIONSHIP + ":"):
            return True
    return False


def _find_instance_roots_with_missing_api(root_prim: Usd.Prim) -> set[Usd.Prim]:
    """Find instance roots containing binding relationships without the applied API."""
    from pxr import Usd, UsdShade

    instance_roots: set[Usd.Prim] = set()
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange(root_prim, predicate):
        if not prim.IsInstanceProxy():
            continue
        if prim.HasAPI(UsdShade.MaterialBindingAPI) or not _has_material_binding(prim):
            continue

        instance_root = prim
        while instance_root and not instance_root.IsInstance():
            instance_root = instance_root.GetParent()
        if instance_root and instance_root.IsValid():
            instance_roots.add(instance_root)
    return instance_roots


def apply_missing_material_binding_api(prim_path: str | None = None, stage: Usd.Stage | None = None) -> int:
    """Apply :class:`UsdShade.MaterialBindingAPI` where bindings exist but the schema is missing.

    Some robot USD assets author ``material:binding`` relationships on visual mesh
    prims without applying the ``UsdShade.MaterialBindingAPI`` schema. Newer USD
    runtimes emit a ``BindingsAtPrim`` warning for every such prim. This helper
    walks the stage and applies the schema in place, silencing the warning without
    changing any binding target.

    Instance proxies are read-only, so any instance root containing an affected
    prim is first made non-instanceable. Callers should pass the smallest root
    that Newton is about to ingest (normally ``/World/envs/env_0``) to avoid
    unnecessarily expanding every cloned environment.

    Args:
        prim_path: Root prim path to traverse. Defaults to the pseudo-root so the
            entire stage is processed.
        stage: Stage to operate on. Defaults to the current stage.

    Returns:
        Number of prims that received a newly applied ``MaterialBindingAPI``.
    """
    from pxr import Usd, UsdShade

    if stage is None:
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
    if stage is None:
        return 0

    if prim_path is None:
        root_prim = stage.GetPseudoRoot()
    else:
        root_prim = stage.GetPrimAtPath(prim_path)
        if root_prim is None or not root_prim.IsValid():
            raise ValueError(f"Prim path '{prim_path}' does not exist on the stage.")
    if root_prim is None:
        return 0

    # Material bindings commonly live below referenced, instanceable robot
    # assets. Instance-proxy descendants cannot receive authored schemas. Make
    # only the affected instance roots editable, then traverse their expanded
    # contents on the next pass. Repeat for nested instances.
    while True:
        instance_roots = _find_instance_roots_with_missing_api(root_prim)
        if not instance_roots:
            break
        for instance_root in instance_roots:
            instance_root.SetInstanceable(False)

    applied_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            continue
        if not _has_material_binding(prim):
            continue
        try:
            UsdShade.MaterialBindingAPI.Apply(prim)
        except Exception:  # noqa: BLE001 - non-editable prims (e.g. inside prototypes) are skipped
            continue
        applied_count += 1
    return applied_count


@contextmanager
def newton_material_binding_api_autofix() -> Iterator[None]:
    """Apply :func:`apply_missing_material_binding_api` before Newton parses the stage.

    Newton builds its model by calling :meth:`newton.ModelBuilder.add_usd`, which
    resolves visual material bindings and emits a ``BindingsAtPrim`` warning for
    every prim that authors bindings without an applied ``MaterialBindingAPI``.
    Because that parse happens deep inside environment construction, the schema
    must be applied *before* it runs. This context manager temporarily wraps
    ``ModelBuilder.add_usd`` so the fix is applied to the stage it is about to
    ingest, then restores the original method on exit.

    The wrapper is a no-op when Newton is unavailable, so it is safe to use around
    environment creation regardless of the active physics backend.
    """
    try:
        from newton import ModelBuilder
    except Exception:  # noqa: BLE001 - Newton not installed / not the active backend
        yield
        return

    original_add_usd = ModelBuilder.add_usd

    def _patched_add_usd(self, source, *args, **kwargs):
        # ``source`` may be a file path or an in-memory stage; only stages can be
        # patched in place before parsing.
        if not isinstance(source, str):
            try:
                root_path = kwargs.get("root_path")
                # Newton first parses the stage outside cloned env subtrees with
                # ``ignore_paths`` and then parses env_0 explicitly. Do not scan
                # the whole stage on that first call, which would de-instance all
                # cloned environments unnecessarily.
                if root_path is not None or not kwargs.get("ignore_paths"):
                    apply_missing_material_binding_api(prim_path=root_path, stage=source)
            except Exception:  # noqa: BLE001 - never block model building on a cosmetic fix
                pass
        return original_add_usd(self, source, *args, **kwargs)

    ModelBuilder.add_usd = _patched_add_usd
    try:
        yield
    finally:
        ModelBuilder.add_usd = original_add_usd
