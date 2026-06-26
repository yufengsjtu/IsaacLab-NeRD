from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import newton
import torch


def compute_contact_fingerprint(
    model: newton.Model,
    contact_source: Any,
) -> dict[str, Any]:
    """Build metadata that identifies the contact-input convention.

    Args:
        model: Newton model used for contact generation.
        contact_source: Either a fixed-ground ``AbstractContact`` instance,
            a ``NewtonContactAdapter`` instance, or a mapping that already
            contains contact metadata.

    Returns:
        A plain-Python dict safe to serialize into dataset metadata.
    """
    if isinstance(contact_source, Mapping):
        source_fp = dict(contact_source)
    elif hasattr(contact_source, "fingerprint"):
        source_fp = dict(contact_source.fingerprint())
    else:
        source_fp = _fixed_ground_fingerprint(model, contact_source)

    return {
        "schema": "isaaclab_neural.contact_fingerprint.v1",
        "model": _model_fingerprint(model),
        "contact": source_fp,
    }


def assert_contact_fingerprint_matches(
    dataset_fingerprint: Mapping[str, Any],
    runtime_fingerprint: Mapping[str, Any],
) -> None:
    """Raise a helpful error when dataset/runtime contact conventions differ."""
    mismatches: list[str] = []
    _compare_values(
        _normalize_for_compare(dataset_fingerprint),
        _normalize_for_compare(runtime_fingerprint),
        path="contact_fingerprint",
        mismatches=mismatches,
    )
    if mismatches:
        formatted = "\n".join(f"  - {msg}" for msg in mismatches)
        raise ValueError(
            "Dataset contact fingerprint does not match runtime contact setup:\n"
            f"{formatted}"
        )


def _model_fingerprint(model: newton.Model) -> dict[str, Any]:
    return {
        "world_count": int(model.world_count),
        "body_count": int(model.body_count),
        "shape_count": int(model.shape_count),
        "up_axis": _up_axis_value(model),
    }


def _fixed_ground_fingerprint(
    model: newton.Model,
    abstract_contact: Any,
) -> dict[str, Any]:
    num_contacts_per_env = int(getattr(abstract_contact, "num_contacts_per_env"))
    num_envs = int(getattr(abstract_contact, "num_envs", model.world_count))
    shape0 = _tensor_like_to_list(
        getattr(abstract_contact, "contact_shape0")
    )[:num_contacts_per_env]
    thickness0 = _round_floats(
        _tensor_like_to_list(getattr(abstract_contact, "contact_thickness0"))[
            :num_contacts_per_env
        ]
    )
    thickness1 = _round_floats(
        _tensor_like_to_list(getattr(abstract_contact, "contact_thickness1"))[
            :num_contacts_per_env
        ]
    )

    return {
        "contact_mode": "fixed_ground",
        "num_envs": num_envs,
        "num_contacts_per_env": num_contacts_per_env,
        "ground_shape_index": int(getattr(abstract_contact, "ground_shape_index", -1)),
        "shapes_start_offset": int(getattr(abstract_contact, "shapes_start_offset", 0)),
        "anchor_rule": "fixed_shape_anchors",
        "depth_rule": "ground_plane_height",
        "thickness_rule": "model_geometry",
        "contact_shape0_first_env": shape0,
        "contact_thickness0_first_env": thickness0,
        "contact_thickness1_first_env": thickness1,
    }


def _up_axis_value(model: newton.Model) -> Any:
    up_axis = getattr(model, "up_axis", None)
    if up_axis is None:
        return None
    if hasattr(up_axis, "name"):
        return up_axis.name
    if hasattr(up_axis, "value"):
        return up_axis.value
    return str(up_axis)


def _tensor_like_to_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _round_floats(value: Any, ndigits: int = 8) -> Any:
    if isinstance(value, list):
        return [_round_floats(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return tuple(_round_floats(v, ndigits) for v in value)
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_compare(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_compare(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_compare(v) for v in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def _compare_values(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[str],
) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            mismatches.append(f"{path}.{key}: missing in runtime")
        for key in sorted(actual_keys - expected_keys):
            mismatches.append(f"{path}.{key}: extra in runtime")
        for key in sorted(expected_keys & actual_keys):
            _compare_values(expected[key], actual[key], f"{path}.{key}", mismatches)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(
                f"{path}: length mismatch dataset={len(expected)} runtime={len(actual)}"
            )
            return
        for i, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_values(expected_item, actual_item, f"{path}[{i}]", mismatches)
        return

    if expected != actual:
        mismatches.append(f"{path}: dataset={expected!r} runtime={actual!r}")
