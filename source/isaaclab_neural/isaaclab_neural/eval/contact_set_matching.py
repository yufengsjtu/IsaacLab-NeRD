# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Permutation-invariant metrics for comparing reconstructed contact sets."""

from __future__ import annotations

import math

import torch


def _contact_vectors(value: torch.Tensor, num_contacts: int) -> torch.Tensor:
    """Reshape flattened contact vectors to ``(batch, contacts, components)``."""
    return value.reshape(value.shape[0], num_contacts, -1)


def _deduplicated_count(points: torch.Tensor, tolerance: float) -> int:
    """Count permutation-invariant radius-connected point clusters."""
    if points.shape[0] == 0:
        return 0
    adjacency = torch.cdist(points, points) <= tolerance
    visited = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    component_count = 0
    for start in range(points.shape[0]):
        if bool(visited[start]):
            continue
        component_count += 1
        pending = [start]
        visited[start] = True
        while pending:
            current = pending.pop()
            neighbors = torch.nonzero(adjacency[current] & ~visited, as_tuple=False).flatten().tolist()
            for neighbor in neighbors:
                visited[neighbor] = True
                pending.append(neighbor)
    return component_count


def _symmetric_selected_max(
    dataset_value: torch.Tensor,
    runtime_value: torch.Tensor,
    dataset_to_runtime: torch.Tensor,
    runtime_to_dataset: torch.Tensor,
) -> float:
    """Return the largest bidirectional error under one tuple correspondence."""
    dataset_error = torch.linalg.vector_norm(
        dataset_value - runtime_value[dataset_to_runtime],
        dim=-1,
    )
    runtime_error = torch.linalg.vector_norm(
        runtime_value - dataset_value[runtime_to_dataset],
        dim=-1,
    )
    return float(torch.maximum(dataset_error.max(), runtime_error.max()))


def _metric_max(values: list[float]) -> float:
    """Return a maximum while preserving NaNs for strict validation."""
    if any(math.isnan(value) for value in values):
        return float("nan")
    return max(values, default=0.0)


def contact_set_matching_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
    num_contacts: int,
    *,
    deduplication_tolerance: float,
    normal_tolerance: float = 1.0e-3,
) -> dict[str, float]:
    """Compare active contacts as unordered point sets.

    Candidate pairs are selected using ``contact_points_1``. A single
    correspondence minimizes the worst normalized point-0, normal, depth, and
    thickness error so all tuple fields retain their association.
    """
    if deduplication_tolerance <= 0:
        raise ValueError("deduplication_tolerance must be positive.")
    if normal_tolerance <= 0:
        raise ValueError("normal_tolerance must be positive.")
    dataset_mask = dataset_inputs["contact_masks"].bool()
    runtime_mask = runtime_inputs["contact_masks"].bool()
    dataset_point0 = _contact_vectors(dataset_inputs["contact_points_0"], num_contacts)
    runtime_point0 = _contact_vectors(runtime_inputs["contact_points_0"], num_contacts)
    dataset_point1 = _contact_vectors(dataset_inputs["contact_points_1"], num_contacts)
    runtime_point1 = _contact_vectors(runtime_inputs["contact_points_1"], num_contacts)
    dataset_normal = _contact_vectors(dataset_inputs["contact_normals"], num_contacts)
    runtime_normal = _contact_vectors(runtime_inputs["contact_normals"], num_contacts)
    dataset_depth = _contact_vectors(dataset_inputs["contact_depths"], num_contacts)
    runtime_depth = _contact_vectors(runtime_inputs["contact_depths"], num_contacts)
    dataset_thickness0 = _contact_vectors(dataset_inputs["contact_thicknesses_0"], num_contacts)
    runtime_thickness0 = _contact_vectors(runtime_inputs["contact_thicknesses_0"], num_contacts)
    dataset_thickness1 = _contact_vectors(dataset_inputs["contact_thicknesses_1"], num_contacts)
    runtime_thickness1 = _contact_vectors(runtime_inputs["contact_thicknesses_1"], num_contacts)

    chamfer_values = []
    hausdorff_values = []
    point0_max_values = []
    normal_max_values = []
    depth_max_values = []
    thickness0_max_values = []
    thickness1_max_values = []
    dataset_unique_counts = []
    runtime_unique_counts = []
    unique_count_differences = []
    one_sided_empty_count = 0

    for env_index in range(dataset_mask.shape[0]):
        dataset_active = dataset_mask[env_index]
        runtime_active = runtime_mask[env_index]
        dataset_points = dataset_point1[env_index, dataset_active]
        runtime_points = runtime_point1[env_index, runtime_active]
        dataset_unique = _deduplicated_count(dataset_points, deduplication_tolerance)
        runtime_unique = _deduplicated_count(runtime_points, deduplication_tolerance)
        dataset_unique_counts.append(dataset_unique)
        runtime_unique_counts.append(runtime_unique)
        unique_count_differences.append(abs(dataset_unique - runtime_unique))

        if dataset_points.shape[0] == 0 and runtime_points.shape[0] == 0:
            chamfer_values.append(0.0)
            hausdorff_values.append(0.0)
            point0_max_values.append(0.0)
            normal_max_values.append(0.0)
            depth_max_values.append(0.0)
            thickness0_max_values.append(0.0)
            thickness1_max_values.append(0.0)
            continue
        if dataset_points.shape[0] == 0 or runtime_points.shape[0] == 0:
            one_sided_empty_count += 1
            chamfer_values.append(float("inf"))
            hausdorff_values.append(float("inf"))
            point0_max_values.append(float("inf"))
            normal_max_values.append(float("inf"))
            depth_max_values.append(float("inf"))
            thickness0_max_values.append(float("inf"))
            thickness1_max_values.append(float("inf"))
            continue

        distances = torch.cdist(dataset_points, runtime_points)
        dataset_nearest = distances.min(dim=1).values
        runtime_nearest = distances.min(dim=0).values
        candidate_pairs = distances <= deduplication_tolerance
        tuple_cost = torch.stack(
            (
                torch.cdist(
                    dataset_point0[env_index, dataset_active],
                    runtime_point0[env_index, runtime_active],
                )
                / deduplication_tolerance,
                torch.cdist(
                    dataset_normal[env_index, dataset_active],
                    runtime_normal[env_index, runtime_active],
                )
                / normal_tolerance,
                torch.cdist(
                    dataset_depth[env_index, dataset_active],
                    runtime_depth[env_index, runtime_active],
                )
                / deduplication_tolerance,
                torch.cdist(
                    dataset_thickness0[env_index, dataset_active],
                    runtime_thickness0[env_index, runtime_active],
                )
                / deduplication_tolerance,
                torch.cdist(
                    dataset_thickness1[env_index, dataset_active],
                    runtime_thickness1[env_index, runtime_active],
                )
                / deduplication_tolerance,
            ),
            dim=0,
        ).amax(dim=0)
        tuple_cost = tuple_cost.masked_fill(~candidate_pairs, float("inf"))
        dataset_to_runtime = tuple_cost.min(dim=1).indices
        runtime_to_dataset = tuple_cost.min(dim=0).indices
        chamfer_values.append(float(0.5 * (dataset_nearest.mean() + runtime_nearest.mean())))
        hausdorff_values.append(float(torch.maximum(dataset_nearest.max(), runtime_nearest.max())))
        point0_max_values.append(
            _symmetric_selected_max(
                dataset_point0[env_index, dataset_active],
                runtime_point0[env_index, runtime_active],
                dataset_to_runtime,
                runtime_to_dataset,
            )
        )
        normal_max_values.append(
            _symmetric_selected_max(
                dataset_normal[env_index, dataset_active],
                runtime_normal[env_index, runtime_active],
                dataset_to_runtime,
                runtime_to_dataset,
            )
        )
        depth_max_values.append(
            _symmetric_selected_max(
                dataset_depth[env_index, dataset_active],
                runtime_depth[env_index, runtime_active],
                dataset_to_runtime,
                runtime_to_dataset,
            )
        )
        thickness0_max_values.append(
            _symmetric_selected_max(
                dataset_thickness0[env_index, dataset_active],
                runtime_thickness0[env_index, runtime_active],
                dataset_to_runtime,
                runtime_to_dataset,
            )
        )
        thickness1_max_values.append(
            _symmetric_selected_max(
                dataset_thickness1[env_index, dataset_active],
                runtime_thickness1[env_index, runtime_active],
                dataset_to_runtime,
                runtime_to_dataset,
            )
        )

    return {
        "set_chamfer_distance_mean": sum(chamfer_values) / max(len(chamfer_values), 1),
        "set_hausdorff_distance_mean": sum(hausdorff_values) / max(len(hausdorff_values), 1),
        "set_hausdorff_distance_max": _metric_max(hausdorff_values),
        "matched_point0_distance_max": _metric_max(point0_max_values),
        "matched_normal_l2_max": _metric_max(normal_max_values),
        "matched_depth_abs_max": _metric_max(depth_max_values),
        "matched_thickness0_abs_max": _metric_max(thickness0_max_values),
        "matched_thickness1_abs_max": _metric_max(thickness1_max_values),
        "dataset_unique_active_mean": sum(dataset_unique_counts) / max(len(dataset_unique_counts), 1),
        "runtime_unique_active_mean": sum(runtime_unique_counts) / max(len(runtime_unique_counts), 1),
        "unique_count_abs_diff_mean": sum(unique_count_differences) / max(len(unique_count_differences), 1),
        "unique_count_abs_diff_max": float(max(unique_count_differences, default=0)),
        "one_sided_empty_count": float(one_sided_empty_count),
        "one_sided_empty_fraction": one_sided_empty_count / max(dataset_mask.shape[0], 1),
    }


def contact_set_tolerance_failures(
    metrics: dict[str, float],
    *,
    point_tolerance: float,
    normal_tolerance: float,
) -> list[str]:
    """Return reasons why matched contact sets are not geometrically equivalent."""
    checks = {
        "set_hausdorff_distance_max": point_tolerance,
        "matched_point0_distance_max": point_tolerance,
        "matched_depth_abs_max": point_tolerance,
        "matched_thickness0_abs_max": point_tolerance,
        "matched_thickness1_abs_max": point_tolerance,
        "matched_normal_l2_max": normal_tolerance,
    }
    failures = []
    if metrics["one_sided_empty_count"] > 0:
        failures.append(f"one_sided_empty_count={metrics['one_sided_empty_count']:.8g}")
    if metrics["unique_count_abs_diff_max"] > 0:
        failures.append(f"unique_count_abs_diff_max={metrics['unique_count_abs_diff_max']:.8g}")
    for key, tolerance in checks.items():
        value = metrics[key]
        if not math.isfinite(value) or value > tolerance:
            failures.append(f"{key}={value:.8g} is non-finite or exceeds {tolerance:.8g}")
    return failures
