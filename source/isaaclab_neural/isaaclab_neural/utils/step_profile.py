# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in segmented timing for NeRD physics / contact hot paths.

Enable with environment variable ``NERD_STEP_PROFILE=1`` or by calling
:func:`enable`. Timers synchronize CUDA so section costs are comparable across
Warp kernels and PyTorch ops. Leave disabled during normal training.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any

from isaaclab_neural.utils.time_report import TimeProfiler, TimeReport

ENV_VAR = "NERD_STEP_PROFILE"

STEP_TIMER_NAMES = (
    "contact_prepare",
    "actuator",
    "solver_update_states",
    "adapter_update",
    "to_neural_inputs",
    "model_forward",
    "eval_fk",
    "env_step_total",
)

_enabled: bool = False
_report: TimeReport | None = None


def _env_flag_enabled() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Return whether NeRD step profiling is active."""
    return _enabled


def get_report() -> TimeReport | None:
    """Return the active :class:`TimeReport`, or ``None`` if profiling is off."""
    return _report


def enable(*, cuda_synchronize: bool = True) -> TimeReport:
    """Enable profiling and return the shared timer report."""
    global _enabled, _report
    _enabled = True
    if _report is None:
        _report = TimeReport(cuda_synchronize=cuda_synchronize)
        _report.add_timers(list(STEP_TIMER_NAMES))
    else:
        _report.cuda_synchronize = cuda_synchronize
        for name in STEP_TIMER_NAMES:
            if name not in _report.timers:
                _report.add_timer(name)
    return _report


def disable() -> None:
    """Disable profiling (does not clear accumulated timings)."""
    global _enabled
    _enabled = False


def reset() -> None:
    """Reset accumulated section timings when profiling is active."""
    if _report is not None:
        _report.reset_timer()


def enable_from_env(*, cuda_synchronize: bool = True) -> TimeReport | None:
    """Enable profiling when ``NERD_STEP_PROFILE`` is set; otherwise no-op."""
    if _env_flag_enabled():
        return enable(cuda_synchronize=cuda_synchronize)
    return None


def summary_dict() -> dict[str, float]:
    """Return ``{timer_name: total_seconds}`` for configured timers."""
    if _report is None:
        return {}
    return {name: timer.total_time for name, timer in _report.timers.items()}


def print_summary(*, steps: int | None = None) -> None:
    """Print accumulated timings, optionally normalized per step."""
    if _report is None:
        print("[nerd_step_profile] inactive")
        return
    totals = summary_dict()
    active = {k: v for k, v in totals.items() if v > 0.0}
    if not active:
        print("[nerd_step_profile] no samples")
        return
    lines = ["[nerd_step_profile]"]
    for name, total in active.items():
        if steps and steps > 0:
            lines.append(f"  {name}: {total * 1000.0 / steps:.3f} ms/step ({total:.3f} s total)")
        else:
            lines.append(f"  {name}: {total:.3f} s")
    print("\n".join(lines))


@contextmanager
def section(timer_name: str):
    """Profile a named hot-path section when profiling is enabled."""
    if not _enabled and _env_flag_enabled():
        enable()
    if not _enabled or _report is None:
        yield
        return
    if timer_name not in _report.timers:
        _report.add_timer(timer_name)
    with TimeProfiler(_report, timer_name):
        yield


def section_cm(timer_name: str) -> Any:
    """Return a context manager for ``timer_name`` (nullcontext when disabled)."""
    if not _enabled or _report is None:
        return nullcontext()
    if timer_name not in _report.timers:
        _report.add_timer(timer_name)
    return TimeProfiler(_report, timer_name)
