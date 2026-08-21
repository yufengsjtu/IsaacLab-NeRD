# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Timing helpers for NeRD training loops."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch


class Timer:
    """Accumulates elapsed time for one named section."""

    def __init__(
        self,
        name: str,
        *,
        record_samples: bool = False,
        cuda_event_timing: bool = False,
    ):
        self.name = name
        self.record_samples = record_samples
        self.cuda_event_timing = cuda_event_timing and torch.cuda.is_available()
        self.total_time = 0.0
        self.samples: list[float] = []
        self.enabled = True
        self.start_time: float | None = None
        self._start_cuda_event: Any | None = None
        self._pending_cuda_events: list[tuple[Any, Any]] = []

    def tic(self) -> None:
        """Start the timer."""
        if not self.enabled:
            return
        if self.cuda_event_timing:
            self._start_cuda_event = torch.cuda.Event(enable_timing=True)
            self._start_cuda_event.record()
        else:
            self.start_time = time.perf_counter()

    def toc(self) -> None:
        """Stop the timer and accumulate or queue the elapsed time."""
        if not self.enabled:
            return
        if self.cuda_event_timing:
            if self._start_cuda_event is None:
                return
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._pending_cuda_events.append((self._start_cuda_event, end_event))
            self._start_cuda_event = None
            return
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.total_time += elapsed
            if self.record_samples:
                self.samples.append(elapsed)
            self.start_time = None

    def finalize_cuda_events(self) -> None:
        """Resolve pending CUDA-event pairs after one outer synchronization."""
        for start_event, end_event in self._pending_cuda_events:
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
            self.total_time += elapsed
            if self.record_samples:
                self.samples.append(elapsed)
        self._pending_cuda_events.clear()

    def has_pending_cuda_events(self) -> bool:
        """Return whether this timer has completed event pairs to resolve."""
        return bool(self._pending_cuda_events)

    def reset(self) -> None:
        """Clear accumulated time and samples."""
        self.enabled = True
        self.start_time = None
        self._start_cuda_event = None
        self._pending_cuda_events.clear()
        self.total_time = 0.0
        self.samples.clear()

    def print(self, string_mode: bool = False, in_second: bool = True) -> str | None:
        """Print or return a formatted timing summary."""
        if in_second:
            timing_report = f"time({self.name}): {self.total_time:.3f} sec"
        else:
            timing_report = f"time({self.name}): {self.total_time / 60:.3f} min"

        if string_mode:
            return timing_report
        print(timing_report)
        return None


class TimeReport:
    """Collection of named timers used by training loops."""

    def __init__(
        self,
        cuda_synchronize: bool = False,
        *,
        cuda_event_timing: bool = False,
        record_samples: bool = False,
    ):
        if cuda_synchronize and cuda_event_timing:
            raise ValueError("cuda_synchronize and cuda_event_timing are mutually exclusive.")
        self.timers: dict[str, Timer] = {}
        self.cuda_synchronize = cuda_synchronize
        self.cuda_event_timing = cuda_event_timing
        self.record_samples = record_samples

    def add_timer(self, timer_name: str) -> None:
        """Add a named timer."""
        self.timers[timer_name] = Timer(
            timer_name,
            record_samples=self.record_samples,
            cuda_event_timing=self.cuda_event_timing,
        )

    def add_timers(self, timer_names: list[str]) -> None:
        """Add multiple named timers."""
        for timer_name in timer_names:
            self.add_timer(timer_name)

    def reset_timer(self, timer_name: str | None = None) -> None:
        """Reset one timer or all timers."""
        if timer_name is None:
            for timer in self.timers.values():
                timer.reset()
        else:
            self.timers[timer_name].reset()

    def set_timer_enabled(self, timer_name: str, enabled: bool) -> None:
        """Enable or disable one timer."""
        self.timers[timer_name].enabled = enabled

    def start_timer(self, timer_name: str) -> None:
        """Start a named timer."""
        timer = self.timers[timer_name]
        if not timer.enabled:
            return
        if self.cuda_synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.tic()

    def end_timer(self, timer_name: str) -> None:
        """Stop a named timer."""
        timer = self.timers[timer_name]
        if not timer.enabled:
            return
        if self.cuda_synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.toc()

    def finalize(self) -> None:
        """Resolve all completed CUDA-event pairs with one synchronization."""
        if any(timer.has_pending_cuda_events() for timer in self.timers.values()):
            torch.cuda.synchronize()
            for timer in self.timers.values():
                timer.finalize_cuda_events()

    def print(self, string_mode: bool = False, in_second: bool = True) -> str | None:
        """Print or return all timer summaries."""
        timing_summary = ", ".join(timer.print(string_mode=True, in_second=in_second) for timer in self.timers.values())
        if string_mode:
            return timing_summary
        print(timing_summary)
        return None

    def as_dict(self) -> dict[str, float]:
        """Return accumulated timer durations in seconds."""
        return {name: timer.total_time for name, timer in self.timers.items()}

    def sample_statistics(self, skip_first: int = 0) -> dict[str, float]:
        """Return mean and percentile statistics for recorded timer samples."""
        if not self.record_samples:
            return {}
        statistics: dict[str, float] = {}
        for name, timer in self.timers.items():
            samples = timer.samples[skip_first:]
            if not samples:
                continue
            values = torch.tensor(samples, dtype=torch.float64)
            statistics[f"{name}_count"] = float(values.numel())
            statistics[f"{name}_mean_seconds"] = float(values.mean())
            statistics[f"{name}_p50_seconds"] = float(torch.quantile(values, 0.50))
            statistics[f"{name}_p95_seconds"] = float(torch.quantile(values, 0.95))
        return statistics


@contextmanager
def TimeProfiler(time_report: TimeReport, timer_name: str):
    """Profile a block using one :class:`TimeReport` timer."""
    time_report.start_timer(timer_name)
    try:
        yield
    finally:
        time_report.end_timer(timer_name)
