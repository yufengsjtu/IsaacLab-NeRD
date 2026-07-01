# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Timing helpers for NeRD training loops."""

from __future__ import annotations

from contextlib import contextmanager
import time

import torch


class Timer:
    """Accumulates elapsed wall-clock time for one named section."""

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0
        self.start_time: float | None = None

    def tic(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def toc(self) -> None:
        """Stop the timer and accumulate elapsed time."""
        if self.start_time is not None:
            self.total_time += time.time() - self.start_time
            self.start_time = None

    def reset(self) -> None:
        """Clear accumulated time."""
        self.start_time = None
        self.total_time = 0.0

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

    def __init__(self, cuda_synchronize: bool = False):
        self.timers: dict[str, Timer] = {}
        self.cuda_synchronize = cuda_synchronize

    def add_timer(self, timer_name: str) -> None:
        """Add a named timer."""
        self.timers[timer_name] = Timer(timer_name)

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

    def start_timer(self, timer_name: str) -> None:
        """Start a named timer."""
        if self.cuda_synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timers[timer_name].tic()

    def end_timer(self, timer_name: str) -> None:
        """Stop a named timer."""
        if self.cuda_synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timers[timer_name].toc()

    def print(self, string_mode: bool = False, in_second: bool = True) -> str | None:
        """Print or return all timer summaries."""
        timing_summary = ", ".join(timer.print(string_mode=True, in_second=in_second) for timer in self.timers.values())
        if string_mode:
            return timing_summary
        print(timing_summary)
        return None


@contextmanager
def TimeProfiler(time_report: TimeReport, timer_name: str):
    """Profile a block using one :class:`TimeReport` timer."""
    time_report.start_timer(timer_name)
    try:
        yield
    finally:
        time_report.end_timer(timer_name)
