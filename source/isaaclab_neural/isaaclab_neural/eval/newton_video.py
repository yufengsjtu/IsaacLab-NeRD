# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless Newton GL video capture with a camera that follows a point in the scene.

Frames are rendered by the Isaac Lab Newton GL visualizer and piped to the system ``ffmpeg`` executable.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from typing import Any

import numpy as np


def newton_video_visualizer_cfg(width: int = 960, height: int = 540, focal_length: float = 20.0) -> Any:
    """Return a headless Newton GL visualizer configuration for offscreen capture.

    Args:
        width: Frame width [px].
        height: Frame height [px].
        focal_length: Camera focal length; larger values zoom in.

    Returns:
        A :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg` to append to ``SimulationCfg.visualizer_cfgs``.
    """
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg  # noqa: PLC0415

    return NewtonGLVisualizerCfg(
        headless=True, window_width=width, window_height=height, enable_picking=False, focal_length=focal_length
    )


class NewtonFollowVideo:
    """Write an mp4 of the Newton GL visualizer while following a target point."""

    def __init__(
        self,
        sim: Any,
        path: str,
        fps: float = 50.0,
        distance: float = 4.0,
        azimuth_deg: float = 120.0,
        elevation_deg: float = 12.0,
        target_height: float = 0.65,
        smoothing: float = 0.1,
    ):
        """Open the output file.

        Args:
            sim: Simulation context with an active Newton GL visualizer.
            path: Output mp4 path.
            fps: Frame rate of the written video [Hz].
            distance: Camera distance from the target [m].
            azimuth_deg: Camera azimuth about the world z axis [deg].
            elevation_deg: Camera elevation above the horizontal plane [deg].
            target_height: Height of the look-at point [m].
            smoothing: Exponential smoothing factor of the look-at point per frame.

        Raises:
            RuntimeError: No capture-capable visualizer is active or ``ffmpeg`` is not installed.
        """
        self.sim = sim
        candidates = [viz for viz in sim.visualizers if hasattr(viz, "render_rgb_array")]
        if not candidates:
            raise RuntimeError("No capture-capable visualizer is active; add newton_video_visualizer_cfg().")
        self.viz = candidates[0]
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to write videos.")
        self.path, self.fps = path, fps
        azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
        self.offset = distance * np.array(
            [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)]
        )
        self.target_height, self.smoothing = target_height, smoothing
        self.lookat: np.ndarray | None = None
        self.process: subprocess.Popen | None = None
        self.frames = 0

    def capture(self, target: np.ndarray) -> None:
        """Render and append one frame.

        Args:
            target: Point to follow in the world frame [m], shape ``[3]``; only its horizontal position is used.
        """
        point = np.array([target[0], target[1], self.target_height], dtype=float)
        self.lookat = point if self.lookat is None else (1 - self.smoothing) * self.lookat + self.smoothing * point
        self.viz.set_camera_view(tuple(self.lookat + self.offset), tuple(self.lookat))
        self.sim.render()
        frame = self.viz.render_rgb_array()
        if frame is None:
            return
        frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if self.process is None:
            height, width = frame.shape[:2]
            self.process = subprocess.Popen(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", self.path,
                ],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        """Finish the mp4 file."""
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
