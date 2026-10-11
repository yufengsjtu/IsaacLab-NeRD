# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a stock RSL-RL checkpoint on an ``isaaclab_neural`` gym task.

``./isaaclab.sh play`` only imports :mod:`isaaclab_tasks`, so NeRD gym ids are
missing from the registry. This module registers those tasks, then runs the
same RSL-RL play entrypoint. It also keeps playback going when TorchScript
export is a no-op.

Example::

    PYTHONPATH="$(pwd)/source/isaaclab_neural:${PYTHONPATH}" ./isaaclab.sh -p \\
      -m isaaclab_neural.eval.play_rsl_rl \\
      --task Isaac-Velocity-Flat-G1 \\
      --checkpoint pretrained/control_policy/g1_flat_terrain/model_1499.pt \\
      --num_envs 1 --real-time --viz newton presets=newton_mjwarp
"""

from __future__ import annotations

import runpy
from collections.abc import Callable
from typing import Any

import isaaclab_neural.envs  # noqa: F401


def _patch_optional_policy_export() -> None:
    """Keep play running when TorchScript export is a no-op.

    Isaac Sim commonly disables JIT (``PYTORCH_JIT=0``). Then
    ``torch.jit.script`` returns the original ``nn.Module``, and rsl-rl's
    ``export_policy_to_jit`` crashes on ``.save()`` before the control loop.
    Playback does not need those files.
    """
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    def _wrap(export_fn: Callable[..., Any]) -> Callable[..., Any]:
        def _safe_export(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return export_fn(self, *args, **kwargs)
            except AttributeError as exc:
                print(f"[WARN] Skipping policy export ({type(exc).__name__}: {exc}). Playback will continue.")
                return None

        return _safe_export

    OnPolicyRunner.export_policy_to_jit = _wrap(OnPolicyRunner.export_policy_to_jit)
    OnPolicyRunner.export_policy_to_onnx = _wrap(OnPolicyRunner.export_policy_to_onnx)


if __name__ == "__main__":
    _patch_optional_policy_export()
    runpy.run_module("isaaclab_rl.entrypoints.backends.play_rsl_rl", run_name="__main__")
