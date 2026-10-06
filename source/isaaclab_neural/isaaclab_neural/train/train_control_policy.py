# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL control policy on a task registered by :mod:`isaaclab_neural.envs`.

Arguments are those of ``isaaclab train`` with the RSL-RL backend. The module form
also works under ``torch.distributed.run`` together with ``--distributed``.
"""

from __future__ import annotations

import sys


def main() -> None:
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

    import isaaclab_neural.envs  # noqa: F401 - registers the NeRD policy tasks

    run(sys.argv[1:])


if __name__ == "__main__":
    main()
