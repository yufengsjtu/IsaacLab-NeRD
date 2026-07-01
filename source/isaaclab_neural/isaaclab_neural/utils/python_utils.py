# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small Python utilities used by NeRD training scripts."""

from __future__ import annotations

from datetime import datetime
import random
import re
from typing import Any

import numpy as np
import torch


def get_single_value(value: str) -> bool | int | float | str | None:
    """Parse one command-line override value."""
    if value in ("True", "False", "true", "false"):
        return value.lower() == "true"
    if value in ("None", "none", "null", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def override_cfg_entry(key: str, value: str, cfg: dict[str, Any]) -> bool:
    """Override one nested config entry addressed by a dotted key."""
    idx = key.find(".")
    if idx == -1:
        if key not in cfg:
            return False
        if value.startswith("[") and value.endswith("]"):
            cfg[key] = [get_single_value(item) for item in value[1:-1].split(",")]
        else:
            cfg[key] = get_single_value(value)
        return True

    if key[:idx] not in cfg:
        return False
    nested_cfg = cfg[key[:idx]]
    return isinstance(nested_cfg, dict) and override_cfg_entry(key[idx + 1 :], value, nested_cfg)


def handle_cfg_overrides(cfg_overrides: str, cfg: dict[str, Any]) -> None:
    """Apply ``key value`` pairs to a nested config dictionary."""
    overrides = cfg_overrides.split()
    if len(overrides) % 2 != 0:
        raise ValueError("cfg_overrides must contain key/value pairs.")
    for index in range(0, len(overrides), 2):
        key = overrides[index]
        value = overrides[index + 1]
        if not override_cfg_entry(key, value, cfg):
            print_error(f"No key {key} in config to override")


def format_dict(values: dict[Any, Any], precision: int = 8, exclude_regex: str | None = None) -> str:
    """Format a dictionary into a compact training-log string."""
    entries = []
    for key, value in values.items():
        if exclude_regex is not None and re.match(exclude_regex, str(key)):
            continue
        key_str = f"{key:.{precision}f}" if isinstance(key, float) else str(key)
        value_str = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
        entries.append(f"{key_str}: {value_str}")
    return "{" + ", ".join(entries) + "}"


def get_time_stamp() -> str:
    """Return a timestamp suitable for log-directory names."""
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_error(*message: object) -> None:
    """Print an error message and raise ``RuntimeError``."""
    print(f"\033[91mERROR {' '.join(map(str, message))}\033[0m")
    raise RuntimeError


def print_ok(*message: object) -> None:
    """Print a success message."""
    print(f"\033[92m{' '.join(map(str, message))}\033[0m")


def print_warning(*message: object) -> None:
    """Print a warning message."""
    print(f"\033[91m{' '.join(map(str, message))}\033[0m")


def print_info(*message: object) -> None:
    """Print an informational message."""
    print(f"\033[96m{' '.join(map(str, message))}\033[0m")


def print_white(*message: object) -> None:
    """Print a white terminal message."""
    print(f"\033[37m{' '.join(map(str, message))}\033[0m")
