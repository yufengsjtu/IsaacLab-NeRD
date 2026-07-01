# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the Neural simulation interfaces for IsaacLab core package."""

import os

try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib

# Conveniences to other module directories via relative paths
ISAACLAB_NEURAL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# Find config/extension.toml: bundled inside the package (wheel install) or in the
# parent directory (editable install).
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_toml_path = os.path.join(_pkg_dir, "config", "extension.toml")
if not os.path.isfile(_toml_path):
    _toml_path = os.path.join(ISAACLAB_NEURAL_EXT_DIR, "config", "extension.toml")

with open(_toml_path, "rb") as f:
    ISAACLAB_NEURAL_METADATA = tomllib.load(f)
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_NEURAL_METADATA["package"]["version"]
