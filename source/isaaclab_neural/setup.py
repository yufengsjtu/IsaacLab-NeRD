# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_neural' python package."""

import os
import shutil

import toml
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Custom build command that bundles config/extension.toml into the package.

    This ensures the toml is available when installed as a regular (non-editable)
    wheel, e.g. when pulled in as a dependency via a file:// URL.
    """

    def run(self):
        super().run()
        src = os.path.join(EXTENSION_PATH, "config", "extension.toml")
        dst_dir = os.path.join(self.build_lib, "isaaclab_neural", "config")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, "extension.toml"))


# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    # INTENTIONALLY disabled to avoid circular dependency with isaaclab_physx, which also depends on isaaclab_newton.
    # This will be re-enabled once we move to UV and pyproject.toml-based packaging.
    # f"isaaclab_physx @ file://{os.path.join(os.path.dirname(EXTENSION_PATH), 'isaaclab_physx')}",
]

EXTRAS_REQUIRE = {
    "all": [
        "prettytable==3.3.0",
        "PyOpenGL-accelerate==3.1.10",
        "pyglet>=2.1.6,<3",
        "newton[sim]==1.2.1",
    ],
}

# Installation operation
setup(
    name="isaaclab_neural",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"": ["*.pyi"]},
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=[
        "isaaclab_neural",
        "isaaclab_neural.physics",
        "isaaclab_neural.solvers",
        "isaaclab_neural.contacts",
        "isaaclab_neural.data",
        "isaaclab_neural.envs",
        "isaaclab_neural.models",
        "isaaclab_neural.train",
        "isaaclab_neural.utils",
        "isaaclab_neural.eval",
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
    cmdclass={"build_py": build_py},
)
