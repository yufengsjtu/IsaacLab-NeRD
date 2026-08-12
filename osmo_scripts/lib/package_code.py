#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package the current IsaacLab-NeRD source tree for OSMO."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import tarfile
from pathlib import Path

EXCLUDED_DIRS_ANYWHERE = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
EXCLUDED_TOP_LEVEL_DIRS = {
    "data",
    "logs",
    "outputs",
}
EXCLUDED_GLOBS = (
    ".env",
    ".env.*",
    "*credential*.json",
    "*credentials*.json",
    "*credential*.sh",
    "*credentials*.sh",
)


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIRS_ANYWHERE:
        return True
    if path.parts and path.parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
        return True

    path_str = path.as_posix()
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_str, pattern) for pattern in EXCLUDED_GLOBS)


def package_code(project_root: Path, archive_path: Path):
    """Create a tarball from tracked and untracked non-ignored files."""
    result = subprocess.run(
        ["git", "ls-files", "-co", "-z", "--exclude-standard"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    relative_paths = [Path(item.decode()) for item in result.stdout.split(b"\0") if item]
    files = [path for path in relative_paths if (project_root / path).is_file() and not _should_skip(path)]

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            archive.add(project_root / path, arcname=Path("IsaacLab-NeRD") / path, recursive=False)

    print(f"Created {archive_path} with {len(files)} file(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--archive-path", required=True)
    args = parser.parse_args()

    package_code(project_root=Path(args.project_root), archive_path=Path(args.archive_path))


if __name__ == "__main__":
    main()
