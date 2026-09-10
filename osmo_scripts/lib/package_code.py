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
# Gitignored runtime assets required by G1 OSMO generate/train jobs.
# Runtime assets required by Franka OSMO generate/train jobs. Keep this glob even
# when the files are untracked so a later gitignore cannot drop them from the
# tarball.
FORCE_INCLUDED_GLOBS = (
    "pretrained/control_policy/g1_flat_terrain/**/*",
    "pretrained/control_policy/franka_reach/**/*",
)


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIRS_ANYWHERE:
        return True
    if path.parts and path.parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
        return True

    path_str = path.as_posix()
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_str, pattern) for pattern in EXCLUDED_GLOBS)


def _git_listed_files(project_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "-z", "--exclude-standard"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def force_included_files(project_root: Path) -> list[Path]:
    """Return required policy assets even when Git ignores them."""
    files: list[Path] = []
    for pattern in FORCE_INCLUDED_GLOBS:
        for path in sorted(project_root.glob(pattern)):
            if path.is_file():
                files.append(path.relative_to(project_root))
    return files


def collect_packaged_files(project_root: Path) -> list[Path]:
    """Return relative files included in the OSMO code archive."""
    files: list[Path] = []
    seen: set[Path] = set()
    for path in (*_git_listed_files(project_root), *force_included_files(project_root)):
        if path in seen:
            continue
        seen.add(path)
        if (project_root / path).is_file() and not _should_skip(path):
            files.append(path)
    return files


def package_code(project_root: Path, archive_path: Path):
    """Create a tarball from repository files plus required policy assets."""
    files = collect_packaged_files(project_root)
    force_included = {path.as_posix() for path in force_included_files(project_root)}
    force_included_count = sum(1 for path in files if path.as_posix() in force_included)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            archive.add(project_root / path, arcname=Path("IsaacLab-NeRD") / path, recursive=False)

    print(f"Created {archive_path} with {len(files)} file(s) ({force_included_count} force-included policy asset(s)).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--archive-path", required=True)
    args = parser.parse_args()

    package_code(project_root=Path(args.project_root), archive_path=Path(args.archive_path))


if __name__ == "__main__":
    main()
