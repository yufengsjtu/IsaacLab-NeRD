#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Install the uploaded Isaac Lab tree so OSMO jobs do not import the image copy.

The OSMO image already contains an Isaac Lab distribution. Workflows previously
installed only ``isaaclab_neural``, so ``import isaaclab`` still resolved to the
image. This helper editable-installs every uploaded ``source/isaaclab*`` package
with ``--no-deps`` and writes a ``site-packages`` hook that prepends those
sources on ``sys.path`` for later interpreters, including the image uv venv
and Isaac Sim's ``python.sh`` fallback.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

# Keep in sync with ``CORE_ISAACLAB_SUBMODULES`` in
# ``isaaclab.cli.commands.install``. Duplicated here so bootstrap can run before
# the uploaded ``isaaclab`` package is importable.
CORE_ISAACLAB_SUBMODULES: list[str] = [
    "isaaclab",
    "isaaclab_ppisp",
    "isaaclab_contrib",
    "isaaclab_newton",
    "isaaclab_physx",
    "isaaclab_assets",
    "isaaclab_experimental",
    "isaaclab_ov",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_rl",
    "isaaclab_visualizers",
]
HOOK_MODULE_NAME = "_isaaclab_osmo_syspath"
HOOK_PTH_NAME = "zzz_isaaclab_osmo.pth"


def default_project_root() -> Path:
    """Return the OSMO checkout path from the environment or the conventional location."""
    configured = os.environ.get("PROJECT_ROOT", "")
    if configured:
        return Path(configured)
    return Path.home() / "code" / "IsaacLab-NeRD"


def _is_installable(path: Path) -> bool:
    return path.is_dir() and ((path / "pyproject.toml").is_file() or (path / "setup.py").is_file())


def source_packages(project_root: Path) -> list[Path]:
    """Return installable ``source/*`` packages in dependency order.

    Core Isaac Lab packages come first, then remaining trees such as
    ``isaaclab_neural`` and optional ``isaaclab_mimic`` / ``isaaclab_teleop``.
    """
    source_dir = project_root / "source"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Isaac Lab source directory not found at {source_dir}")

    ordered: list[Path] = []
    seen: set[str] = set()
    for name in CORE_ISAACLAB_SUBMODULES:
        path = source_dir / name
        if _is_installable(path):
            ordered.append(path)
            seen.add(name)
    extras = sorted(
        path for path in source_dir.iterdir() if path.name not in seen and _is_installable(path)
    )
    ordered.extend(extras)
    if not ordered:
        raise RuntimeError(f"No installable Isaac Lab packages found under {source_dir}")
    return ordered


def pythonpath_entries(project_root: Path) -> list[str]:
    """Return absolute source directories that must precede the image packages."""
    return [str(path.resolve()) for path in source_packages(project_root)]


def export_pythonpath(project_root: Path) -> str:
    """Return a ``PYTHONPATH`` value that prefers the uploaded Isaac Lab tree."""
    return os.pathsep.join(pythonpath_entries(project_root))


def apply_sys_path(project_root: Path) -> list[str]:
    """Prepend uploaded sources on ``sys.path`` and ``PYTHONPATH`` in this process."""
    local_paths = pythonpath_entries(project_root)
    remainder = [path for path in sys.path if path not in local_paths]
    sys.path[:] = [*local_paths, *remainder]
    existing = [
        path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path and path not in local_paths
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join([*local_paths, *existing])
    return local_paths


def _writable_site_packages() -> Path:
    """Return a site-packages directory that this interpreter can write."""
    candidates: list[Path] = []
    try:
        candidates.extend(Path(item) for item in site.getsitepackages())
    except Exception:
        pass
    purelib = sysconfig.get_path("purelib")
    if purelib:
        candidates.append(Path(purelib))
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            probe = resolved / ".isaaclab_osmo_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return resolved
        except OSError:
            continue
    raise RuntimeError("No writable site-packages directory for the local Isaac Lab sys.path hook.")


def write_sys_path_hook(project_root: Path, site_packages: Path | None = None) -> Path:
    """Write a ``.pth`` hook that prepends uploaded Isaac Lab sources on startup.

    Args:
        project_root: Uploaded Isaac Lab repository root.
        site_packages: Directory that receives the hook. When omitted, the first
            writable interpreter site-packages path is used.

    Returns:
        Path to the written ``.pth`` file.
    """
    destination = site_packages.resolve() if site_packages is not None else _writable_site_packages()
    destination.mkdir(parents=True, exist_ok=True)
    local_paths = pythonpath_entries(project_root)
    rendered_paths = ",\n".join(f"    {path!r}" for path in local_paths)
    module_path = destination / f"{HOOK_MODULE_NAME}.py"
    module_path.write_text(
        "# Auto-generated by osmo_scripts/lib/local_isaaclab.py. Do not edit.\n"
        "from __future__ import annotations\n\n"
        "import os\n"
        "import sys\n\n"
        f"LOCAL_SOURCE_PATHS = [\n{rendered_paths},\n]\n\n\n"
        "def apply() -> None:\n"
        '    """Prepend uploaded Isaac Lab sources so they shadow the container image."""\n'
        "    remainder = [path for path in sys.path if path not in LOCAL_SOURCE_PATHS]\n"
        "    sys.path[:] = [*LOCAL_SOURCE_PATHS, *remainder]\n"
        "    existing = [\n"
        "        path\n"
        '        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)\n'
        "        if path and path not in LOCAL_SOURCE_PATHS\n"
        "    ]\n"
        '    os.environ["PYTHONPATH"] = os.pathsep.join([*LOCAL_SOURCE_PATHS, *existing])\n\n\n'
        "apply()\n",
        encoding="utf-8",
    )
    pth_path = destination / HOOK_PTH_NAME
    pth_path.write_text(f"import {HOOK_MODULE_NAME}\n", encoding="utf-8")
    print(f"Wrote Isaac Lab sys.path hook at {pth_path}", flush=True)
    return pth_path


def install_local_packages(project_root: Path) -> list[Path]:
    """Editable-install uploaded Isaac Lab packages without pulling image deps."""
    packages = source_packages(project_root)
    names = [path.name for path in packages]
    print(f"Removing image Isaac Lab distributions when present: {', '.join(names)}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *names],
        check=False,
    )
    for path in packages:
        print(f"Installing uploaded editable package {path.name} from {path}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            *[arg for path in packages for arg in ("--editable", str(path))],
        ],
        check=True,
    )
    return packages


def _module_origin(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin and spec.origin not in {"frozen", "namespace", "built-in"}:
        return Path(spec.origin).resolve()
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0]).resolve()
    return None


def verify_local_imports(project_root: Path) -> None:
    """Fail if any uploaded package still resolves to the container image."""
    apply_sys_path(project_root)
    failures: list[str] = []
    for path in source_packages(project_root):
        origin = _module_origin(path.name)
        if origin is None:
            failures.append(f"{path.name}: not importable after installing the uploaded tree")
            continue
        package_root = path.resolve()
        if origin != package_root and package_root not in origin.parents:
            failures.append(f"{path.name}: {origin} is not under {package_root}")
        else:
            print(f"{path.name} -> {origin}", flush=True)
    if failures:
        raise RuntimeError(
            "OSMO would import Isaac Lab packages from the container image instead of the"
            " uploaded repository:\n" + "\n".join(failures)
        )


def bootstrap(project_root: Path) -> None:
    """Install the uploaded tree, install the sys.path hook, and verify imports."""
    install_local_packages(project_root)
    write_sys_path_hook(project_root)
    env = os.environ.copy()
    uploaded_pythonpath = export_pythonpath(project_root)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        uploaded_pythonpath if not existing else os.pathsep.join((uploaded_pythonpath, existing))
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "verify",
            "--project-root",
            str(project_root),
        ],
        check=True,
        env=env,
    )
    print("Uploaded Isaac Lab tree is on sys.path ahead of the container image.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("bootstrap", "install", "hook", "verify", "export-pythonpath"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--project-root", type=Path, default=None)

    args = parser.parse_args()
    project_root = (args.project_root or default_project_root()).resolve()
    if args.command == "bootstrap":
        bootstrap(project_root)
    elif args.command == "install":
        install_local_packages(project_root)
    elif args.command == "hook":
        write_sys_path_hook(project_root)
    elif args.command == "verify":
        verify_local_imports(project_root)
    elif args.command == "export-pythonpath":
        print(export_pythonpath(project_root))
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
