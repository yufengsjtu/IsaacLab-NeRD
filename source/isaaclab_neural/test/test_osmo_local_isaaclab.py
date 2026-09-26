# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for overlaying the uploaded Isaac Lab tree on the OSMO image."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OSMO_SCRIPTS = REPO_ROOT / "osmo_scripts"


def _write_source_package(project_root: Path, name: str, marker: str) -> Path:
    package_root = project_root / "source" / name
    module_dir = package_root / name
    module_dir.mkdir(parents=True)
    (package_root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")
    (module_dir / "__init__.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    return package_root


@pytest.mark.parametrize("operation", ["_load_or_create_dataset", "_replace_dataset"])
def test_nvdataset_service_errors_do_not_trigger_writes(monkeypatch, operation):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import nvdataset_io

    def fail(**kwargs):
        raise RuntimeError("400 sdk_version_not_supported")

    client = SimpleNamespace(
        load_dataset=fail, create_and_load_dataset=lambda **kwargs: pytest.fail("unexpected write")
    )
    with pytest.raises(RuntimeError, match="sdk_version_not_supported"):
        getattr(nvdataset_io, operation)(client, "test")


def test_nvdataset_existing_code_requires_explicit_replace(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import nvdataset_io

    client = SimpleNamespace(
        load_dataset=lambda **kwargs: SimpleNamespace(id="existing"),
        force_delete_dataset=lambda **kwargs: pytest.fail("unexpected deletion"),
    )
    with pytest.raises(FileExistsError):
        nvdataset_io._replace_dataset(client, "existing")


def test_nvdataset_exact_missing_name_can_be_created(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import nvdataset_io

    def missing(**kwargs):
        raise ValueError("Did not find dataset", kwargs["name"])

    client = SimpleNamespace(load_dataset=missing, create_and_load_dataset=lambda **kwargs: kwargs["name"])
    assert nvdataset_io._load_or_create_dataset(client, "new") == "new"


def test_entrypoint_uses_preinstalled_nvdataset_without_online_pip():
    text = (OSMO_SCRIPTS / "entrypoint.sh").read_text()
    install = text.split("install_storage_deps()", 1)[1].split("upload_storage_output()", 1)[0]
    nvdataset = install.split("nvdataset)", 1)[1].split(";;", 1)[0]
    assert "/opt/nvdataset-venv/bin/python" in nvdataset
    assert "pip install" not in nvdataset


def test_nvdataset_sdk_version_is_checked_without_service_calls(monkeypatch):
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import nvdataset_io

    monkeypatch.setattr(nvdataset_io, "version", lambda name: "0.93.0")
    with pytest.raises(RuntimeError, match="0.95.0"):
        nvdataset_io._check_sdk_version()
    monkeypatch.setattr(nvdataset_io, "version", lambda name: "0.95.0")
    assert nvdataset_io._check_sdk_version() == "0.95.0"


def test_nvdataset_delegate_isolates_simulator_python_paths(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import nvdataset_io

    interpreter = tmp_path / "bin/python"
    interpreter.parent.mkdir()
    interpreter.write_text("test-interpreter")
    monkeypatch.setenv("NVDATASET_PYTHON", str(interpreter))
    monkeypatch.setenv("PYTHONPATH", "/simulator/python")
    monkeypatch.setenv("PYTHONHOME", "/simulator/home")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/simulator/libs")
    calls = []
    monkeypatch.setattr(nvdataset_io.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert nvdataset_io._delegate(["check"])
    command, kwargs = calls[0]
    assert command[0] == str(interpreter) and command[-1] == "check"
    assert not {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"} & kwargs["env"].keys()
    assert kwargs["check"]


def test_core_submodule_order_matches_install_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib.local_isaaclab import CORE_ISAACLAB_SUBMODULES as osmo_core

    from isaaclab.cli.commands.install import CORE_ISAACLAB_SUBMODULES

    assert osmo_core == list(CORE_ISAACLAB_SUBMODULES)


def test_source_packages_installs_core_then_neural(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib.local_isaaclab import source_packages

    names = [path.name for path in source_packages(REPO_ROOT)]
    assert names[0] == "isaaclab"
    assert "isaaclab_neural" in names
    assert names.index("isaaclab") < names.index("isaaclab_neural")
    assert (REPO_ROOT / "source" / "isaaclab" / "pyproject.toml").is_file()


def test_init_sh_bootstraps_uploaded_isaaclab_tree() -> None:
    text = (OSMO_SCRIPTS / "init.sh").read_text(encoding="utf-8")
    assert "local_isaaclab.py" in text
    assert "bootstrap" in text
    assert "source/isaaclab_neural" not in text


def test_cli_export_pythonpath_accepts_project_root_after_command() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(OSMO_SCRIPTS / "lib" / "local_isaaclab.py"),
            "export-pythonpath",
            "--project-root",
            str(REPO_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    entries = result.stdout.strip().split(os.pathsep)
    assert str((REPO_ROOT / "source" / "isaaclab").resolve()) == entries[0]
    assert str((REPO_ROOT / "source" / "isaaclab_neural").resolve()) in entries


def test_entrypoint_reexports_uploaded_isaaclab_pythonpath() -> None:
    text = (OSMO_SCRIPTS / "entrypoint.sh").read_text(encoding="utf-8")
    assert "export-pythonpath" in text
    assert "local_isaaclab.py verify" in text
    assert "export PYTHONPATH=" in text


def test_entrypoint_prefers_image_venv_over_kit_python() -> None:
    text = (OSMO_SCRIPTS / "entrypoint.sh").read_text(encoding="utf-8")
    venv_offset = text.index("/opt/isaaclab-venv")
    fallback_offset = text.index("Using Isaac Sim python.sh")
    assert "export VIRTUAL_ENV=" in text
    assert venv_offset < fallback_offset
    assert "return" in text[venv_offset:fallback_offset]


def test_init_sh_fails_fast_without_venv_deps() -> None:
    text = (OSMO_SCRIPTS / "init.sh").read_text(encoding="utf-8")
    bootstrap_offset = text.index("bootstrap")
    lazy_offset = text.index("import lazy_loader")
    assert lazy_offset > bootstrap_offset


def test_install_local_packages_force_reinstalls_without_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib.local_isaaclab import install_local_packages

    _write_source_package(tmp_path, "isaaclab", "uploaded")
    _write_source_package(tmp_path, "isaaclab_neural", "uploaded")
    calls: list[tuple[list[str], bool]] = []

    def fake_run(cmd, check=False):
        calls.append((list(cmd), bool(check)))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("lib.local_isaaclab.subprocess.run", fake_run)
    packages = install_local_packages(tmp_path)

    assert [path.name for path in packages] == ["isaaclab", "isaaclab_neural"]
    uninstall_cmd, uninstall_check = calls[0]
    assert uninstall_cmd[3] == "uninstall"
    assert uninstall_cmd[-2:] == ["isaaclab", "isaaclab_neural"]
    assert uninstall_check is False
    assert len(calls) == 2
    install_cmd, install_check = calls[1]
    assert "--no-deps" in install_cmd
    assert "--force-reinstall" in install_cmd
    assert install_cmd.count("--editable") == 2
    assert install_check is True


def test_verify_local_imports_rejects_image_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import local_isaaclab

    _write_source_package(tmp_path, "isaaclab", "uploaded")
    image_origin = tmp_path / "image" / "isaaclab" / "__init__.py"
    monkeypatch.setattr(local_isaaclab, "apply_sys_path", lambda _root: [])
    monkeypatch.setattr(local_isaaclab, "_module_origin", lambda _name: image_origin)
    with pytest.raises(RuntimeError, match="container image"):
        local_isaaclab.verify_local_imports(tmp_path)


def test_sys_path_hook_shadows_image_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib.local_isaaclab import write_sys_path_hook

    project_root = tmp_path / "repo"
    _write_source_package(project_root, "isaaclab", "uploaded")
    image_site = tmp_path / "image_site"
    image_pkg = image_site / "isaaclab"
    image_pkg.mkdir(parents=True)
    (image_pkg / "__init__.py").write_text("MARKER = 'image'\n", encoding="utf-8")
    hook_site = tmp_path / "hook_site"
    write_sys_path_hook(project_root, site_packages=hook_site)

    probe = r"""
import importlib.util
import site
import sys
site.addsitedir(sys.argv[1])
spec = importlib.util.find_spec("isaaclab")
print(spec.origin)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(hook_site)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(image_site),
            "PYTHONNOUSERSITE": "1",
        },
    )
    origin = Path(result.stdout.strip()).resolve()
    assert origin.is_relative_to((project_root / "source" / "isaaclab").resolve())
    assert "image_site" not in origin.as_posix()


def test_package_code_keeps_robot_assets_but_excludes_local_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(OSMO_SCRIPTS))
    from lib import package_code

    assets = [
        Path("pretrained/control_policy/g1_flat_terrain/model.pt"),
        Path("pretrained/control_policy/franka_reach/model.pt"),
    ]
    source = Path("source/isaaclab_neural/new_module.py")
    artifacts = [Path("osmo_outputs/run/output.log"), Path(".worktree/other/source/test.py"), Path("imgui.ini")]
    for path in [*assets, source, *artifacts]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("test", encoding="utf-8")
    monkeypatch.setattr(package_code, "_git_listed_files", lambda root: [source, *artifacts])

    assert set(package_code.collect_packaged_files(tmp_path)) == {source, *assets}
