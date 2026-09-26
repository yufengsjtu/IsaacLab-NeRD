# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NV-Datasets helpers used by local submit scripts and OSMO jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


def _check_sdk_version() -> str:
    from packaging.version import Version

    installed = version("nvdataset")
    if Version(installed) < Version("0.95.0"):
        raise RuntimeError(
            f"NV-Datasets SDK {installed} is unsupported; "
            f"install a supported SDK (at least 0.95.0) in {sys.executable}."
        )
    return installed


def _delegate(arguments: list[str]) -> bool:
    """Run storage operations in the image's isolated SDK interpreter when configured."""
    interpreter = os.environ.get("NVDATASET_PYTHON")
    if not interpreter or os.path.abspath(interpreter) == os.path.abspath(sys.executable):
        return False
    if not Path(interpreter).is_file():
        raise FileNotFoundError(f"Configured NV-Datasets interpreter is unavailable: {interpreter}")
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment["VIRTUAL_ENV"] = str(Path(interpreter).parent.parent)
    environment["PATH"] = str(Path(interpreter).parent) + os.pathsep + environment.get("PATH", "")
    subprocess.run([interpreter, str(Path(__file__).resolve()), *arguments], check=True, env=environment)
    return True


def _load_or_create_dataset(client, name: str, description: str = ""):
    try:
        return client.load_dataset(name=name)
    except ValueError as exc:
        if exc.args != ("Did not find dataset", name):
            raise
        print(f"Dataset {name!r} was not found; creating it.")
        return client.create_and_load_dataset(name=name, description=description)


def _replace_dataset(client, name: str, description: str = "", *, allow_replace: bool = False):
    try:
        dataset = client.load_dataset(name=name)
    except ValueError as exc:
        if exc.args != ("Did not find dataset", name):
            raise
        print(f"Dataset {name!r} was not found; creating it.")
        return client.create_and_load_dataset(name=name, description=description)

    if not allow_replace:
        raise FileExistsError(
            f"Dataset {name!r} already exists. Use a new code dataset name; deletion requires explicit --allow-replace."
        )
    dataset_id = dataset.id
    print(f"Dataset {name!r} exists; deleting it so this upload replaces its contents.")
    if hasattr(client, "force_delete_dataset"):
        client.force_delete_dataset(dataset_id=dataset_id)
    else:
        client.delete_dataset(dataset_id=dataset_id)
    return client.create_and_load_dataset(name=name, description=description)


def _cache_dataset(dataset, output_dir: Path, prefix: str = ""):
    """Cache a dataset locally without removing unrelated output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.strip("/")

    if not prefix:
        dataset.cache_local(str(output_dir))
        return

    prefix_path = Path(prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise ValueError(f"Dataset prefix must be a safe relative path, got {prefix!r}.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nvdataset-", dir=output_dir.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prefix_cached = _cache_dataset_prefix(dataset, temp_dir, prefix)
        move_staged_path = True
        if prefix_cached:
            staged_path = temp_dir / prefix
            if not staged_path.exists():
                # Prefix-aware SDKs may place the selected contents directly
                # in the requested output directory.
                staged_path = temp_dir
                move_staged_path = False
        else:
            print(
                f"NV-Datasets SDK does not support prefix downloads; "
                f"caching the full dataset temporarily before selecting {prefix!r}."
            )
            dataset.cache_local(str(temp_dir))
            staged_path = temp_dir / prefix

        if not staged_path.exists():
            raise FileNotFoundError(f"Dataset prefix {prefix!r} was not found in the downloaded dataset.")
        _merge_cached_path(staged_path, output_dir / prefix, move_when_new=move_staged_path)


def _cache_dataset_prefix(dataset, temp_dir: Path, prefix: str) -> bool:
    """Try prefix-aware SDK signatures, returning whether one is supported."""
    try:
        import nvdataset

        filters = [
            nvdataset.types.Filter(
                op=nvdataset.types.FilterOperator.STARTS_WITH,
                field=nvdataset.types.Field(name="key", value=prefix),
            )
        ]
        dataset.cache_local(str(temp_dir), filters=filters)
        return True
    except (AttributeError, TypeError):
        pass

    for kwargs in ({"prefix": prefix}, {"path": prefix}, {"dataset_path": prefix}):
        try:
            dataset.cache_local(str(temp_dir), **kwargs)
            return True
        except TypeError:
            continue
    for kwargs in (
        {"prefix": prefix, "output_dir": str(temp_dir)},
        {"path": prefix, "output_dir": str(temp_dir)},
        {"dataset_path": prefix, "output_dir": str(temp_dir)},
    ):
        try:
            dataset.cache_local(**kwargs)
            return True
        except TypeError:
            continue
    return False


def _merge_cached_path(source: Path, destination: Path, *, move_when_new: bool = True) -> None:
    """Merge one cached file or directory into the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            if not destination.is_dir():
                raise FileExistsError(f"Cannot merge directory into existing file: {destination}")
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif move_when_new:
            shutil.move(str(source), str(destination))
        else:
            shutil.copytree(source, destination)
    else:
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(f"Cannot replace existing directory with file: {destination}")
        shutil.copy2(source, destination)


def download_dataset(name: str, output_dir: Path, snapshot: str = "", prefix: str = ""):
    """Download an NV-Datasets dataset to a local directory."""
    if snapshot:
        raise RuntimeError(
            "Snapshot-specific NV-Datasets download is not supported through the Python SDK path yet. "
            f"Requested dataset={name!r}, snapshot={snapshot!r}."
        )

    if _delegate(
        ["download", "--dataset", name, "--output-dir", str(output_dir), "--snapshot", snapshot, "--prefix", prefix]
    ):
        return
    _check_sdk_version()
    import nvdataset

    client = nvdataset.NVDatasetClient()
    dataset = client.load_dataset(name=name)
    _cache_dataset(dataset, output_dir, prefix)


def upload_directory(name: str, source_dir: Path, description: str = ""):
    """Upload a directory into an existing or newly created NV-Datasets dataset."""
    if _delegate(
        ["upload-directory", "--dataset", name, "--source-dir", str(source_dir), "--description", description]
    ):
        return
    _check_sdk_version()
    import nvdataset

    client = nvdataset.NVDatasetClient()
    dataset = _load_or_create_dataset(client, name, description)
    dataset.upload_directory(source_dir=str(source_dir))


def replace_dataset_with_files(name: str, files: list[Path], description: str = "", *, allow_replace: bool = False):
    """Upload a new code dataset; replacing existing contents requires explicit authorization."""
    arguments = ["replace-files", "--dataset", name, "--description", description]
    if allow_replace:
        arguments.append("--allow-replace")
    if _delegate([*arguments, "--", *map(str, files)]):
        return
    _check_sdk_version()
    import nvdataset

    client = nvdataset.NVDatasetClient()
    dataset = _replace_dataset(client, name, description, allow_replace=allow_replace)
    dataset.upload_files(files=[str(path) for path in files])


def _cmd_check(_args: argparse.Namespace):
    if _delegate(["check"]):
        return
    installed = _check_sdk_version()
    import _cffi_backend  # noqa: F401
    import dateutil.parser  # noqa: F401
    import nvdataset.cli.cli  # noqa: F401

    print(f"Validated nvdataset {installed} dependencies in {sys.executable}.")


def _cmd_download(args: argparse.Namespace):
    download_dataset(
        name=args.dataset,
        output_dir=Path(args.output_dir),
        snapshot=args.snapshot,
        prefix=args.prefix,
    )


def _cmd_upload_directory(args: argparse.Namespace):
    upload_directory(
        name=args.dataset,
        source_dir=Path(args.source_dir),
        description=args.description,
    )


def _cmd_replace_files(args: argparse.Namespace):
    replace_dataset_with_files(
        name=args.dataset,
        files=[Path(path) for path in args.files],
        description=args.description,
        allow_replace=args.allow_replace,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate imports used by nvdataset helpers.")
    check_parser.set_defaults(func=_cmd_check)

    download_parser = subparsers.add_parser("download", help="Download an NV-Datasets dataset.")
    download_parser.add_argument("--dataset", required=True)
    download_parser.add_argument("--output-dir", required=True)
    download_parser.add_argument("--snapshot", default="")
    download_parser.add_argument("--prefix", default="")
    download_parser.set_defaults(func=_cmd_download)

    upload_parser = subparsers.add_parser("upload-directory", help="Upload a directory to a dataset.")
    upload_parser.add_argument("--dataset", required=True)
    upload_parser.add_argument("--source-dir", required=True)
    upload_parser.add_argument("--description", default="")
    upload_parser.set_defaults(func=_cmd_upload_directory)

    replace_parser = subparsers.add_parser("replace-files", help="Replace a dataset with files.")
    replace_parser.add_argument("--dataset", required=True)
    replace_parser.add_argument("--description", default="")
    replace_parser.add_argument(
        "--allow-replace", action="store_true", help="Explicitly allow deleting an existing dataset before upload."
    )
    replace_parser.add_argument("files", nargs="+")
    replace_parser.set_defaults(func=_cmd_replace_files)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
