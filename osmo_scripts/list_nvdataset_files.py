#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List files in an NV-Datasets dataset.

The script first tries SDK-side file listing methods. If the installed
``nvdataset`` package does not expose a listing API, pass ``--download-to`` to
cache the dataset locally and list the downloaded files.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import sys


def _iter_file_like_items(value) -> Iterable[object]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, dict):
        for key in ("files", "items", "data", "results"):
            if key in value:
                return _iter_file_like_items(value[key])
        return value.values()
    try:
        return iter(value)
    except TypeError:
        return (value,)


def _item_path(item: object) -> str:
    if isinstance(item, (str, bytes)):
        return item.decode() if isinstance(item, bytes) else item
    if isinstance(item, dict):
        for key in ("path", "name", "filename", "file_name", "key"):
            if key in item and item[key]:
                return str(item[key])
        datum = item.get("datum")
        if isinstance(datum, dict):
            for key in ("path", "name", "filename", "file_name", "key"):
                if key in datum and datum[key]:
                    return str(datum[key])
        if datum is not None:
            for attr in ("path", "name", "filename", "file_name", "key"):
                if hasattr(datum, attr):
                    value = getattr(datum, attr)
                    if value:
                        return str(value)
    for attr in ("path", "name", "filename", "file_name", "key"):
        if hasattr(item, attr):
            value = getattr(item, attr)
            if value:
                return str(value)
    datum = getattr(item, "datum", None)
    if datum is not None:
        for attr in ("path", "name", "filename", "file_name", "key"):
            if hasattr(datum, attr):
                value = getattr(datum, attr)
                if value:
                    return str(value)
    return repr(item)


def _item_deleted_at(item: object):
    if isinstance(item, dict):
        if item.get("deleted_at") is not None:
            return item["deleted_at"]
        datum = item.get("datum")
        if isinstance(datum, dict):
            return datum.get("deleted_at")
        if datum is not None and hasattr(datum, "deleted_at"):
            return getattr(datum, "deleted_at")
        return None
    if hasattr(item, "deleted_at"):
        return getattr(item, "deleted_at")
    datum = getattr(item, "datum", None)
    if datum is not None and hasattr(datum, "deleted_at"):
        return getattr(datum, "deleted_at")
    return None


def _try_remote_list(dataset, include_deleted: bool) -> list[str] | None:
    method_names = (
        "list_files",
        "list_file",
        "files",
        "get_files",
        "get_file_list",
        "list_objects",
        "objects",
    )
    for method_name in method_names:
        if not hasattr(dataset, method_name):
            continue
        method = getattr(dataset, method_name)
        try:
            value = method() if callable(method) else method
        except TypeError:
            continue
        except Exception as exc:
            print(f"[WARN] dataset.{method_name} failed: {exc}", file=sys.stderr)
            continue
        paths = sorted(
            _item_path(item)
            for item in _iter_file_like_items(value)
            if include_deleted or _item_deleted_at(item) is None
        )
        if paths:
            return paths
    return None


def _list_local_files(root: Path, prefix: str) -> list[str]:
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if prefix and not relative_path.startswith(prefix.rstrip("/") + "/") and relative_path != prefix.rstrip("/"):
            continue
        files.append(relative_path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="NV-Datasets dataset name.")
    parser.add_argument("--prefix", default="", help="Only show files under this dataset-relative prefix.")
    parser.add_argument("--include-deleted", action="store_true", help="Also show soft-deleted files returned by the SDK.")
    parser.add_argument(
        "--download-to",
        type=Path,
        default=None,
        help="Cache the dataset to this directory and list local files if remote listing is unavailable.",
    )
    args = parser.parse_args()

    try:
        import nvdataset
    except ImportError as exc:
        print("[FATAL] Install nvdataset first, or run start.sh once so it installs the package.", file=sys.stderr)
        raise SystemExit(2) from exc

    if not os.environ.get("NGC_API_KEY"):
        print("[FATAL] Set NGC_API_KEY.", file=sys.stderr)
        return 2
    if not os.environ.get("NVDATASET_TENANTID"):
        print("[FATAL] Set NVDATASET_TENANTID.", file=sys.stderr)
        return 2

    client = nvdataset.NVDatasetClient()
    dataset = client.load_dataset(name=args.dataset)

    paths = _try_remote_list(dataset, include_deleted=args.include_deleted)
    if paths is None:
        if args.download_to is None:
            print(
                "[FATAL] This nvdataset package did not expose a remote file listing API. "
                "Re-run with --download-to <dir> to cache and list files locally.",
                file=sys.stderr,
            )
            return 1
        args.download_to.mkdir(parents=True, exist_ok=True)
        dataset.cache_local(str(args.download_to))
        paths = _list_local_files(args.download_to, args.prefix)
    elif args.prefix:
        prefix = args.prefix.rstrip("/")
        paths = [path for path in paths if path == prefix or path.startswith(prefix + "/")]

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
