#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delete an NV-Datasets dataset or selected files.

The script defaults to dry-run mode. Pass ``--yes`` to perform deletion.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable


def _check_credentials() -> int:
    if not os.environ.get("NGC_API_KEY"):
        print("[FATAL] Set NGC_API_KEY.", file=sys.stderr)
        return 2
    if not os.environ.get("NVDATASET_TENANTID"):
        print("[FATAL] Set NVDATASET_TENANTID.", file=sys.stderr)
        return 2
    return 0


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


def _item_key(item: object) -> str:
    if isinstance(item, bytes):
        return item.decode()
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("key", "path", "name", "filename", "file_name"):
            if key in item and item[key]:
                return str(item[key])
        datum = item.get("datum")
        if isinstance(datum, dict):
            for key in ("key", "path", "name", "filename", "file_name"):
                if key in datum and datum[key]:
                    return str(datum[key])
        if datum is not None:
            for attr in ("key", "path", "name", "filename", "file_name"):
                if hasattr(datum, attr):
                    value = getattr(datum, attr)
                    if value:
                        return str(value)
    for attr in ("key", "path", "name", "filename", "file_name"):
        if hasattr(item, attr):
            value = getattr(item, attr)
            if value:
                return str(value)
    datum = getattr(item, "datum", None)
    if datum is not None:
        for attr in ("key", "path", "name", "filename", "file_name"):
            if hasattr(datum, attr):
                value = getattr(datum, attr)
                if value:
                    return str(value)
    return repr(item)


def _item_datum_id(item: object):
    if isinstance(item, dict):
        datum = item.get("datum")
        if isinstance(datum, dict):
            return datum.get("id")
        if datum is not None and hasattr(datum, "id"):
            return getattr(datum, "id")
        return item.get("datum_id") or item.get("id")
    datum = getattr(item, "datum", None)
    if datum is not None and hasattr(datum, "id"):
        return getattr(datum, "id")
    return getattr(item, "datum_id", None) or getattr(item, "id", None)


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


def _list_remote_items(dataset) -> list[object]:
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
        items = list(_iter_file_like_items(value))
        if items:
            return items
    raise RuntimeError(
        "This nvdataset package did not expose a remote file listing API. Cannot safely delete by --prefix or --file."
    )


def _matches(key: str, prefixes: list[str], files: list[str], regexes: list[re.Pattern[str]]) -> bool:
    normalized_files = {file.strip("/") for file in files}
    if key.strip("/") in normalized_files:
        return True
    for prefix in prefixes:
        normalized = prefix.strip("/")
        if key == normalized or key.startswith(normalized + "/"):
            return True
    return any(regex.search(key) for regex in regexes)


def _try_call(obj, method_name: str, *args, **kwargs) -> bool:
    if not hasattr(obj, method_name):
        return False
    method = getattr(obj, method_name)
    try:
        method(*args, **kwargs)
        return True
    except TypeError:
        return False
    except Exception as exc:
        print(f"[WARN] {type(obj).__name__}.{method_name} failed: {exc}", file=sys.stderr)
        return False


def _delete_dataset(client, dataset, name: str, dry_run: bool) -> int:
    dataset_id = getattr(dataset, "id", None)
    print(f"Dataset: {name}")
    print(f"Dataset id: {dataset_id}")
    if dry_run:
        print("[dry-run] Would delete the entire dataset. Re-run with --yes to delete.")
        return 0

    if hasattr(client, "force_delete_dataset"):
        client.force_delete_dataset(dataset_id=dataset_id)
    else:
        client.delete_dataset(dataset_id=dataset_id)
    print("Deleted dataset.")
    return 0


def _delete_file(client, dataset, dataset_id, item) -> bool:
    key = _item_key(item)
    datum_id = _item_datum_id(item)

    attempts = (
        (dataset, "delete_file", (), {"key": key}),
        (dataset, "delete_file", (), {"path": key}),
        (dataset, "delete_file", (key,), {}),
        (dataset, "remove_file", (), {"key": key}),
        (dataset, "remove_file", (key,), {}),
        (dataset, "delete_files", (), {"keys": [key]}),
        (dataset, "delete_files", ([key],), {}),
        (dataset, "remove_files", (), {"keys": [key]}),
        (client, "delete_file", (), {"dataset_id": dataset_id, "key": key}),
        (client, "delete_file", (), {"dataset_id": dataset_id, "path": key}),
        (client, "delete_files", (), {"dataset_id": dataset_id, "keys": [key]}),
        (client, "remove_file", (), {"dataset_id": dataset_id, "key": key}),
    )
    for obj, method_name, args, kwargs in attempts:
        if _try_call(obj, method_name, *args, **kwargs):
            return True

    if datum_id is not None:
        datum_attempts = (
            (client, "force_delete_datum", (), {"datum_id": datum_id}),
            (client, "delete_datum", (), {"datum_id": datum_id}),
            (client, "delete_file", (), {"datum_id": datum_id}),
            (dataset, "delete_datum", (), {"datum_id": datum_id}),
        )
        for obj, method_name, args, kwargs in datum_attempts:
            if _try_call(obj, method_name, *args, **kwargs):
                return True

    if hasattr(item, "delete"):
        try:
            item.delete()
            return True
        except Exception as exc:
            print(f"[WARN] file item delete() failed for {key}: {exc}", file=sys.stderr)

    return False


def _delete_selected_files(
    client,
    dataset,
    name: str,
    prefixes: list[str],
    files: list[str],
    regexes: list[re.Pattern[str]],
    dry_run: bool,
) -> int:
    items = [item for item in _list_remote_items(dataset) if _item_deleted_at(item) is None]
    matched = sorted((item for item in items if _matches(_item_key(item), prefixes, files, regexes)), key=_item_key)

    print(f"Dataset: {name}")
    print(f"Matched {len(matched)} file(s).")
    for item in matched:
        print(_item_key(item))

    if not matched:
        return 0
    if dry_run:
        print("[dry-run] Would delete the matched files. Re-run with --yes to delete.")
        return 0

    dataset_id = getattr(dataset, "id", None)
    matched_keys = [_item_key(item) for item in matched]
    if hasattr(dataset, "delete_files"):
        try:
            dataset.delete_files(files=matched_keys, refresh_dataset=True)
            remaining_items = [item for item in _list_remote_items(dataset) if _item_deleted_at(item) is None]
            remaining_keys = {
                _item_key(item) for item in remaining_items if _matches(_item_key(item), prefixes, files, regexes)
            }
            if not remaining_keys:
                for key in matched_keys:
                    print(f"Deleted: {key}")
                return 0
            print("[WARN] dataset.delete_files returned, but these files are still listed:", file=sys.stderr)
            for key in sorted(remaining_keys):
                print(key, file=sys.stderr)
        except Exception as exc:
            print(f"[WARN] dataset.delete_files failed: {exc}", file=sys.stderr)

    failed = []
    for item in matched:
        key = _item_key(item)
        if _delete_file(client, dataset, dataset_id, item):
            print(f"Deleted: {key}")
        else:
            failed.append(key)

    if failed:
        print("[FATAL] Failed to delete these files with the installed nvdataset SDK:", file=sys.stderr)
        for key in failed:
            print(key, file=sys.stderr)
        print("Available client delete/remove methods:", file=sys.stderr)
        print([name for name in dir(client) if "delete" in name.lower() or "remove" in name.lower()], file=sys.stderr)
        print("Available dataset delete/remove methods:", file=sys.stderr)
        print([name for name in dir(dataset) if "delete" in name.lower() or "remove" in name.lower()], file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="NV-Datasets dataset name.")
    parser.add_argument("--delete-dataset", action="store_true", help="Delete the entire dataset.")
    parser.add_argument(
        "--prefix", action="append", default=[], help="Delete files under this dataset-relative prefix."
    )
    parser.add_argument("--file", action="append", default=[], help="Delete one exact dataset-relative file key.")
    parser.add_argument(
        "--regex", action="append", default=[], help="Delete files whose dataset-relative key matches this regex."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Actually delete. Without this flag, the script is dry-run only."
    )
    args = parser.parse_args()

    if args.delete_dataset and (args.prefix or args.file or args.regex):
        print("[FATAL] --delete-dataset cannot be combined with --prefix, --file, or --regex.", file=sys.stderr)
        return 2
    if not args.delete_dataset and not args.prefix and not args.file and not args.regex:
        print("[FATAL] Specify --delete-dataset, --prefix, --file, or --regex.", file=sys.stderr)
        return 2
    try:
        regexes = [re.compile(pattern) for pattern in args.regex]
    except re.error as exc:
        print(f"[FATAL] Invalid --regex: {exc}", file=sys.stderr)
        return 2

    credential_status = _check_credentials()
    if credential_status:
        return credential_status

    try:
        import nvdataset
    except ImportError as exc:
        print("[FATAL] Install nvdataset first, or run start.sh once so it installs the package.", file=sys.stderr)
        raise SystemExit(2) from exc

    client = nvdataset.NVDatasetClient()
    dataset = client.load_dataset(name=args.dataset)

    if args.delete_dataset:
        return _delete_dataset(client, dataset, args.dataset, dry_run=not args.yes)
    return _delete_selected_files(client, dataset, args.dataset, args.prefix, args.file, regexes, dry_run=not args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
