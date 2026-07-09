#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download an NV-Datasets dataset to a local directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from lib.nvdataset_io import download_dataset


def _check_credentials() -> int:
    if not os.environ.get("NGC_API_KEY"):
        print("[FATAL] Set NGC_API_KEY.", file=sys.stderr)
        return 2
    if not os.environ.get("NVDATASET_TENANTID"):
        print("[FATAL] Set NVDATASET_TENANTID.", file=sys.stderr)
        return 2
    return 0


def _print_downloaded_files(output_dir: Path):
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(path.relative_to(output_dir).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="NV-Datasets dataset name to download.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory to store downloaded files.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Dataset-relative prefix to download, for example anymal-c-fixed-ground/Anymal-C.",
    )
    parser.add_argument(
        "--snapshot",
        default="",
        help="Dataset snapshot name. Currently unsupported by the shared SDK download path.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before downloading.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print downloaded file paths after caching completes.",
    )
    args = parser.parse_args()

    credential_status = _check_credentials()
    if credential_status:
        return credential_status

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_dataset(
            name=args.dataset,
            output_dir=args.output_dir,
            snapshot=args.snapshot,
            prefix=args.prefix,
        )
    except ImportError as exc:
        print("[FATAL] Install nvdataset first, or run start.sh once so it installs the package.", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"Downloaded NV-Datasets dataset {args.dataset!r} to {args.output_dir}.")
    if args.list:
        _print_downloaded_files(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
