#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenStack Swift helpers used by local submit scripts and OSMO jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

DEFAULT_SEGMENT_SIZE_BYTES = 1024**3


class _SegmentReader:
    """Limit reads from a shared stream to one retryable Swift segment."""

    def __init__(self, stream, size: int):
        self._stream = stream
        self._start = stream.tell()
        self._size = size
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._size - self._position
        if remaining == 0:
            return b""
        if size < 0 or size > remaining:
            size = remaining
        data = self._stream.read(size)
        self._position += len(data)
        return data

    def tell(self) -> int:
        """Return the current position relative to this segment."""
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Seek within this segment so swiftclient can retry a failed PUT."""
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported whence value: {whence}.")
        if position < 0 or position > self._size:
            raise ValueError(f"Cannot seek to {position} in a {self._size}-byte segment.")
        self._stream.seek(self._start + position)
        self._position = position
        return position


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Set {name} before using the Swift storage backend.")
    return value


def _auth_url_candidates(auth_url: str) -> list[str]:
    """Return likely legacy Swift authentication endpoints."""
    normalized = auth_url.rstrip("/")
    candidates = [normalized]
    if normalized.rsplit("/", maxsplit=1)[-1] not in {"v1.0", "v2.0", "v3"}:
        candidates.extend((f"{normalized}/auth/v1.0", f"{normalized}/v1.0"))
    return list(dict.fromkeys(candidates))


def connect():
    """Connect to Swift using credentials from environment variables."""
    from swiftclient.client import Connection

    auth_url = _required_env("SWIFT_AUTH_URL")
    user = _required_env("SWIFT_USER")
    key = _required_env("SWIFT_AUTH_KEY")
    auth_version = os.environ.get("SWIFT_AUTH_VERSION", "1")
    failures = []
    for candidate in _auth_url_candidates(auth_url):
        try:
            connection = Connection(
                authurl=candidate,
                user=user,
                key=key,
                auth_version=auth_version,
                retries=1,
            )
            connection.head_account()
            return connection, candidate
        except Exception as exc:
            status = getattr(exc, "http_status", None)
            detail = type(exc).__name__
            if status is not None:
                detail += f" HTTP {status}"
            failures.append(f"{candidate}: {detail}")
    raise RuntimeError("Swift authentication failed for all candidate endpoints: " + "; ".join(failures))


def ensure_container(connection, container: str) -> None:
    """Create a container if it does not already exist."""
    try:
        connection.head_container(container)
    except Exception as exc:
        if getattr(exc, "http_status", None) != 404:
            raise
        connection.put_container(container)


def upload_file(container: str, source: Path, object_name: str) -> None:
    """Upload one file, replacing an object with the same name."""
    connection, _ = connect()
    ensure_container(connection, container)
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    object_name = _safe_object_name(object_name)
    _upload_object(connection, container, source, object_name)
    print(f"Uploaded {source} to Swift {container}/{object_name}.")


def download_object(container: str, object_name: str, destination: Path) -> None:
    """Download one object to a local file."""
    connection, _ = connect()
    object_name = _safe_object_name(object_name)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _, body = connection.get_object(container, object_name, resp_chunk_size=1024 * 1024)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            for chunk in body:
                stream.write(chunk)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    print(f"Downloaded Swift {container}/{object_name} to {destination}.")


def download_prefix(
    container: str,
    prefix: str,
    output_dir: Path,
    *,
    model_filename: str | None = None,
) -> None:
    """Download objects under a prefix while preserving object paths.

    Args:
        container: Swift container name.
        prefix: Object-name prefix to download.
        output_dir: Local destination directory.
        model_filename: When set, skip ``.pt`` files whose basename does not
            match this filename. Non-model files are always downloaded.
    """
    connection, _ = connect()
    prefix = _safe_prefix(prefix)
    _, objects = connection.get_container(container, prefix=prefix, full_listing=True)
    if not objects:
        raise FileNotFoundError(f"No Swift objects found under {container}/{prefix}.")

    if model_filename is not None and PurePosixPath(model_filename).name != model_filename:
        raise ValueError("model_filename must be a basename, not a path.")
    selected_objects = []
    for item in objects:
        object_name = _safe_object_name(str(item["name"]))
        if object_name.endswith("/"):
            continue
        object_path = PurePosixPath(object_name)
        if model_filename is not None and object_path.suffix == ".pt" and object_path.name != model_filename:
            continue
        selected_objects.append(object_name)
    if model_filename is not None and not any(
        PurePosixPath(object_name).name == model_filename for object_name in selected_objects
    ):
        raise FileNotFoundError(f"No model named {model_filename!r} found under {container}/{prefix}.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, object_name in enumerate(selected_objects, start=1):
        print(f"Downloading {index}/{len(selected_objects)}: {object_name}", flush=True)
        destination = _safe_destination(output_dir, object_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _, body = connection.get_object(container, object_name, resp_chunk_size=1024 * 1024)
        with destination.open("wb") as stream:
            for chunk in body:
                stream.write(chunk)
    print(f"Downloaded {len(selected_objects)} Swift object(s) from {container}/{prefix} to {output_dir}.")


def upload_directory(container: str, source_dir: Path, prefix: str = "", *, resume: bool = False) -> None:
    """Upload all files in a directory under an optional object prefix."""
    connection, _ = connect()
    ensure_container(connection, container)
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    prefix = _safe_prefix(prefix)
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    for path in files:
        relative_name = path.relative_to(source_dir).as_posix()
        object_name = _safe_object_name(f"{prefix}/{relative_name}" if prefix else relative_name)
        _upload_object(connection, container, path, object_name, resume=resume)
    print(f"Uploaded {len(files)} file(s) from {source_dir} to Swift {container}/{prefix}.")


def _upload_object(connection, container: str, source: Path, object_name: str, *, resume: bool = False) -> None:
    size = source.stat().st_size
    if resume and _remote_object_has_size(connection, container, object_name, size):
        print(f"Skipping existing Swift object {container}/{object_name}.")
        return
    segment_size = int(os.environ.get("SWIFT_SEGMENT_SIZE_BYTES", DEFAULT_SEGMENT_SIZE_BYTES))
    if segment_size <= 0:
        raise ValueError("SWIFT_SEGMENT_SIZE_BYTES must be positive.")
    if size <= segment_size:
        with source.open("rb") as stream:
            connection.put_object(
                container,
                object_name,
                contents=stream,
                content_length=size,
            )
        return

    manifest = []
    segment_count = (size + segment_size - 1) // segment_size
    print(
        f"Uploading large Swift object {container}/{object_name} "
        f"as {segment_count} segment(s) of at most {segment_size} bytes."
    )
    with source.open("rb") as stream:
        for index in range(segment_count):
            current_size = min(segment_size, size - index * segment_size)
            segment_name = _safe_object_name(f".segments/{object_name}/{index:08d}")
            existing_etag = (
                _matching_segment_etag(
                    connection,
                    container,
                    segment_name,
                    source,
                    offset=index * segment_size,
                    size=current_size,
                )
                if resume
                else None
            )
            if existing_etag is not None:
                stream.seek(current_size, os.SEEK_CUR)
                manifest.append(
                    {
                        "path": f"/{container}/{segment_name}",
                        "etag": existing_etag,
                        "size_bytes": current_size,
                    }
                )
                print(f"Reusing segment {index + 1}/{segment_count} for {object_name}.")
                continue
            etag = connection.put_object(
                container,
                segment_name,
                contents=_SegmentReader(stream, current_size),
                content_length=current_size,
            )
            if not etag:
                headers = connection.head_object(container, segment_name)
                etag = headers["etag"]
            manifest.append(
                {
                    "path": f"/{container}/{segment_name}",
                    "etag": str(etag).strip('"'),
                    "size_bytes": current_size,
                }
            )
            print(f"Uploaded segment {index + 1}/{segment_count} for {object_name}.")

    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    connection.put_object(
        container,
        object_name,
        contents=manifest_bytes,
        content_length=len(manifest_bytes),
        content_type="application/json",
        query_string="multipart-manifest=put",
    )
    print(f"Created static large object manifest for {container}/{object_name}.")


def _remote_object_has_size(connection, container: str, object_name: str, size: int) -> bool:
    """Return whether an existing object has the expected assembled size."""
    try:
        headers = connection.head_object(container, object_name)
    except Exception as exc:
        if getattr(exc, "http_status", None) == 404:
            return False
        raise
    return int(headers.get("content-length", -1)) == size


def _matching_segment_etag(
    connection,
    container: str,
    segment_name: str,
    source: Path,
    *,
    offset: int,
    size: int,
) -> str | None:
    """Return the ETag when a remote segment exactly matches the local bytes."""
    try:
        headers = connection.head_object(container, segment_name)
    except Exception as exc:
        if getattr(exc, "http_status", None) == 404:
            return None
        raise
    if int(headers.get("content-length", -1)) != size:
        return None

    digest = hashlib.md5(usedforsecurity=False)
    remaining = size
    with source.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                return None
            digest.update(chunk)
            remaining -= len(chunk)
    remote_etag = str(headers.get("etag", "")).strip('"')
    if digest.hexdigest() != remote_etag:
        return None
    return remote_etag


def _safe_object_name(value: str) -> str:
    path = PurePosixPath(value.strip("/"))
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe Swift object name: {value!r}")
    return path.as_posix()


def _safe_prefix(value: str) -> str:
    if not value.strip("/"):
        return ""
    return _safe_object_name(value).rstrip("/")


def _safe_destination(root: Path, object_name: str) -> Path:
    destination = (root / Path(*PurePosixPath(object_name).parts)).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Swift object escapes output directory: {object_name!r}")
    return destination


def _cmd_check(_args: argparse.Namespace) -> None:
    connection, auth_url = connect()
    headers = connection.head_account()
    object_count = headers.get("x-account-object-count", "unknown")
    container_count = headers.get("x-account-container-count", "unknown")
    print(f"Swift authentication succeeded via {auth_url}; containers={container_count}, objects={object_count}.")


def _cmd_ensure_container(args: argparse.Namespace) -> None:
    connection, _ = connect()
    ensure_container(connection, args.container)
    print(f"Swift container {args.container!r} is available.")


def _cmd_upload_file(args: argparse.Namespace) -> None:
    upload_file(args.container, Path(args.source), args.object_name)


def _cmd_download_object(args: argparse.Namespace) -> None:
    download_object(args.container, args.object_name, Path(args.output))


def _cmd_download_prefix(args: argparse.Namespace) -> None:
    download_prefix(
        args.container,
        args.prefix,
        Path(args.output_dir),
        model_filename=args.model_filename,
    )


def _cmd_upload_directory(args: argparse.Namespace) -> None:
    upload_directory(args.container, Path(args.source_dir), args.prefix, resume=args.resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate Swift authentication.")
    check_parser.set_defaults(func=_cmd_check)

    ensure_parser = subparsers.add_parser("ensure-container", help="Create a container when absent.")
    ensure_parser.add_argument("--container", required=True)
    ensure_parser.set_defaults(func=_cmd_ensure_container)

    upload_file_parser = subparsers.add_parser("upload-file", help="Upload or replace one object.")
    upload_file_parser.add_argument("--container", required=True)
    upload_file_parser.add_argument("--source", required=True)
    upload_file_parser.add_argument("--object-name", required=True)
    upload_file_parser.set_defaults(func=_cmd_upload_file)

    download_object_parser = subparsers.add_parser("download-object", help="Download one object.")
    download_object_parser.add_argument("--container", required=True)
    download_object_parser.add_argument("--object-name", required=True)
    download_object_parser.add_argument("--output", required=True)
    download_object_parser.set_defaults(func=_cmd_download_object)

    download_prefix_parser = subparsers.add_parser("download-prefix", help="Download an object prefix.")
    download_prefix_parser.add_argument("--container", required=True)
    download_prefix_parser.add_argument("--prefix", required=True)
    download_prefix_parser.add_argument("--output-dir", required=True)
    download_prefix_parser.add_argument(
        "--model-filename",
        help="Download non-model files plus only the .pt model with this basename.",
    )
    download_prefix_parser.set_defaults(func=_cmd_download_prefix)

    upload_directory_parser = subparsers.add_parser("upload-directory", help="Upload a directory tree.")
    upload_directory_parser.add_argument("--container", required=True)
    upload_directory_parser.add_argument("--source-dir", required=True)
    upload_directory_parser.add_argument("--prefix", default="")
    upload_directory_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete objects and matching multipart segments.",
    )
    upload_directory_parser.set_defaults(func=_cmd_upload_directory)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
