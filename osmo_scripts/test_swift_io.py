# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from osmo_scripts.lib import swift_io


class SwiftError(Exception):
    def __init__(self, status: int):
        self.http_status = status


class FakeConnection:
    def __init__(self):
        self.containers: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}
        self.manifests: dict[tuple[str, str], list[dict]] = {}

    def head_container(self, container: str):
        if container not in self.containers:
            raise SwiftError(404)

    def put_container(self, container: str):
        self.containers.add(container)

    def put_object(
        self,
        container: str,
        object_name: str,
        *,
        contents,
        content_length: int,
        content_type: str | None = None,
        query_string: str | None = None,
    ):
        content = contents.read() if hasattr(contents, "read") else contents
        assert len(content) == content_length
        if query_string == "multipart-manifest=put":
            assert content_type == "application/json"
            self.manifests[(container, object_name)] = json.loads(content)
        self.objects[(container, object_name)] = content
        return hashlib.md5(content, usedforsecurity=False).hexdigest()

    def head_object(self, container: str, object_name: str):
        try:
            content = self.objects[(container, object_name)]
        except KeyError as exc:
            raise SwiftError(404) from exc
        if manifest := self.manifests.get((container, object_name)):
            content = b"".join(
                self.objects[tuple(segment["path"].strip("/").split("/", maxsplit=1))] for segment in manifest
            )
        return {
            "content-length": str(len(content)),
            "etag": hashlib.md5(content, usedforsecurity=False).hexdigest(),
        }

    def get_container(self, container: str, *, prefix: str, full_listing: bool):
        assert full_listing
        objects = [
            {"name": object_name}
            for stored_container, object_name in self.objects
            if stored_container == container and object_name.startswith(prefix)
        ]
        return {}, objects

    def get_object(self, container: str, object_name: str, *, resp_chunk_size: int, headers=None):
        assert resp_chunk_size > 0
        if manifest := self.manifests.get((container, object_name)):
            content = b"".join(
                self.objects[tuple(segment["path"].strip("/").split("/", maxsplit=1))] for segment in manifest
            )
        else:
            content = self.objects[(container, object_name)]
        response_headers = {}
        if headers and (range_header := headers.get("Range")):
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            response_headers["content-range"] = f"bytes {offset}-{len(content) - 1}/{len(content)}"
            content = content[offset:]
        return response_headers, iter((content,))


class TestSwiftIO(unittest.TestCase):
    def test_auth_url_candidates_and_legacy_probe(self):
        self.assertEqual(
            swift_io._auth_url_candidates("https://pdx.s8k.io/"),
            [
                "https://pdx.s8k.io",
                "https://pdx.s8k.io/auth/v1.0",
                "https://pdx.s8k.io/v1.0",
            ],
        )
        attempts = []

        class ProbeConnection:
            def __init__(self, *, authurl: str, **_kwargs):
                self.authurl = authurl
                attempts.append(authurl)

            def head_account(self):
                if not self.authurl.endswith("/auth/v1.0"):
                    raise SwiftError(401)
                return {}

        swiftclient_module = ModuleType("swiftclient")
        client_module = ModuleType("swiftclient.client")
        client_module.Connection = ProbeConnection
        environment = {
            "SWIFT_AUTH_URL": "https://pdx.s8k.io",
            "SWIFT_USER": "test-user",
            "SWIFT_AUTH_KEY": "test-key",
        }
        with (
            mock.patch.dict(sys.modules, {"swiftclient": swiftclient_module, "swiftclient.client": client_module}),
            mock.patch.dict(os.environ, environment, clear=False),
        ):
            _, auth_url = swift_io.connect()

        self.assertEqual(auth_url, "https://pdx.s8k.io/auth/v1.0")
        self.assertEqual(attempts, ["https://pdx.s8k.io", "https://pdx.s8k.io/auth/v1.0"])

    def test_directory_upload_and_prefix_download(self):
        connection = FakeConnection()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "dataset.hdf5").write_bytes(b"dataset")
            (source / "nested").mkdir()
            (source / "nested" / "metadata.json").write_bytes(b"metadata")

            with mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")):
                swift_io.upload_directory("datasets", source, "cache/env")
                destination = root / "download"
                swift_io.download_prefix("datasets", "cache", destination)

            self.assertEqual((destination / "cache/env/dataset.hdf5").read_bytes(), b"dataset")
            self.assertEqual((destination / "cache/env/nested/metadata.json").read_bytes(), b"metadata")

    def test_prefix_download_filters_model_files(self):
        connection = FakeConnection()
        connection.objects = {
            ("outputs", "run/final_model.pt"): b"final",
            ("outputs", "run/best_eval_model.pt"): b"best",
            ("outputs", "run/summary/events.out.tfevents.test"): b"events",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir)
            with mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")):
                swift_io.download_prefix("outputs", "run", destination, model_filename="final_model.pt")

            self.assertEqual((destination / "run/final_model.pt").read_bytes(), b"final")
            self.assertFalse((destination / "run/best_eval_model.pt").exists())
            self.assertEqual((destination / "run/summary/events.out.tfevents.test").read_bytes(), b"events")

    def test_large_object_segments_download_and_resume(self):
        connection = FakeConnection()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset.hdf5"
            source.write_bytes(b"0123456789")
            environment = {"SWIFT_SEGMENT_SIZE_BYTES": "4"}
            with (
                mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")),
                mock.patch.dict(os.environ, environment, clear=False),
            ):
                swift_io.upload_file("datasets", source, "cache/dataset.hdf5")
                swift_io.upload_directory("datasets", root, "cache", resume=True)
                destination = root / "downloaded.hdf5"
                swift_io.download_object("datasets", "cache/dataset.hdf5", destination)

            self.assertEqual(len(connection.manifests[("datasets", "cache/dataset.hdf5")]), 3)
            self.assertEqual(
                connection.objects[("datasets", ".segments/cache/dataset.hdf5/00000000")],
                b"0123",
            )
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_large_object_manifest_payload_is_retryable(self):
        class RetryManifestConnection(FakeConnection):
            def __init__(self):
                super().__init__()
                self.retry_payload: bytes | None = None

            def put_object(self, container: str, object_name: str, **kwargs):
                contents = kwargs["contents"]
                if kwargs.get("query_string") == "multipart-manifest=put":
                    first_payload = contents.read()
                    contents.seek(0)
                    self.retry_payload = contents.read()
                    contents.seek(0)
                    assert self.retry_payload == first_payload
                return super().put_object(container, object_name, **kwargs)

        connection = RetryManifestConnection()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "dataset.hdf5"
            source.write_bytes(b"0123456789")
            with (
                mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")),
                mock.patch.dict(os.environ, {"SWIFT_SEGMENT_SIZE_BYTES": "4"}, clear=False),
            ):
                swift_io.upload_file("datasets", source, "cache/dataset.hdf5")

        self.assertIsNotNone(connection.retry_payload)

    def test_upload_resume_replaces_same_size_corrupt_object(self):
        connection = FakeConnection()
        connection.objects[("datasets", "cache/dataset.hdf5")] = b"xxxx"
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "dataset.hdf5"
            source.write_bytes(b"data")
            with mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")):
                swift_io.upload_directory("datasets", Path(temp_dir), "cache", resume=True)

        self.assertEqual(connection.objects[("datasets", "cache/dataset.hdf5")], b"data")

    def test_download_retries_transient_head_failure(self):
        class FlakyHeadConnection(FakeConnection):
            def __init__(self):
                super().__init__()
                self.head_attempts = 0

            def head_object(self, container: str, object_name: str):
                self.head_attempts += 1
                if self.head_attempts == 1:
                    raise ConnectionError("transient HEAD failure")
                return super().head_object(container, object_name)

        connection = FlakyHeadConnection()
        connection.objects[("datasets", "cache/dataset.hdf5")] = b"dataset"
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "dataset.hdf5"
            with (
                mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")),
                mock.patch.object(swift_io.time, "sleep"),
            ):
                swift_io.download_object("datasets", "cache/dataset.hdf5", destination)

            self.assertEqual(connection.head_attempts, 2)
            self.assertEqual(destination.read_bytes(), b"dataset")

    def test_download_resumes_partial_object_for_up_to_five_attempts(self):
        class FlakyConnection(FakeConnection):
            def __init__(self):
                super().__init__()
                self.download_attempts = 0
                self.requested_ranges = []

            def get_object(self, container: str, object_name: str, *, resp_chunk_size: int, headers=None):
                self.download_attempts += 1
                self.requested_ranges.append(headers.get("Range") if headers else None)
                if self.download_attempts == 1:

                    def interrupted_body():
                        yield b"0123"
                        raise ConnectionError("connection interrupted")

                    return {}, interrupted_body()
                return super().get_object(
                    container,
                    object_name,
                    resp_chunk_size=resp_chunk_size,
                    headers=headers,
                )

        connection = FlakyConnection()
        connection.objects[("datasets", "cache/dataset.hdf5")] = b"0123456789"
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "dataset.hdf5"
            with (
                mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")),
                mock.patch.object(swift_io.time, "sleep"),
            ):
                swift_io.download_object("datasets", "cache/dataset.hdf5", destination)

            self.assertEqual(destination.read_bytes(), b"0123456789")
            self.assertEqual(connection.download_attempts, 2)
            self.assertEqual(connection.requested_ranges, [None, "bytes=4-"])
            self.assertFalse((destination.parent / ".dataset.hdf5.part").exists())

    def test_download_stops_after_five_failed_attempts(self):
        class FailingConnection(FakeConnection):
            def __init__(self):
                super().__init__()
                self.download_attempts = 0

            def get_object(self, container: str, object_name: str, *, resp_chunk_size: int, headers=None):
                self.download_attempts += 1

                def interrupted_body():
                    raise ConnectionError("connection interrupted")
                    yield

                return {}, interrupted_body()

        connection = FailingConnection()
        connection.objects[("datasets", "cache/dataset.hdf5")] = b"dataset"
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "dataset.hdf5"
            with (
                mock.patch.object(swift_io, "connect", return_value=(connection, "https://auth")),
                mock.patch.object(swift_io.time, "sleep"),
                self.assertRaisesRegex(RuntimeError, "after 5 attempts"),
            ):
                swift_io.download_object("datasets", "cache/dataset.hdf5", destination)

        self.assertEqual(connection.download_attempts, 5)

    def test_segment_reader_can_seek_within_segment(self):
        stream = io.BytesIO(b"prefix-segment-suffix")
        stream.seek(len(b"prefix-"))
        reader = swift_io._SegmentReader(stream, len(b"segment"))

        self.assertEqual(reader.read(3), b"seg")
        self.assertEqual(reader.seek(0), 0)
        self.assertEqual(reader.read(), b"segment")
        self.assertEqual(reader.seek(-3, os.SEEK_END), len(b"segment") - 3)
        self.assertEqual(reader.read(), b"ent")
        with self.assertRaisesRegex(ValueError, "Cannot seek"):
            reader.seek(1, os.SEEK_END)

    def test_rejects_unsafe_object_names(self):
        with self.assertRaises(ValueError):
            swift_io._safe_object_name("../secret")


if __name__ == "__main__":
    unittest.main()
