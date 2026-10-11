#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve Docker Hub login material for OSMO REGISTRY credentials.

OSMO applies ``REGISTRY`` credentials outside workflow YAML. This helper reads
``DOCKERHUB_USERNAME`` / ``DOCKERHUB_TOKEN`` or ``~/.docker/config.json`` and
writes username/token files for ``osmo credential set --payload-file``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

DOCKER_HUB_REGISTRY = "docker.io"
DOCKER_CONFIG_AUTH_KEYS = (
    "https://index.docker.io/v1/",
    "https://index.docker.io/v1",
    "https://registry-1.docker.io",
    "registry-1.docker.io",
    "docker.io",
)


class RegistryCredentialError(RuntimeError):
    """Raised when Docker Hub credentials cannot be resolved."""


def docker_config_path() -> Path:
    """Return the Docker CLI config path."""
    return Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker")) / "config.json"


def load_docker_hub_login(
    username: str | None = None,
    token: str | None = None,
    config_path: Path | None = None,
) -> tuple[str, str]:
    """Return ``(username, token)`` for Docker Hub.

    Environment variables win, then explicit arguments, then Docker's stored
    ``auths`` entry. The token is never printed.

    Args:
        username: Optional Docker Hub username override.
        token: Optional Docker Hub access token override.
        config_path: Optional Docker ``config.json`` path.

    Returns:
        Username and access token or password.

    Raises:
        RegistryCredentialError: If no usable Docker Hub login is available.
    """
    resolved_user = (username or os.environ.get("DOCKERHUB_USERNAME") or "").strip()
    resolved_token = (token or os.environ.get("DOCKERHUB_TOKEN") or "").strip()
    if resolved_user and resolved_token:
        return resolved_user, resolved_token

    path = config_path if config_path is not None else docker_config_path()
    if not path.is_file():
        raise RegistryCredentialError(
            "Docker Hub credentials were not found. Run `docker login`, or set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryCredentialError(f"Could not parse Docker config at {path}.") from exc

    auths = payload.get("auths") or {}
    entry = None
    for key in DOCKER_CONFIG_AUTH_KEYS:
        if key in auths:
            entry = auths[key]
            break
    if not isinstance(entry, dict):
        raise RegistryCredentialError(f"No Docker Hub auth entry in {path}. Run `docker login` for docker.io.")
    auth_b64 = entry.get("auth")
    if not auth_b64:
        raise RegistryCredentialError(
            "Docker Hub auth in config.json has no `auth` field. Re-run `docker login` "
            "or set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN."
        )
    try:
        decoded = base64.b64decode(auth_b64).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RegistryCredentialError("Docker Hub auth entry in config.json is not valid base64.") from exc
    if ":" not in decoded:
        raise RegistryCredentialError("Docker Hub auth entry did not contain username:token.")
    config_user, config_token = decoded.split(":", 1)
    config_user = config_user.strip()
    config_token = config_token.strip()
    if not config_user or not config_token:
        raise RegistryCredentialError("Docker Hub auth entry was missing a username or token.")
    return resolved_user or config_user, resolved_token or config_token


def write_payload_files(username_file: Path, auth_file: Path, **kwargs) -> tuple[str, str]:
    """Write username and token files for ``osmo credential set --payload-file``.

    Returns:
        The username and the auth-file path as strings. The token itself is not
        returned so callers do not accidentally print it.
    """
    user, token = load_docker_hub_login(**kwargs)
    username_file.write_text(user, encoding="utf-8")
    auth_file.write_text(token, encoding="utf-8")
    username_file.chmod(0o600)
    auth_file.chmod(0o600)
    return user, str(auth_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser(
        "write-payload-files",
        help="Write Docker Hub username and token files for OSMO.",
    )
    write_parser.add_argument("--username-file", required=True)
    write_parser.add_argument("--auth-file", required=True)
    args = parser.parse_args()
    if args.command == "write-payload-files":
        user, _auth_path = write_payload_files(Path(args.username_file), Path(args.auth_file))
        print(user)


if __name__ == "__main__":
    main()
