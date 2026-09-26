# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register managed Nucleus/W&B credentials and attach secret-free references to workflow templates."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path


def validate_credential_name(name: str) -> None:
    """Reject names that could change the structure of a workflow template."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
        raise ValueError("Invalid Nucleus credential name.")


def _service_bindings(service: str) -> dict[str, str]:
    if service == "nucleus":
        return {"OMNI_USER": "omni_user", "OMNI_PASS": "omni_pass"}
    if service == "wandb":
        return {"WANDB_API_KEY": "api_key"}
    raise ValueError(f"Unsupported managed credential service: {service}.")


def register_credential(service: str = "nucleus") -> str:
    """Create a fresh managed credential without exposing its payload in arguments or logs."""
    bindings = _service_bindings(service)
    if any(not os.environ.get(variable) for variable in bindings):
        raise ValueError(f"Set the required environment variables for {service}: {', '.join(bindings)}.")
    if service == "nucleus" and os.environ["OMNI_USER"] != "$omni-api-token":
        raise ValueError("Set OMNI_USER to the literal '$omni-api-token'.")
    name = f"so101-{service}-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=f"{service}-credential-") as directory:
        paths = {}
        for variable, key in bindings.items():
            path = Path(directory) / key
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(os.environ[variable])
            paths[key] = path
        result = subprocess.run(
            [
                "osmo",
                "credential",
                "set",
                name,
                "--type",
                "GENERIC",
                "--payload-file",
                *(f"{key}={path}" for key, path in paths.items()),
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(
            f"Credential registration failed or was not confirmed for {name} (exit {result.returncode}). "
            "Check its existence before retrying; response details were suppressed to protect the payload."
        )
    return name


def attach_credential(template: Path, output: Path, name: str, service: str = "nucleus") -> Path:
    """Attach only a credential reference, preserving file paths when relocating the template."""
    import yaml

    validate_credential_name(name)
    body, separator, defaults = template.read_text().partition("\ndefault-values:\n")
    marker = "    credentials:\n"
    if not separator or body.count(marker) != 1:
        raise ValueError("Expected a single-task OSMO template with credentials and default-values.")
    values = yaml.safe_load(defaults)
    if name == values.get("storage_credential") or re.search(
        rf"^      [\"\']?{re.escape(name)}[\"\']?:\s*$", body, re.MULTILINE
    ):
        raise ValueError("Service and storage credentials must have distinct names; credential already attached.")
    for key, value in values.items():
        if key.endswith("_localpath") and isinstance(value, str) and not Path(value).is_absolute():
            values[key] = str((template.resolve().parent / value).resolve())
    binding = f"      {json.dumps(name)}:\n" + "".join(
        f"        {variable}: {key}\n" for variable, key in _service_bindings(service).items()
    )
    with output.open("x") as stream:
        stream.write(body.replace(marker, marker + binding, 1))
        stream.write(separator)
        stream.write(textwrap.indent(yaml.safe_dump(values, sort_keys=False), "  "))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--service", choices=("nucleus", "wandb"), default="nucleus")
    attach = commands.add_parser("attach")
    attach.add_argument("--service", choices=("nucleus", "wandb"), default="nucleus")
    attach.add_argument("--template", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--credential", required=True)
    args = parser.parse_args()
    if args.command == "register":
        print(register_credential(args.service))
    else:
        attach_credential(args.template, args.output, args.credential, args.service)


if __name__ == "__main__":
    main()
