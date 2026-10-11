# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare an immutable rough-MBPO bundle locally; submit only with explicit --confirm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml
from lib.package_code import package_code
from lib.preset import load_preset

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path: Path) -> str:
    """Hash an explicitly selected input without logging its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


NVDATASET_IO = ROOT / "osmo_scripts" / "lib" / "nvdataset_io.py"
NVDATASET_REQUIREMENTS = ROOT / "osmo_scripts" / "nvdataset_requirements.in"
SAFE_NAME = r"[A-Za-z0-9_.-]+"

BOOTSTRAP = r"""set -euo pipefail
export PROJECT_ROOT=/tmp/mbpo-source/IsaacLab-NeRD
export MBPO_INPUT_DIR=/tmp/mbpo-inputs
export MBPO_OUTPUT_ROOT=/tmp/mbpo-output
export MBPO_RUN_OUTPUT="$MBPO_OUTPUT_ROOT/$WORKFLOW_ID"
if [ -x /opt/isaaclab-venv/bin/python ]; then
    export VIRTUAL_ENV=/opt/isaaclab-venv
    export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
# OSMO nodes cannot resolve the internal package index; the image ships an isolated SDK venv.
if [ -z "${NVDATASET_PYTHON:-}" ] || [ ! -x "$NVDATASET_PYTHON" ]; then
    echo "[FATAL] Image lacks the preinstalled NV-Datasets SDK (NVDATASET_PYTHON)." >&2
    exit 2
fi
nvdataset_io() {
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV "$NVDATASET_PYTHON" /tmp/nvdataset_io.py "$@"
}
nvdataset_io check
upload_output() {
    local status=$?
    trap - EXIT
    set +e
    if [ -n "$(ls -A "$MBPO_RUN_OUTPUT" 2>/dev/null)" ]; then
        echo "Uploading $MBPO_RUN_OUTPUT to NV-Datasets $NVDATASET_OUTPUT_DATASET/$WORKFLOW_ID"
        nvdataset_io upload-directory --dataset "$NVDATASET_OUTPUT_DATASET" --source-dir "$MBPO_OUTPUT_ROOT" \
            --description "IsaacLab-NeRD G1 rough MBPO outputs for $WORKFLOW_ID."
        local upload_status=$?
        echo "NV-Datasets output upload finished with status $upload_status"
        if [ "$status" -eq 0 ]; then
            status=$upload_status
        fi
    fi
    exit "$status"
}
trap upload_output EXIT
mkdir -p "$MBPO_RUN_OUTPUT"
nvdataset_io download --dataset "$NVDATASET_INPUT_DATASET" --output-dir "$MBPO_INPUT_DIR"
python3 - <<'PY'
import hashlib, json, os, pathlib, shlex
manifest = json.loads(pathlib.Path('/tmp/mbpo-inputs.json').read_text())
root = pathlib.Path(os.environ['MBPO_INPUT_DIR'])
expected = {
    'CODE': ('code.tar.gz', manifest['code_snapshot_sha256']),
    'DYNAMICS': (manifest['dynamics']['basename'], manifest['dynamics']['sha256']),
    'REFERENCE': (manifest['reference']['basename'], manifest['reference']['sha256']),
}
lines = []
for key, (name, digest) in expected.items():
    found = [path for path in root.rglob(name) if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one downloaded {name}, found {len(found)}.')
    with found[0].open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
            raise RuntimeError(f'Downloaded {name} differs from the approved snapshot.')
    lines.append(f'export MBPO_{key}={shlex.quote(str(found[0]))}\n')
pathlib.Path('/tmp/mbpo-input-paths.sh').write_text(''.join(lines))
print('Verified NV-Datasets inputs:', {key: value[0] for key, value in expected.items()})
PY
source /tmp/mbpo-input-paths.sh
mkdir -p /tmp/mbpo-source
tar --no-same-owner -xzf "$MBPO_CODE" -C /tmp/mbpo-source
python3 "$PROJECT_ROOT/osmo_scripts/lib/local_isaaclab.py" bootstrap --project-root "$PROJECT_ROOT"
if ! python3 -c 'import wandb' >/dev/null 2>&1; then
    uv pip install --python "$(command -v python3)" --exclude-newer CUTOFF 'wandb==0.26.1'
fi
for package in isaaclab_neural isaaclab isaaclab_newton isaaclab_tasks isaaclab_assets isaaclab_rl; do
    export PYTHONPATH="$PROJECT_ROOT/source/$package${PYTHONPATH:+:$PYTHONPATH}"
done
python3 -m isaaclab_neural.rl.rsl_rl.mbpo_train \
    --preset-file "$PROJECT_ROOT/osmo_scripts/presets/g1_mbpo_rough.yaml" \
    --input-manifest /tmp/mbpo-inputs.json \
    --dynamics "$MBPO_DYNAMICS" \
    --reference-cache "$MBPO_REFERENCE" \
    --output "$MBPO_RUN_OUTPUT/mbpo" SMOKE
"""


def render_workflow(
    preset: dict,
    name: str,
    input_dataset: str,
    credential: str,
    entity: str,
    manifest: Path,
    *,
    smoke_only: bool = False,
) -> dict:
    """Render a single-GPU NV-Datasets workflow with credential references, never secret values."""
    storage = preset["storage"]
    if storage.get("backend") != "nvdataset":
        raise ValueError("The MBPO preset requires the NV-Datasets storage backend.")
    for value in (name, input_dataset, credential, storage["credential"], storage["output_dataset"]):
        if not re.fullmatch(SAFE_NAME, value):
            raise ValueError("Unsafe workflow, dataset or credential name.")
    if credential == storage["credential"]:
        raise ValueError("W&B and NV-Datasets credentials must be separate OSMO credentials.")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    bootstrap = BOOTSTRAP.replace("CUTOFF", cutoff).replace("SMOKE", "--smoke-only" if smoke_only else "")
    resources, workflow = preset["resources"], preset["workflow"]
    return {
        "workflow": {
            "name": name,
            "timeout": {"exec_timeout": workflow["exec_timeout"], "queue_timeout": workflow["queue_timeout"]},
            "resources": {
                "default": {
                    "cpu": resources["num_cpu"],
                    "gpu": resources["num_gpu"],
                    "memory": resources["memory"],
                    "storage": resources["storage"],
                    "platform": resources["platform"],
                }
            },
            "tasks": [
                {
                    "name": "mbpo",
                    "image": workflow["image"],
                    "command": ["bash"],
                    "args": ["/tmp/mbpo.sh"],
                    "environment": {
                        "ACCEPT_EULA": "Y",
                        "PRIVACY_CONSENT": "Y",
                        "OMNI_KIT_ALLOW_ROOT": "1",
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                        "WORKFLOW_ID": "{{workflow_id}}",
                        "WANDB_ENTITY": entity,
                        "NVDATASET_INPUT_DATASET": input_dataset,
                        "NVDATASET_OUTPUT_DATASET": storage["output_dataset"],
                    },
                    "credentials": {
                        credential: {"WANDB_API_KEY": "api_key"},
                        storage["credential"]: {"NGC_API_KEY": "nvapi_key", "NVDATASET_TENANTID": "tenant_id"},
                    },
                    "files": [
                        {"path": "/tmp/mbpo.sh", "contents": bootstrap},
                        {"path": "/tmp/mbpo-inputs.json", "localpath": str(manifest)},
                        {"path": "/tmp/nvdataset_io.py", "localpath": str(NVDATASET_IO)},
                    ],
                }
            ],
        }
    }


def prepare(args) -> None:
    """Audit selected training assets and create a local plan without any remote writes."""
    preset = load_preset(args.preset)
    if args.preset != "g1_mbpo_rough" or preset["workflow"].get("kind") != "mbpo":
        raise ValueError("This launcher currently supports only the g1_mbpo_rough preset.")
    if preset["resources"]["num_gpu"] != 1:
        raise ValueError("The portable MBPO workers currently share exactly one assigned GPU.")
    dynamics = args.dynamics.resolve()
    reference_manifest = json.loads(args.reference_manifest.read_text())
    reference = reference_manifest["reference"]
    reference_path = Path(reference["path"]).resolve()
    if file_digest(reference_path) != reference["sha256"]:
        raise ValueError("Original reference cache differs from its recorded manifest.")
    sources = reference.get("source_hashes", {})
    if not sources:
        raise ValueError("Reference cache has no auditable original training-source lineage.")
    for source, expected in sources.items():
        if file_digest(Path(source)) != expected:
            raise ValueError("An original recipe training source changed; no upload was performed.")
    cache = torch.load(reference_path, map_location="cpu", weights_only=True)
    if cache.get("source_hashes") != sources or "sequence_target" not in cache:
        raise ValueError("The recipe cache lacks matching source lineage or full-history truth.")
    checkpoint = torch.load(dynamics, map_location="cpu", weights_only=False)
    solver = checkpoint["cfg"]["env"]["neural_solver_cfg"]
    if (
        checkpoint["loss_weights"].numel() != 87
        or solver.get("num_states_history") != 10
        or solver.get("contact_representation") != "contact_tokens"
        or cache["sequence_target"].shape[1:] != (10, 87)
    ):
        raise ValueError("The selected assets do not match the37-joint/H10/contact-token G1 contract.")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    name = f"{preset['workflow']['base_name']}-{identifier}"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "code.tar.gz"
    package_code(ROOT, archive)
    credential = (
        args.wandb_credential if args.wandb_credential is not None else "g1-mbpo-wandb-" + uuid.uuid4().hex[:16]
    )
    manifest = {
        "schema_version": 1,
        "code_snapshot_sha256": file_digest(archive),
        "dynamics": {"sha256": file_digest(dynamics), "basename": dynamics.name, "initialization": "flat_F_pre"},
        "reference": {
            "sha256": reference["sha256"],
            "source_hashes": sources,
            "raw_sources_verified_before_upload": True,
            "anchors": len(cache["target"]),
            "basename": reference_path.name,
        },
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    if len({"code.tar.gz", dynamics.name, reference_path.name}) != 3:
        raise ValueError("Code, dynamics and reference inputs need distinct file names in one dataset.")
    manifest_path = output / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    storage = preset["storage"]
    input_dataset = f"{storage['input_dataset_prefix']}-{identifier}"
    entity = os.environ.get("WANDB_ENTITY", "")
    if not entity:
        raise ValueError("Set WANDB_ENTITY to make the approved project destination explicit.")
    credentials = subprocess.run(["osmo", "credential", "list"], check=True, capture_output=True, text=True).stdout
    if not re.search(rf"^{re.escape(storage['credential'])}\s+GENERIC\b", credentials, re.MULTILINE):
        raise ValueError(f"OSMO GENERIC credential {storage['credential']!r} is required for NV-Datasets.")
    spec = render_workflow(preset, name, input_dataset, credential, entity, manifest_path, smoke_only=args.smoke_only)
    workflow_path = output / "workflow.yaml"
    workflow_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    plan = {
        "name": name,
        "pool": args.pool or preset["workflow"]["pool"],
        "priority": "NORMAL",
        "workflow": str(workflow_path),
        "workflow_sha256": file_digest(workflow_path),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": file_digest(manifest_path),
        "wandb": {"entity": entity, **preset["wandb"]},
        "credential": credential,
        "create_wandb_credential": args.wandb_credential is None,
        "resources": preset["resources"],
        "image": preset["workflow"]["image"],
        "smoke_only": args.smoke_only,
        "storage": {
            "backend": "nvdataset",
            "credential": storage["credential"],
            "input_dataset": input_dataset,
            "output_dataset": storage["output_dataset"],
            "output_directory": "<workflow_id>",
            "index_url": storage["index_url"],
        },
        "uploads": [
            {"path": str(archive), "sha256": manifest["code_snapshot_sha256"]},
            {"path": str(dynamics), "sha256": manifest["dynamics"]["sha256"]},
            {"path": str(reference_path), "sha256": reference["sha256"]},
        ],
        "preset": args.preset,
    }
    (output / "submission_plan.json").write_text(json.dumps(plan, indent=2))
    print(
        json.dumps(
            {
                "prepared": str(output),
                "workflow": name,
                "pool": plan["pool"],
                "remote_writes": False,
                "raw_training_sources_verified": len(sources),
                "storage": plan["storage"],
                "wandb": plan["wandb"],
                "upload_bytes": sum(Path(entry["path"]).stat().st_size for entry in plan["uploads"]),
            }
        ),
        flush=True,
    )


def upload_inputs(plan: dict, directory: Path) -> None:
    """Create the plan's new NV-Datasets input dataset from an isolated SDK environment."""
    storage = plan["storage"]
    if storage.get("backend") != "nvdataset":
        raise ValueError("This plan does not use NV-Datasets storage; prepare a fresh plan.")
    missing = [key for key in ("NGC_API_KEY", "NVDATASET_TENANTID") if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} before uploading NV-Datasets inputs.")
    uploaded = directory / "INPUTS_UPLOADED"
    if uploaded.exists():
        return
    started = directory / "INPUT_UPLOAD_ATTEMPTED"
    if started.exists():
        raise RuntimeError(
            f"An earlier upload to {storage['input_dataset']} was interrupted; inspect it before retrying."
        )
    started.write_text(json.dumps({"dataset": storage["input_dataset"], "at": datetime.now(timezone.utc).isoformat()}))
    environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "VIRTUAL_ENV")}
    subprocess.run(
        [
            "uv", "run", "--no-project", "--python", "3.12",
            "--with-requirements", str(NVDATASET_REQUIREMENTS), "--index", storage["index_url"],
            "python", str(NVDATASET_IO), "create-files", "--dataset", storage["input_dataset"],
            "--description", f"Immutable G1 rough MBPO inputs for {plan['name']}.",
            *(entry["path"] for entry in plan["uploads"]),
        ],
        check=True,
        env=environment,
    )  # fmt: skip
    uploaded.write_text(started.read_text())


def submit(args) -> None:
    """Inject an existing environment key securely, validate, upload and submit exactly once after confirmation."""
    if not args.confirm:
        raise ValueError("Submission requires --confirm after review of submission_plan.json.")
    directory = args.prepared.resolve()
    plan = json.loads((directory / "submission_plan.json").read_text())
    for key in ("workflow", "input_manifest"):
        if file_digest(Path(plan[key])) != plan[f"{key}_sha256"]:
            raise ValueError("The prepared workflow or manifest changed; prepare and approve a fresh plan.")
    for entry in plan["uploads"]:
        if file_digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("A prepared upload changed; prepare and approve a fresh plan.")
    marker = directory / "SUBMISSION_ATTEMPTED"
    if marker.exists():
        raise RuntimeError("This plan already attempted a submission; query its workflow name before any retry.")
    if plan.get("create_wandb_credential", True):
        key = os.environ.get("WANDB_API_KEY")
        if not key:
            raise RuntimeError(
                "WANDB_API_KEY is not available in the environment; no credentials or workflow were created."
            )
        with tempfile.TemporaryDirectory(prefix="g1-mbpo-credential-") as temporary:
            payload = Path(temporary) / "api_key"
            descriptor = os.open(payload, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(key)
            subprocess.run(
                [
                    "osmo",
                    "credential",
                    "set",
                    plan["credential"],
                    "--type",
                    "GENERIC",
                    "--payload-file",
                    f"api_key={payload}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
    subprocess.run(["osmo", "workflow", "validate", plan["workflow"], "--pool", plan["pool"]], check=True)
    upload_inputs(plan, directory)
    marker.write_text(json.dumps({"workflow_name": plan["name"], "at": datetime.now(timezone.utc).isoformat()}))
    result = subprocess.run(
        [
            "osmo",
            "workflow",
            "submit",
            plan["workflow"],
            "--pool",
            plan["pool"],
            "--priority",
            plan["priority"],
            "--format-type",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    (directory / "submit_stdout.txt").write_text(result.stdout)
    (directory / "submit_stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("OSMO submission did not return success. Query the unique workflow name before retrying.")
    print(result.stdout, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("prepare")
    local.add_argument("--preset", default="g1_mbpo_rough")
    local.add_argument("--dynamics", type=Path, required=True)
    local.add_argument("--reference-manifest", type=Path, required=True)
    local.add_argument("--output-dir", type=Path, required=True)
    local.add_argument("--pool")
    local.add_argument("--wandb-credential", help="Reuse an existing OSMO GENERIC credential with an api_key entry.")
    local.add_argument("--smoke-only", action="store_true")
    local.set_defaults(function=prepare)
    remote = commands.add_parser("submit")
    remote.add_argument("--prepared", type=Path, required=True)
    remote.add_argument("--confirm", action="store_true")
    remote.set_defaults(function=submit)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
