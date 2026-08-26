# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the exact fixed-epoch checkpoint contract from W&B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import CHECKPOINT_EPOCH, EXPECTED_SEEDS, sha256_file


def download_checkpoint_set(
    source_manifest: str | Path,
    output_root: str | Path,
    *,
    entity: str,
    project: str,
    api: Any,
) -> Path:
    """Download and verify one six-checkpoint old/new fixed-epoch set."""
    manifest_path = Path(source_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    encoder = str(manifest.get("encoder", ""))
    if manifest.get("schema_version") != 1 or encoder not in ("a", "d"):
        raise ValueError("Unsupported checkpoint source manifest.")
    if int(manifest.get("checkpoint_epoch", -1)) != CHECKPOINT_EPOCH:
        raise ValueError(f"Expected checkpoint epoch {CHECKPOINT_EPOCH}.")
    entries = list(manifest.get("checkpoints", []))
    expected_pairs = {(group, seed) for group in ("old", "new") for seed in EXPECTED_SEEDS}
    actual_pairs = {(str(entry.get("group")), int(entry.get("seed", -1))) for entry in entries}
    if actual_pairs != expected_pairs or len(entries) != len(expected_pairs):
        raise ValueError("Checkpoint source manifest does not contain the exact old/new three-seed set.")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for entry in sorted(entries, key=lambda item: (str(item["group"]), int(item["seed"]))):
        target = (root / entry["path"]).resolve()
        if root not in target.parents:
            raise ValueError(f"Checkpoint path escapes output root: {entry['path']!r}.")
        target.parent.mkdir(parents=True, exist_ok=True)
        run = api.run(f"{entity}/{project}/{entry['wandb_run_id']}")
        remote = run.file(f"model_epoch{CHECKPOINT_EPOCH}.pt")
        downloaded = Path(remote.download(root=str(target.parent), replace=True).name).resolve()
        if downloaded != target:
            raise ValueError(f"W&B downloaded checkpoint to unexpected path: {downloaded}.")
        if target.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Checkpoint size mismatch: {target}.")
        if sha256_file(target) != entry["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {target}.")

    output_manifest = root / f"checkpoint_manifest_{encoder}.json"
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    import wandb

    download_checkpoint_set(
        args.source_manifest,
        args.output_root,
        entity=args.entity,
        project=args.project,
        api=wandb.Api(),
    )


if __name__ == "__main__":
    main()
