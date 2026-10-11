# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent online MBPO dynamics fitting on observed rough-MJ histories and flat recipe replay."""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from isaaclab_neural.data.opd import file_sha256
from isaaclab_neural.eval.dynamics_consistency import state_errors
from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_history_loss, _backward_rollout_loss
from isaaclab_neural.rl.rsl_rl.mbpo import HISTORY_KEYS, initialize_branch_scene, project_history
from isaaclab_neural.rl.rsl_rl.mbpo_loop import atomic_json, wait_for
from isaaclab_neural.train.opd_sequence import G1SequenceStudent
from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint, save_checkpoint


def select_windows(data: dict, rows: torch.Tensor) -> dict:
    """Subset real H10/future windows while keeping exact native-contact CSR frame membership."""
    history = data["states"].shape[1]
    output = {
        key: value.index_select(0, rows)
        for key, value in data.items()
        if isinstance(value, torch.Tensor) and key != "native_counts"
    }
    counts = data["native_counts"]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    frames = (rows[:, None] * history + torch.arange(history)).reshape(-1)
    starts, lengths = offsets[frames], counts[frames]
    index = torch.repeat_interleave(starts, lengths)
    index += torch.arange(int(lengths.sum())) - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
    output["native"] = {key: value.index_select(0, index) for key, value in data["native"].items()}
    output["native_counts"] = lengths
    return output


@torch.no_grad()
def build_cache(solver, windows: dict, device) -> dict:
    """Prepare actual full-history MJ labels in the pretrained model's coordinate representation."""
    chunks, targets, truth, sequences = {}, [], [], []
    count = len(windows["states"])
    for start in range(0, count, 256):
        part = select_windows(windows, torch.arange(start, min(count, start + 256)))
        batch = project_history(part, solver)
        batch["next_states"] = part["next_states"].to(device)
        solver.process_neural_model_inputs(batch)
        target = solver.convert_next_states_to_prediction(
            states=batch["states"], next_states=batch["next_states"], dt=0.005
        )
        targets.append(target[:, -1].cpu())
        sequences.append(target.cpu())
        truth.append(batch.pop("next_states")[:, -1].cpu())
        for key, value in batch.items():
            chunks.setdefault(key, []).append(value.detach().cpu())
    cache = {
        "inputs": {key: torch.cat(values) for key, values in chunks.items()},
        "target": torch.cat(targets),
        "truth": torch.cat(truth),
        "sequence_target": torch.cat(sequences),
        "sequence_label_origin": "observed_mj_post_step",
        "indices": list(range(count)),
        "source_hashes": {},
        "families": [],
        "origin": "rough_mbpo_MJ_training",
    }
    if not all(
        torch.isfinite(value).all() for value in (*cache["inputs"].values(), cache["target"], cache["sequence_target"])
    ):
        raise FloatingPointError("Non-finite observed-MJ training input or label.")
    return cache


def to_records(windows: dict) -> list[dict]:
    """Convert raw physical windows to the differentiable sequence student's real-target format."""
    history = windows["states"].shape[1]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), windows["native_counts"].cumsum(0))).tolist()
    records = []
    for row in range(len(windows["states"])):
        frames = []
        for step in range(history):
            first, last = offsets[row * history + step : row * history + step + 2]
            frame = {key: windows[key][row, step] for key in HISTORY_KEYS}
            frame["native"] = {key: value[first:last] for key, value in windows["native"].items()}
            frames.append(frame)
        records.append(
            {
                "history": frames,
                "actions": windows["actions"][row],
                "action_targets": windows["action_targets"][row],
                "oracle_states": windows["oracle_states"][row],
                "physics": {
                    key.removeprefix("physics/"): value[row]
                    for key, value in windows.items()
                    if key.startswith("physics/")
                },
            }
        )
    return records


@torch.no_grad()
def region_errors(student, records: list[dict]) -> dict:
    """Measure held-out, real-action4/16-physical-step endpoint errors in SI units."""
    results = {"4": {}, "16": {}}
    for start in range(0, len(records), student.num_envs):
        batch = records[start : start + student.num_envs]
        prediction = student.rollout(batch, max_physics_steps=16)[0]
        truth = torch.stack([row["oracle_states"] for row in batch]).to(prediction.device)
        for horizon in (4, 16):
            predicted, actual = prediction[:, horizon - 1], truth[:, horizon - 1]
            values = state_errors(predicted, actual, 44)
            values["root_height_bias_m"] = predicted[:, 2] - actual[:, 2]
            values["root_height_abs_m"] = (predicted[:, 2] - actual[:, 2]).abs()
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Non-finite held-out rough-terrain rollout.")
                results[str(horizon)].setdefault(key, []).append(value.cpu())
    return {
        horizon: {key: float(torch.cat(values).mean()) for key, values in metrics.items()}
        for horizon, metrics in results.items()
    }


def load_recipe_cache(entry: dict) -> dict:
    """Verify an immutable recipe artifact whose original HDF5 sources were audited before upload."""
    from isaaclab_neural.rl.rsl_rl.active_cycle import _require_full_history_cache

    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("Packaged pretraining recipe cache changed.")
    cache = torch.load(entry["path"], map_location="cpu", weights_only=True)
    if not entry.get("source_hashes") or cache["source_hashes"] != entry["source_hashes"]:
        raise ValueError("Recipe cache source lineage differs from the audited upload manifest.")
    _require_full_history_cache("reference", cache)
    return cache


def run_model_fit(env, args, config: dict) -> None:
    """Refit a persistent model/Adam pair, keeping rough holdout rows and frozen normalization separate."""
    from isaaclab_neural.generate.opd import simulation_protocol

    root, base = Path(config["run_dir"]), env.unwrapped
    stop = root / "FAILED"
    for name in ("model_data", "models", "model_fits"):
        (root / name).mkdir(parents=True, exist_ok=True)
    context = initialize_branch_scene(base)
    wait_for(root / "model_data" / "meta.json", config["wait_timeout_s"], stop)
    metadata = json.loads((root / "model_data" / "meta.json").read_text())
    if (
        metadata["protocol"] != simulation_protocol(base)
        or metadata["terrain_context"] != context
        or metadata["holdout_env_ids"] != config["model_holdout_env_ids"]
    ):
        raise ValueError("Collector and dynamics learner have different terrain/physics/holdout contracts.")
    initial = load_checkpoint(args.dynamics, device=base.device)
    solver = base.neural_adapter.solver
    model = reconstruct_model_from_checkpoint(initial, solver, device=base.device).requires_grad_(True)
    weights = initial["loss_weights"].to(base.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model_fit_learning_rate"])
    student = G1SequenceStudent(
        base, solver, model, metadata, weights, pd_mode=config["model_pd_mode"], deterministic_geometry=None
    )
    reference = load_recipe_cache(config["reference_cache"])
    frozen = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith(("input_rms.", "output_rms.", "body_net_contact_force_rms."))
    }
    data = ModelData(
        root,
        max_cached_files=config["model_cache_files"],
        holdout_ids=config["model_holdout_env_ids"],
        max_sample_files=config["model_fit_files_per_chunk"],
        recent_eval_iterations=config["model_recent_iterations"],
    )
    rng = np.random.default_rng(config["seed"] + 5000)
    round_index, last_through, updates_total = 0, -1, 0
    old_stream = []
    last_progress = time.monotonic()
    try:
        while True:
            if stop.exists():
                raise RuntimeError(stop.read_text()[:500])
            iterations = data.iterations()
            done = (root / "COLLECTION_DONE.json").exists()
            through = iterations[-1] if iterations else -1
            if done and through <= last_through:
                break
            ready = through + 1 >= config["model_first_update"] and through > last_through
            ready = ready and (last_through < 0 or done or through - last_through >= config["model_update_every"])
            if not ready:
                if time.monotonic() - last_progress > config["wait_timeout_s"]:
                    raise TimeoutError("Online model fitting received no eligible new MJ data.")
                time.sleep(1)
                continue
            holdout = data.sample(through, config["model_eval_rows"], rng, holdout=True)
            training = data.sample(through, config["model_fit_batch"], rng, holdout=False)
            enough = len(holdout) == config["model_eval_rows"] and len(training) == config["model_fit_batch"]
            atomic_json(
                {
                    "data_through_iteration": through,
                    "ready": enough,
                    "heldout_rows": len(holdout),
                    "training_rows": len(training),
                    "required_heldout_rows": config["model_eval_rows"],
                    "required_training_rows": config["model_fit_batch"],
                },
                root / "MODEL_DATA_READY.json",
            )
            if not enough:
                if done:
                    raise RuntimeError("Collected data cannot cover the declared disjoint train/holdout batches.")
                time.sleep(1)
                continue
            held_records = to_records(data.gather(holdout))
            model.eval()
            before = region_errors(student, held_records)
            started = time.monotonic()
            curve = []
            completed = 0
            while completed < config["model_fit_round_updates"]:
                if stop.exists():
                    raise RuntimeError("MBPO peer stopped during model fitting.")
                logical = min(config["model_fit_chunk_updates"], config["model_fit_round_updates"] - completed)
                batch_size = config["model_fit_batch"]
                selected = data.sample(through, logical * batch_size, rng, holdout=False)
                logical = min(logical, len(selected) // batch_size)
                selected = selected[: logical * batch_size]
                windows = data.gather(selected)
                cache = build_cache(solver, windows, base.device)
                ids = rng.permutation(len(selected)).tolist()
                multi_rows = config["model_multistep_rows"]
                multi_ids = (
                    rng.choice(len(selected), logical * multi_rows, replace=False)
                    if config["model_multistep_weight"]
                    else []
                )
                records = (
                    to_records(select_windows(windows, torch.as_tensor(multi_ids, dtype=torch.long)))
                    if len(multi_ids)
                    else []
                )
                for index in range(logical):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    while len(old_stream) < batch_size:
                        old_stream.extend(rng.permutation(len(reference["target"])).tolist())
                    old_ids, old_stream = old_stream[:batch_size], old_stream[batch_size:]
                    values = {
                        "new": _backward_history_loss(
                            model,
                            cache,
                            ids[index * batch_size : (index + 1) * batch_size],
                            weights,
                            base.device,
                            config["model_fit_micro_batch"],
                            source="current",
                            truth_weight=0.5,
                        ),
                        "recipe": _backward_history_loss(
                            model,
                            reference,
                            old_ids,
                            weights,
                            base.device,
                            config["model_fit_micro_batch"],
                            source="reference",
                            truth_weight=config["model_recipe_weight"],
                        ),
                    }
                    values["multistep"] = (
                        _backward_rollout_loss(
                            student,
                            records,
                            list(range(index * multi_rows, (index + 1) * multi_rows)),
                            config["model_multistep_weight"],
                            16,
                        )
                        if records
                        else 0.0
                    )
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    completed += 1
                    updates_total += 1
                    if completed % 32 == 0 or completed == config["model_fit_round_updates"]:
                        curve.append({"update": completed, **values, "gradient_norm": float(norm)})
                del cache, windows, records
            if any(not torch.equal(value, model.state_dict()[key]) for key, value in frozen.items()):
                raise RuntimeError("Online MBPO changed frozen pretrained normalization statistics.")
            model.eval()
            after = region_errors(student, held_records)
            path = root / "models" / f"model_r{round_index:03d}.pt"
            checkpoint_cfg = copy.deepcopy(initial["cfg"])
            checkpoint_cfg["mbpo"] = {
                "terrain_context": context,
                "model_round": round_index,
                "data_through_iteration": through,
                "optimizer_updates": updates_total,
            }
            save_checkpoint(
                path, model, "G1-Rough-Native-ContactTokens", checkpoint_cfg, epoch=updates_total, loss_weights=weights
            )
            published = {
                "round": round_index,
                "path": str(path),
                "sha256": file_sha256(path),
                "data_through_iteration": through,
                "terrain_context": context,
            }
            atomic_json(published, root / "models" / "latest.json")
            report = {
                "round": round_index,
                "data_through_iteration": through,
                "optimizer_updates": updates_total,
                "updates": completed,
                "curve": curve,
                "heldout_before": before,
                "heldout_after": after,
                "normalization_frozen": True,
                "flat_retention_gate_applied": False,
                "action": "published",
                "seconds": time.monotonic() - started,
            }
            atomic_json(report, root / "model_fits" / f"round_{round_index:03d}.json")
            with (root / "model_fit.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            print(json.dumps(report, allow_nan=False), flush=True)
            round_index, last_through = round_index + 1, through
            last_progress = time.monotonic()
        atomic_json(
            {
                "rounds": round_index,
                "optimizer_updates": updates_total,
                "normalization_frozen": True,
                "terrain_context": context,
            },
            root / "MODEL_FIT_DONE.json",
        )
    except Exception as error:
        atomic_json({"role": "model_fit", "error": repr(error)}, stop)
        raise


class ModelData:
    """Bounded in-memory index of immutable MJ data files with permanent world-based train/holdout roles."""

    def __init__(
        self,
        root: Path,
        holdout_from: int | None = None,
        max_cached_files: int = 16,
        holdout_ids: list[int] | None = None,
        max_sample_files: int = 512,
        recent_eval_iterations: int = 50,
    ):
        if max_sample_files < 1 or recent_eval_iterations < 1:
            raise ValueError("Model sampling requires positive file and validation-window budgets.")
        self.max_sample_files, self.recent_eval_iterations = max_sample_files, recent_eval_iterations
        if (holdout_from is None) == (holdout_ids is None):
            raise ValueError("Choose one explicit holdout-ID set or a trailing-world split.")
        self.root, self.holdout_from, self.max_cached_files = root / "model_data", holdout_from, max_cached_files
        self.holdout_ids = torch.tensor(holdout_ids, dtype=torch.long) if holdout_ids is not None else None
        self.cache: OrderedDict[int, dict] = OrderedDict()

    def iterations(self) -> list[int]:
        return sorted(int(path.stem.split("_")[1]) for path in self.root.glob("data_*.pt"))

    def load(self, iteration: int) -> dict:
        if iteration not in self.cache:
            if len(self.cache) >= self.max_cached_files:
                self.cache.popitem(last=False)
            self.cache[iteration] = torch.load(
                self.root / f"data_{iteration:05d}.pt", map_location="cpu", weights_only=False, mmap=True
            )
        self.cache.move_to_end(iteration)
        return self.cache[iteration]

    def sample(self, through: int, count: int, rng, *, holdout: bool) -> list[tuple[int, int]]:
        """Sample approximately uniformly from all available real iterations without leaking holdout worlds."""
        iterations = [value for value in self.iterations() if value <= through]
        chosen = (
            [value for value in iterations if value > through - self.recent_eval_iterations]
            if holdout
            else rng.choice(iterations, min(self.max_sample_files, len(iterations)), replace=False)
        )
        pool = []
        for iteration in sorted(chosen):
            ids = self.load(int(iteration))["env_ids"]
            selected = torch.isin(ids, self.holdout_ids) if self.holdout_ids is not None else ids >= self.holdout_from
            valid = selected if holdout else ~selected
            pool.extend((int(iteration), int(row)) for row in valid.nonzero().flatten())
        picks = rng.choice(len(pool), min(count, len(pool)), replace=False)
        return sorted(pool[index] for index in picks)

    def gather(self, pairs: list[tuple[int, int]]) -> dict:
        """Gather selected real histories and their corresponding native contact arrays."""
        grouped = {}
        for iteration, row in pairs:
            grouped.setdefault(iteration, []).append(row)
        parts = []
        for iteration, rows in sorted(grouped.items()):
            part = select_windows(self.load(iteration), torch.tensor(rows, dtype=torch.long))
            part["iteration"] = torch.full((len(rows),), iteration, dtype=torch.long)
            parts.append(part)
        output = {key: torch.cat([part[key] for part in parts]) for key in parts[0] if key != "native"}
        output["native"] = {key: torch.cat([part["native"][key] for part in parts]) for key in parts[0]["native"]}
        return output
