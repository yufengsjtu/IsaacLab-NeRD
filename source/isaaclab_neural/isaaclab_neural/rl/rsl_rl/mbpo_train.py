# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded rough-G1 MBPO workflow: restore check, coupled smoke, PPO training and held-out MJ evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from isaaclab_neural.data.opd import file_sha256

DEFAULTS = {
    "terrain": "rough",
    "terrain_profile": "stock",
    "physics_events": "stock",
    "contact_mdp": "geometry",
    "feet_contact_separation": 0.0025,
    "ppo_runner": "stock_rough",
    "seed": 42,
    "policy_seed": 0,
    "iterations": 3000,
    "real_envs": 32,
    "model_envs": 4096,
    "model_fit_envs": 16,
    "warmup_iterations": 32,
    "max_warmup_iterations": 128,
    "require_model_before_ppo": True,
    "initial_learning_rate": 1e-3,
    "buffer_capacity": 20480,
    "anchor_steps_per_iteration": 4,
    "rollout_schedule": [0, 1, 24, 24],
    "branch_standing_fraction": 0.5,
    "ppo_real_mix_repeat": 1,
    "delete_consumed_buffers": True,
    "wait_timeout_s": 3600,
    "checkpoint_interval": 50,
    "checkpoints": [],
    "model_data_steps": [11, 23],
    "model_holdout_env_ids": [7, 15, 23, 31],
    "model_first_update": 25,
    "model_update_every": 10,
    "model_fit_learning_rate": 5e-5,
    "model_fit_round_updates": 256,
    "model_fit_chunk_updates": 64,
    "model_fit_files_per_chunk": 512,
    "model_recent_iterations": 50,
    "model_pd_mode": "torch",
    "model_fit_batch": 128,
    "model_fit_micro_batch": 32,
    "model_multistep_weight": 0.5,
    "model_multistep_rows": 16,
    "model_recipe_weight": 0.5,
    "model_eval_rows": 256,
    "model_cache_files": 16,
    "terrain_seed": 73542,
    "terrain_rows": 10,
    "terrain_cols": 20,
    "terrain_difficulty_min": 0.0,
    "terrain_difficulty_max": 1.0,
    "policy_observation_corruption": True,
    "max_output_gib": 64,
    "min_free_disk_gib": 10,
    "walltime_seconds": 169200,
    "evaluation_contact_mdp": "stock",
    "evaluation_seeds": [30, 31, 32],
    "evaluation_num_envs": 128,
    "evaluation_steps": 1000,
}


def validate_config(settings: dict) -> dict:
    """Resolve and validate a bounded, single-GPU rough MBPO configuration before launching workers."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown rough MBPO settings: {sorted(unknown)}")
    config = copy.deepcopy({**DEFAULTS, **settings})
    limits = {
        "iterations": (1, 3000),
        "real_envs": (4, 128),
        "model_envs": (8, 4096),
        "warmup_iterations": (2, 256),
        "max_warmup_iterations": (2, 1024),
        "checkpoint_interval": (1, 1000),
        "model_fit_files_per_chunk": (1, 1024),
        "model_recent_iterations": (1, 512),
        "model_fit_batch": (8, 1024),
        "model_fit_micro_batch": (1, 128),
        "model_fit_round_updates": (1, 1024),
        "model_fit_chunk_updates": (1, 256),
        "model_eval_rows": (1, 1024),
        "model_cache_files": (1, 64),
        "model_first_update": (1, 256),
        "model_update_every": (1, 100),
        "model_multistep_rows": (1, 128),
        "terrain_rows": (1, 20),
        "terrain_cols": (1, 20),
        "anchor_steps_per_iteration": (1, 24),
        "ppo_real_mix_repeat": (1, 32),
        "evaluation_num_envs": (4, 512),
        "evaluation_steps": (1, 1000),
    }
    for key, (low, high) in limits.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid {key}; expected integer in [{low}, {high}].")
    if type(config["model_fit_envs"]) is not int or config["model_fit_envs"] not in (4, 16):
        raise ValueError("MBPO model fitting requires the existing worker's 4 or 16 scratch-world contract.")
    if config["terrain"] != "rough" or not config["require_model_before_ppo"]:
        raise ValueError("This preset requires rough terrain and an online model update before PPO.")
    heldout = config["model_holdout_env_ids"]
    if (
        not heldout
        or len(set(heldout)) != len(heldout)
        or len(heldout) >= config["real_envs"]
        or any(type(index) is not int or not 0 <= index < config["real_envs"] for index in heldout)
    ):
        raise ValueError("MJ dynamics holdout IDs must be unique, valid and leave training worlds available.")
    if not config["model_first_update"] <= config["warmup_iterations"] <= config["max_warmup_iterations"]:
        raise ValueError("Model warm-up must become eligible before its bounded collection window ends.")
    for key, choices in {
        "terrain_profile": {"fixed", "stock"},
        "physics_events": {"nominal", "stock"},
        "contact_mdp": {"geometry"},
        "evaluation_contact_mdp": {"geometry", "stock"},
        "ppo_runner": {"stock_rough", "nerd_legacy"},
        "model_pd_mode": {"torch", "newton_fma"},
    }.items():
        if config[key] not in choices:
            raise ValueError(f"Unsupported {key}: {config[key]}")
    if (
        config["model_multistep_rows"] > config["model_fit_batch"]
        or config["model_fit_micro_batch"] > config["model_fit_batch"]
    ):
        raise ValueError("Model microbatches and multistep rows cannot exceed the logical batch.")
    if config["buffer_capacity"] < config["real_envs"] * config["anchor_steps_per_iteration"]:
        raise ValueError("Branch capacity cannot hold one collector iteration.")
    if config["model_data_steps"] != [11, 23]:
        raise ValueError("The H10/16-physics-step window must align with the existing24-control-step PPO rollout.")
    separation = config["feet_contact_separation"]
    if type(separation) not in (int, float) or not 0 <= separation <= 0.01:
        raise ValueError("feet_contact_separation must lie in [0, 0.01] m.")
    if not 0 <= config["terrain_difficulty_min"] <= config["terrain_difficulty_max"] <= 1:
        raise ValueError("Invalid rough terrain difficulty range.")
    if not 0 <= config["branch_standing_fraction"] <= 1:
        raise ValueError("Invalid standing reset fraction.")
    first, last, low, high = config["rollout_schedule"]
    if not 0 <= first < last or not 1 <= low <= high <= 24:
        raise ValueError("Rough model branches must stay within1..24 control steps.")
    for key in (
        "initial_learning_rate",
        "model_fit_learning_rate",
        "wait_timeout_s",
        "max_output_gib",
        "min_free_disk_gib",
        "walltime_seconds",
    ):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid positive budget/rate: {key}.")
    if any(
        type(seed) is not int or seed < 0
        for seed in (config["seed"], config["policy_seed"], config["terrain_seed"], *config["evaluation_seeds"])
    ):
        raise ValueError("All random seeds must be nonnegative integers.")
    checkpoints = config["checkpoints"] or range(
        config["checkpoint_interval"], config["iterations"] + 1, config["checkpoint_interval"]
    )
    config["checkpoints"] = sorted(
        {step for step in checkpoints if 0 < step <= config["iterations"]} | {config["iterations"]}
    )
    return config


def smoke_config(config: dict) -> dict:
    """Use the same terrain/model representation for a small coupled smoke, never its learned weights for training."""
    smoke = {
        **config,
        "iterations": 2,
        "real_envs": 8,
        "model_envs": 32,
        "model_fit_envs": 4,
        "warmup_iterations": 4,
        "max_warmup_iterations": 16,
        "model_fit_files_per_chunk": 16,
        "buffer_capacity": 128,
        "checkpoints": [1, 2],
        "model_holdout_env_ids": [1, 3, 5, 7],
        "model_first_update": 2,
        "model_update_every": 2,
        "model_fit_batch": 16,
        "model_fit_micro_batch": 4,
        "model_multistep_rows": 4,
        "model_fit_round_updates": 2,
        "model_fit_chunk_updates": 2,
        "model_eval_rows": 4,
        "wait_timeout_s": 1800,
    }
    validate_config({key: smoke[key] for key in DEFAULTS})
    return smoke


def worker_command(config: dict, config_path: Path, output: Path, role: str, policy: Path | None) -> list[str]:
    """Build a worker argv without shell interpolation or embedded credentials."""
    model = role == "model_fit"
    backend = "mjwarp" if role in {"init", "mj_collect", "probe_mj"} else "nerd"
    stage = "init" if role == "init" else "active-stage-fit" if model else "study-rl"
    worlds = (
        8
        if role in {"init", "probe_mj", "probe_nerd"}
        else config["model_fit_envs"]
        if model
        else config["real_envs"]
        if role == "mj_collect"
        else config["model_envs"]
    )
    module = "isaaclab_neural.generate.opd" if role == "init" else "isaaclab_neural.rl.rsl_rl.mbpo_worker"
    command = [
        sys.executable,
        "-m",
        module,
        "--stage",
        stage,
        "--backend",
        backend,
        "--terrain",
        "rough",
        "--scene-config",
        str(config_path),
        "--dynamics",
        config["dynamics_path"],
        "--output",
        str(output),
        "--num-envs",
        str(worlds),
        "--seed",
        str(config["seed"]),
        "--device",
        "cuda:0",
    ]
    if role == "init":
        command += ["--policy-seed", str(config["policy_seed"])]
    else:
        command += ["--active_fit_config" if model else "--rl_study_config", str(config_path)]
    if policy is not None and not model:
        command += ["--policy-snapshot", str(policy)]
    return command


class Workflow:
    """Own subprocess lifetimes, budgets, verification artifacts and one W&B run."""

    def __init__(self, config: dict, output: Path, logging: dict):
        self.config, self.output = config, output
        self.started = time.monotonic()
        self.offsets: dict[Path, int] = {}
        self.mj_seen: set[Path] = set()
        self.mj_steps: dict[str, int] = {}
        self.run = None
        self.save_checkpoints = logging.get("save_checkpoints", False)
        output.mkdir(parents=True, exist_ok=False)
        _write(config, output / "config.json")
        if logging.get("enabled", True):
            import wandb

            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError("Online W&B logging requires a workflow-injected WANDB_API_KEY.")
            self.run = wandb.init(
                project=logging["project"],
                group=logging["group"],
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=f"{logging['group']}-{os.environ.get('WORKFLOW_ID', output.name)}",
                job_type="mbpo-ppo",
                config=config,
                dir=str(output),
                mode="online",
                settings=wandb.Settings(console="off"),
            )
            for prefix, step in (("ppo", "updates"), ("mj", "iteration"), ("model", "round")):
                self.run.define_metric(f"{prefix}/{step}")
                self.run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/{step}")
            _write(
                {
                    "url": self.run.url,
                    "entity": self.run.entity,
                    "project": self.run.project,
                    "id": self.run.id,
                    "group": logging["group"],
                },
                output / "wandb_run.json",
            )
            print(f"W&B run: {self.run.url}", flush=True)

    def execute(self, *, smoke_only: bool = False) -> None:
        success = False
        try:
            config_path = self.output / "config.json"
            self._one(
                worker_command(self.config, config_path, self.output / "initial", "init", None),
                self.output / "init.log",
            )
            policy = self.output / "initial" / "policy_init.pt"
            probe = {**self.config, "policy_observation_corruption": False, "role": "probe_mj"}
            probe_path = self.output / "probe_config.json"
            _write(probe, probe_path)
            self._one(
                worker_command(probe, probe_path, self.output / "probe_mj", "probe_mj", policy),
                self.output / "probe_mj.log",
            )
            probe.update(role="probe_nerd", capture=str(self.output / "probe_mj" / "capture.pt"))
            _write(probe, probe_path)
            self._one(
                worker_command(probe, probe_path, self.output / "probe_nerd", "probe_nerd", policy),
                self.output / "probe_nerd.log",
            )
            self._coupled(smoke_config(self.config), "smoke", policy)
            if not smoke_only:
                self._coupled(self.config, "train", policy)
                self._evaluate()
            _write(
                {
                    "restore_check_passed": True,
                    "coupled_smoke_passed": True,
                    "smoke_only": smoke_only,
                    "training_updates": 0 if smoke_only else self.config["iterations"],
                    "flat_retention_gate_applied": False,
                },
                self.output / "workflow_result.json",
            )
            self._artifacts(smoke_only)
            success = True
        except BaseException as error:
            _write({"error": repr(error), "training_complete": False}, self.output / "workflow_failure.json")
            raise
        finally:
            if self.run is not None:
                try:
                    if not success:
                        self._artifacts(smoke_only=True)
                finally:
                    self.run.finish(exit_code=0 if success else 1)

    def _one(self, command: list[str], logfile: Path) -> None:
        print(json.dumps({"stage": logfile.stem, "command": command}), flush=True)
        with logfile.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    self._budget()
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f"{logfile.stem} failed with exit code{process.returncode}; see{logfile.name}.")
            finally:
                _stop(process)

    def _coupled(self, config: dict, name: str, policy: Path) -> None:
        root = self.output / name
        root.mkdir()
        settings = {**config, "run_dir": str(root)}
        processes, streams = {}, []
        try:
            for role in ("mj_collect", "nerd_train", "model_fit"):
                config_path = root / f"{role}_config.json"
                _write({**settings, "role": role}, config_path)
                command = worker_command(settings, config_path, root / f"{role}_out", role, policy)
                _write({"argv": command}, root / f"{role}_command.json")
                stream = (root / f"{role}.log").open("w")
                streams.append(stream)
                processes[role] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while any(process.poll() is None for process in processes.values()):
                self._budget()
                self._metrics(root, name)
                failed = {
                    role: process.returncode for role, process in processes.items() if process.poll() not in (None, 0)
                }
                if failed:
                    _write(failed, root / "FAILED")
                    raise RuntimeError(f"{name} MBPO worker failed: {failed}")
                time.sleep(5)
            self._metrics(root, name)
            _write({role: process.returncode for role, process in processes.items()}, root / "exit_codes.json")
            train = json.loads((root / "TRAINING_DONE.json").read_text())
            model = json.loads((root / "MODEL_FIT_DONE.json").read_text())
            if train["ppo_updates"] != config["iterations"] or model["optimizer_updates"] < 1:
                raise RuntimeError("Coupled MBPO did not complete its actual PPO/model optimizer updates.")
        finally:
            if any(process.poll() is None for process in processes.values()):
                _write({"reason": "coordinator stopped"}, root / "FAILED")
            for process in processes.values():
                _stop(process)
            for stream in streams:
                stream.close()

    def _evaluate(self) -> None:
        root = self.output / "evaluation"
        root.mkdir()
        policy = self.output / "train" / "checkpoints" / f"policy_{self.config['iterations']:04d}.pt"
        for seed in self.config["evaluation_seeds"]:
            config = {
                **self.config,
                "mode": "eval",
                "role": "evaluate",
                "contact_mdp": self.config["evaluation_contact_mdp"],
                "seed": seed,
                "env_seed": seed,
                "policy_sha256": file_sha256(policy),
                "updates": 0,
                "num_envs": self.config["evaluation_num_envs"],
                "eval_steps": self.config["evaluation_steps"],
                "init_at_random_ep_len": False,
                "terrain_seed": self.config["terrain_seed"] + 1000 + seed,
            }
            path = root / f"seed_{seed}.json"
            _write(config, path)
            command = [
                sys.executable,
                "-m",
                "isaaclab_neural.rl.rsl_rl.mbpo_worker",
                "--stage",
                "study-rl",
                "--backend",
                "mjwarp",
                "--terrain",
                "rough",
                "--scene-config",
                str(path),
                "--rl_study_config",
                str(path),
                "--dynamics",
                self.config["dynamics_path"],
                "--policy-snapshot",
                str(policy),
                "--num-envs",
                str(config["num_envs"]),
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
                "--output",
                str(root / f"seed_{seed}"),
            ]
            self._one(command, root / f"seed_{seed}.log")
            report = json.loads((root / f"seed_{seed}" / "report.json").read_text())
            if self.run is not None:
                self.run.log(_numeric_metrics(report["evaluation"], f"evaluation/seed_{seed}"))

    def _budget(self) -> None:
        if time.monotonic() - self.started > self.config["walltime_seconds"]:
            raise TimeoutError("Declared MBPO walltime budget exhausted.")
        if shutil.disk_usage(self.output).free < self.config["min_free_disk_gib"] * 2**30:
            raise RuntimeError("Insufficient free storage; stopping this workflow's workers.")
        if output_size(self.output) > self.config["max_output_gib"] * 2**30:
            raise RuntimeError("Declared MBPO output-size budget exhausted.")

    def _metrics(self, root: Path, phase: str) -> None:
        for filename, prefix in (("nerd_train.jsonl", "ppo"), ("model_fit.jsonl", "model")):
            path = root / filename
            if not path.exists():
                continue
            with path.open() as stream:
                stream.seek(self.offsets.get(path, 0))
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line.endswith("\n"):
                        self.offsets[path] = start
                        break
                    row = json.loads(line)
                    print(
                        json.dumps(
                            {
                                "phase": phase,
                                "role": prefix,
                                **{key: value for key, value in row.items() if not isinstance(value, (list, dict))},
                            }
                        ),
                        flush=True,
                    )
                    if self.run is not None:
                        metrics = _numeric_metrics(row, prefix if phase == "train" else f"smoke/{prefix}")
                        if prefix == "ppo" and phase == "train":
                            metrics["ppo/updates"] = row["ppo_updates"]
                        self.run.log(metrics)
        for path in sorted((root / "mj_stats").glob("stats_*.json")):
            if path in self.mj_seen:
                continue
            row = json.loads(path.read_text())
            self.mj_steps[phase] = self.mj_steps.get(phase, 0) + row["mj_physical_transitions"]
            row["confirmed_physical_transitions_total"] = self.mj_steps[phase]
            if self.run is not None:
                self.run.log(_numeric_metrics(row, "mj" if phase == "train" else "smoke/mj"))
            self.mj_seen.add(path)

    def _artifacts(self, smoke_only: bool) -> None:
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(f"mbpo-g1-rough-{self.run.id}", type="mbpo-logs")
        for path in self.output.rglob("*"):
            if (
                path.is_file()
                and path.suffix in {".json", ".jsonl", ".log"}
                and "wandb" not in path.relative_to(self.output).parts
            ):
                artifact.add_file(str(path), name=str(path.relative_to(self.output)))
        if not smoke_only and self.save_checkpoints:
            policy = self.output / "train" / "checkpoints" / f"policy_{self.config['iterations']:04d}.pt"
            model = Path(json.loads((self.output / "train" / "models" / "latest.json").read_text())["path"])
            artifact.add_file(str(policy), name="policy_final.pt")
            artifact.add_file(str(model), name="dynamics_final.pt")
        self.run.log_artifact(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset-file", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    preset = yaml.safe_load(args.preset_file.read_text())
    manifest = json.loads(args.input_manifest.read_text())
    config = validate_config(preset["experiment"])
    smoke_config(config)
    for name, path in (("dynamics", args.dynamics), ("reference", args.reference_cache)):
        if file_sha256(path) != manifest[name]["sha256"]:
            raise ValueError(f"Downloaded {name} differs from the approved immutable input.")
    config.update(
        dynamics_path=str(args.dynamics.resolve()),
        dynamics_sha256=manifest["dynamics"]["sha256"],
        reference_cache={**manifest["reference"], "path": str(args.reference_cache.resolve())},
        code_snapshot_sha256=manifest["code_snapshot_sha256"],
    )
    Workflow(config, args.output.resolve(), preset["wandb"]).execute(smoke_only=args.smoke_only)


def output_size(root: Path) -> int:
    """Return the size [bytes] of regular files under ``root`` while workers keep writing.

    Workers atomically rename ``*.partial`` files and delete consumed buffers, so a file listed by
    the scan may vanish before it is measured; such files are skipped instead of aborting training.
    """
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _numeric_metrics(value: dict, prefix: str) -> dict:
    output = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            output.update(_numeric_metrics(item, name))
        elif isinstance(item, (int, float)) and math.isfinite(item):
            output[name] = item
    return output


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _write(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2))
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
