# Allegro Cube Reorientation

Allegro NeRD predicts the fixed-base Allegro hand and the free DexCube together,
following the whole-scene approach of SO101 Keyboard and Franka Lift. It is not
the FlashNeRD two-way coupling, where NeRD predicts only the hand's free motion and
MJWarp solves the cube and all contacts.

All tasks derive from the core manager-based `Isaac-Reorient-Cube-Allegro`:

- 16 hand joints plus one `FREE` cube joint: 23 generalized positions, 22
  velocities and 22 generalized-force values per world;
- `dt=1/120 s` with two MJWarp substeps (240 Hz internal), decimation 4 (30 Hz
  policy), 10 s episodes;
- the core rewards, goal command, resets and `object_out_of_reach` termination,
  without domain randomization (the core `reset_only` events);
- Newton's collision pipeline instead of MuJoCo contacts, so the native contacts
  NeRD consumes are the ones that drove the teacher. MJWarp capacities are raised
  to `njmax=256` and `nconmax=128`: the core `njmax=80` overflowed (91 rows
  requested) with Newton contacts;
- 128 native contact tokens packed with `body_round_robin_pair_atomic`. A
  zero-action probe required up to 88 tokens; 64 truncated 65% of frames.

The cube USD reports an invalid `{1, 1, 1}` inertia and Newton substitutes a
small-sphere approximation. The core task has the same behavior, so teacher and
NeRD agree, but it is not the physical DexCube inertia.

## Registered task IDs

| Purpose | Task ID |
| --- | --- |
| Policy training and evaluation | `Isaac-Reorient-Cube-Allegro-Policy-v0` |
| Dataset generation (CUDA graphs disabled) | `Isaac-Reorient-Cube-Allegro-Dataset-Gen-v0` |
| NeRD inference/RL | `Isaac-Reorient-Cube-Allegro-NeRD-v0` |

## Teacher policy

No pretrained Allegro checkpoint exists. Train one on the policy task so its
contact pipeline matches data collection. `isaaclab_neural.train.train_control_policy`
registers the NeRD tasks and forwards the `isaaclab train` RSL-RL arguments; it
also runs under `torch.distributed.run` with `--distributed`:

```bash
export PYTHONPATH="$PWD/source/isaaclab_neural${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync python -m isaaclab_neural.train.train_control_policy \
  --task Isaac-Reorient-Cube-Allegro-Policy-v0 --num_envs 4096
```

## Data

`generate/scripts/generate_allegro_reorient.sh` records 600-frame trajectories
into `data/datasets/allegro-reorient-newton-contacts-v1/Allegro-Reorient/`. The
mixed recipe expects clean-policy, noisy-policy and zero-action splits, each with
a separately seeded validation file:

```bash
SCRIPT=./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_allegro_reorient.sh
POLICY=logs/rsl_rl/allegro_hand/<run>/model_<iter>.pt
$SCRIPT --sample-mode policy --policy-checkpoint "$POLICY" \
  --num-envs 1024 --num-transitions 6000000 --seed 0 --dataset-name dataset_policy_train.hdf5
$SCRIPT --sample-mode policy --policy-checkpoint "$POLICY" --policy-action-noise-std 0.1 \
  --num-envs 1024 --num-transitions 2000000 --seed 1 --dataset-name dataset_policy_noisy_train.hdf5
$SCRIPT --sample-mode action --zero-actions \
  --num-envs 1024 --num-transitions 2000000 --seed 2 --dataset-name dataset_zero_action_train.hdf5
```

Repeat each command with a new seed, a smaller transition count and the
corresponding `*_valid.hdf5` name. The core action term rescales actions to the
joint limits, so zero actions hold the hand at the middle of its joint ranges,
not at the default pose.

## Training

```bash
uv run --no-sync python -m isaaclab_neural.train.train \
  --task Isaac-Reorient-Cube-Allegro-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Allegro/transformer_reorient_mixed.yaml \
  --cfg-overrides "algorithm.eval.policy_checkpoint $POLICY" \
  --num-envs 256 --device cuda:0 --visualizer none
```

The recipe uses the Franka Lift model, losses and weighted mixed-source loading
with the SO101 throughput settings: the streaming trunk (`relative_lag` attention
spans), the frame cache, FP16 online inference, `inference_compile` and pinned,
non-blocking batch transfers, plus fused Adam. Training stays FP32. Like SO101,
`contact_filter: solver_active` keeps only penetrating native pairs; the filter is
applied to stored pairs at training time, so it can be ablated without recollection. As for SO101 Keyboard, `algorithm.eval.mode: allegro_policy` replaces
dataset rollouts: every fifth epoch rank 0 runs the teacher closed loop in the
training environment with the current weights, logs
`eval_closed_loop/mean_goals_per_episode` and `drop_rate`, and saves
`best_eval_model.pt` for the most goals per episode. Pass `--eval-interval 0` when
no teacher checkpoint is available. Keyboard's unique-transition sliding windows
and key head are not used. The SO101 Keyboard result (94.7% with the key head
versus 4.2% without) suggests evaluating a cube head if closed-loop results are
poor.

## OSMO pipeline

Two presets separate the teacher from NeRD, as for SO101 Keyboard.

**1. Teacher** (`allegro_reorient_teacher`, one GPU): the core Isaac Lab RSL-RL
configuration (8192 worlds, 24 steps per iteration, 5000 iterations) on the
Newton-contact policy task. RSL-RL uploads every saved checkpoint to the W&B project
`rl` of the API key's default entity, so submit with `--enable-wandb`. The run id is
`allegro-<8 hex digits>`, printed as `Teacher W&B run: wandb:<entity>/rl/allegro-...`;
set `teacher_policy.wandb_entity` or `--wandb-entity` to use another entity. The workflow
output also keeps the run directory and the MJWarp baseline `teacher_policy/baseline.json`.

```bash
./osmo_scripts/start.sh --preset allegro_reorient_teacher --enable-wandb --pool <osmo-pool>
```

**2. Stage the teacher.** Download the final `model_4999.pt` from W&B to
`pretrained/control_policy/allegro_reorient/model_4999.pt`. Without `params/agent.yaml`
beside it, loaders use the registered default RSL-RL configuration, which is the one the
teacher used. Submission refuses the NeRD preset if this file would not be packaged.

**3. Data and NeRD** (`allegro_reorient_mixed`, eight GPUs): evaluates the packaged
teacher on MJWarp first (`policy_evaluation.json`) and stops before collection below
`min_mean_goals_per_episode` (2.0). It then collects eight balanced shards concurrently,
one GPU each (`parallel_collection_gpus: 8`; the clean-policy data is split into three
2M-transition shards with distinct seeds), or reuses the dataset cache, and trains NeRD
with closed-loop evaluation against the same teacher. Finally, on the then idle GPU 0,
`throughput_benchmark` times `best_eval_model.pt` against native MJWarp with Newton and
with MuJoCo contacts, unoptimized and optimized NeRD, at 1024 and 8192 worlds, plus the
model-only sweep. Reports are in `throughput/`; a failed benchmark is recorded in
`throughput/summary.json` without failing the workflow.

```bash
./osmo_scripts/start.sh --preset allegro_reorient_mixed --pool <osmo-pool>
```

`allegro_reorient_teacher_smoke` (30 iterations, run id `allegro-smoke-<id>`) and
`allegro_reorient_mixed_smoke` (two GPUs, tiny shards, two epochs, its own dataset cache)
exercise the same stages; stage the smoke teacher's `model_29.pt` under
`pretrained/control_policy/allegro_reorient/smoke/` before submitting the second one.

## Throughput comparison

`eval/allegro_benchmark.py` mirrors `keyboard_benchmark`. Run each configuration on an
otherwise idle GPU and compare the JSON reports:

```bash
B="uv run --no-sync python -m isaaclab_neural.eval.allegro_benchmark"
# Model only: one full-window call on synthetic inputs over a batch sweep.
$B model --nerd-checkpoint "$CHECKPOINT" --autocast fp16 --report logs/allegro_bench/model_fp16.json
# End to end with the teacher: native MJWarp (CUDA graph), then NeRD.
$B env --checkpoint "$POLICY" --num-envs 8192 --report logs/allegro_bench/mjwarp_8192.json
$B env --checkpoint "$POLICY" --num-envs 8192 --mujoco-contacts --report logs/allegro_bench/mjwarp_mujoco_8192.json
$B env --checkpoint "$POLICY" --nerd-checkpoint "$CHECKPOINT" --num-envs 8192 \
  --report logs/allegro_bench/nerd_8192.json
```

`env_control_steps_per_s` and `physics_frames_per_s` are the headline numbers;
`breakdown_physics_fraction` and `nerd_profile_ms_per_control_step` locate the cost.
`--inference-precision fp32 --no-inference-compile --no-frame-cache` reproduces the
unoptimized NeRD path, and `allegro_reorient_policy` accepts the same overrides for the
matching closed-loop parity check.

A 1024-world smoke run with an untrained model on a GPU shared with other jobs (not a
valid measurement) showed the NeRD step dominated by native contact handling
(`adapter_update`, `adapter_encode_tokens`, `contact_prepare`), with the model forward
pass under a third of the solver time.

## Closed-loop evaluation

`eval/allegro_reorient_policy.py` runs the policy closed loop and reports goals
reached per completed episode and the drop rate (`object_out_of_reach`
terminations). Run it once with MJWarp and once with `--nerd-checkpoint`:

```bash
uv run --no-sync python -m isaaclab_neural.eval.allegro_reorient_policy \
  --checkpoint "$POLICY" --report logs/allegro_eval/mjwarp.json --num-envs 64
uv run --no-sync python -m isaaclab_neural.eval.allegro_reorient_policy \
  --checkpoint "$POLICY" --nerd-checkpoint "$CHECKPOINT" --report logs/allegro_eval/nerd.json --num-envs 64
```

Episodes still running after `--control-steps` are not counted. Select NeRD
checkpoints by this closed-loop comparison rather than validation loss.

**Validation status:** the teacher stage (with offline W&B checkpoint upload),
policy/noisy/zero-action collection, NeRD training with closed-loop evaluation and
both evaluator backends ran end to end on small single-GPU smoke runs. Neither OSMO
workflow has run, and no teacher policy, dataset or NeRD checkpoint has been qualified
yet.
