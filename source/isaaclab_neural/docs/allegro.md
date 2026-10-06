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
with the SO101 streaming trunk (`relative_lag` attention spans and the frame
cache). As for SO101 Keyboard, `algorithm.eval.mode: allegro_policy` replaces
dataset rollouts: every fifth epoch rank 0 runs the teacher closed loop in the
training environment with the current weights, logs
`eval_closed_loop/mean_goals_per_episode` and `drop_rate`, and saves
`best_eval_model.pt` for the most goals per episode. Pass `--eval-interval 0` when
no teacher checkpoint is available. Keyboard's unique-transition sliding windows
and key head are not used. The SO101 Keyboard result (94.7% with the key head
versus 4.2% without) suggests evaluating a cube head if closed-loop results are
poor.

## OSMO pipeline

`osmo_scripts/presets/allegro_reorient_mixed.yaml` runs everything in one
eight-GPU workflow:

1. If `policy_checkpoint` (`pretrained/control_policy/allegro_reorient/model.pt`,
   with `params/agent.yaml` beside it) is absent, `teacher_policy` trains the RSL-RL
   teacher with `torch.distributed.run` (8 ranks x 4096 worlds, 5000 iterations) and
   copies the run to `teacher_policy/` in the workflow output.
2. The teacher is evaluated closed loop on MJWarp (`teacher_policy/baseline.json`).
   Fewer than `min_mean_goals_per_episode` (2.0) goals per episode stops the workflow
   before data collection.
3. A newly trained teacher always regenerates the six datasets instead of reading
   the cache, because cached data may come from another policy.
4. NeRD trains on eight GPUs with closed-loop evaluation against the same teacher.

```bash
./osmo_scripts/start.sh --preset allegro_reorient_mixed --pool <osmo-pool>
```

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

**Validation status:** teacher training (also through a one-rank
`torch.distributed.run`), the teacher baseline gate, policy/noisy/zero-action
collection, NeRD training with closed-loop evaluation and both evaluator backends
ran end to end on small single-GPU smoke runs. The eight-GPU OSMO workflow has not
run, and no teacher policy, dataset or NeRD checkpoint has been qualified yet.
