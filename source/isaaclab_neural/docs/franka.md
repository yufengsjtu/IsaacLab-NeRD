# Franka NeRD

Franka NeRD uses the core Lift Menagerie Panda for both Reach and Lift. The two
scenes intentionally have different dynamics contracts:

- Reach is contact-free: one Panda per world, no table, 18 state values and 9
  generalized-force values.
- Lift predicts the Panda and one free cube together: the table is static
  collision geometry, native contacts use shared per-body tokens, and the model
  sees 31 state values and 15 generalized-force values.

Data, NeRD checkpoints and control policies from the previous Legacy-Panda recipe
are incompatible. Regenerate datasets and normalization statistics after changing
the robot, physics step or contact layout.

Run commands from the checkout root. When sharing another worktree's environment:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/.venv
export PYTHONPATH="$PWD/source/isaaclab_neural:$PWD/source/isaaclab_newton:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
```

## Required policy assets

Provision the control-policy checkpoints shown below separately before policy data
collection or closed-loop evaluation; installing the Python packages does not
supply these weights. Keep Reach's saved `params/agent.yaml` beside its checkpoint
so inference uses the original runner configuration, and retain `params/env.yaml`
for provenance. The OSMO packager includes these local assets when present. NeRD
training checkpoints, datasets, videos and run logs remain separate experiment
artifacts.

## Reach

`Isaac-Reach-Franka-Dataset-Gen-v0`, `Isaac-Reach-Franka-Policy-v0`, and
`Isaac-Reach-Franka-NeRD-v0` use `FRANKA_PANDA_LIFT_CFG`: the core Lift
Menagerie USD and its calibrated arm and mimic-gripper drives. The scene keeps
the Reach timing (`dt=1/60 s`, decimation 2), reward, command and 7-D
`q_default + 0.5 * action` policy contract. Dataset-Gen removes the table and
self-collisions. Random clips temporarily use held incremental joint targets;
policy and zero-action clips restore the policy action.

The policy used by the maintained dataset recipes is:

```text
pretrained/control_policy/franka_reach_menagerie/model_999.pt
```

Play it in the exact policy/data scene:

```bash
uv run --no-sync python -m isaaclab_neural.eval.play_control_policy \
  --task Isaac-Reach-Franka-Policy-v0 \
  --checkpoint pretrained/control_policy/franka_reach_menagerie/model_999.pt \
  --num_envs 16 --viz newton presets=newton_mjwarp
```

Generate the maintained splits locally:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_reach.sh \
  --num-envs 1024 --num-transitions 10000000 --dataset-name dataset_train.hdf5
```

The Reach YAMLs and OSMO presets use
`data/datasets/menagerie-60hz/Franka-Reach/` and cache names beginning with
`franka-reach-menagerie-60hz-`. For policy-only NeRD training:

```bash
./osmo_scripts/start.sh --preset franka_reach_policy_train \
  --dataset-cache-mode off --pool <osmo-pool>
```

### Small local recipes

The former `franka_reach_newton_native_smoke`, `franka_reach_policy_train_smoke`,
`franka_reach_mixed_smoke`, and `franka_reach_replay_smoke` OSMO presets were
removed. For a small training check, copy a maintained Reach preset into a local
YAML and pass it with `./osmo_scripts/start.sh --preset-file /path/to/local.yaml`.
Keep `resources.num_gpu` and `experiment.num_gpus` consistent, reduce the scene
replicas and transition budgets (including explicit per-split `num_transitions`),
and choose a separate `workflow.dataset_subdir` so the check does not replace a
formal dataset cache. For dataset playback, use the existing
`isaaclab_neural.eval.replay_franka_dataset` CLI directly.

## Lift

`Isaac-Lift-Cube-Franka-Dataset-Gen-v0` and
`Isaac-Lift-Cube-Franka-NeRD-v0` derive from core `Isaac-Lift-Franka`:

- Menagerie Panda with the core Lift 9-D relative-joint-position action;
- `dt=1/120 s`, decimation 4;
- one fixed 5 cm cube rather than heterogeneous shapes;
- a static 0.8 m x 1.5 m table with the same 0.255 m top as core Lift,
  extended downward to prevent full-gravity reset drops tunneling through;
- nominal full-gravity dynamics, without ADR or physics randomization;
- 64 Newton-native contact tokens packed with
  `body_round_robin_pair_atomic` in body-frame state mode.

The static table is deliberate. Core Lift's kinematic `RigidObjectCfg` table is
imported into Newton as a second free joint and would incorrectly become model
state. The core reset-bank ground-clearance check excludes `panda_link1`: the
asset's default primitive collider for that fixed-base link extends below the
ground plane, so including it rejects every reset draw. No USD collider variant
is required, keeping local and OSMO asset revisions compatible.

The official Newton Lift policy is stored at:

```text
pretrained/official/Isaac-Lift-Franka_newtonmjwarp_none_rsl_rl.pt
```

Play it in the exact data-generation scene:

```bash
uv run --no-sync python -m isaaclab_neural.eval.play_control_policy \
  --task Isaac-Lift-Cube-Franka-Dataset-Gen-v0 \
  --checkpoint pretrained/official/Isaac-Lift-Franka_newtonmjwarp_none_rsl_rl.pt \
  --num_envs 16 --viz newton presets=newton_mjwarp
```

The custom player skips policy export. Current `isaaclab play` tries to export
multi-observation-group RSL-RL actors before playback, which fails for this Lift
checkpoint. Add `--steps N --viz none` for a finite headless check, or
`--no-play-mode` to retain the training environment configuration.

Generate policy data:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_lift.sh \
  --sample-mode policy \
  --policy-checkpoint pretrained/official/Isaac-Lift-Franka_newtonmjwarp_none_rsl_rl.pt \
  --dataset-dir data/datasets/lift-core-policy-normal-force-v1 \
  --dataset-name dataset_train.hdf5 --num-envs 1024 --num-transitions 10000000
```

The maintained policy-data OSMO recipe generates 10M train, 1M policy-valid and
1M zero-action-valid transitions, then trains the token model:

```bash
./osmo_scripts/start.sh --preset franka_lift_policy_train \
  --dataset-cache-mode off --pool <osmo-pool>
```

The training config is
`source/isaaclab_neural/isaaclab_neural/train/cfg/Franka/transformer_lift.yaml`.
Do not mix previous Legacy/contrib Lift datasets with the core-Lift data.

The `franka_lift_mixed` preset uses 6M clean-policy, 2M noisy-policy and 2M
independently seeded zero-action training transitions with sampling weights
0.6/0.2/0.2. Policy noise has standard deviation 0.1 and does not add a new
[-1, 1] clipping operation to the core Lift action contract. All sources include
filtered normal-force labels. Policy/zero-action validation seeds and sizes match
the policy-only recipe; an additional noisy-policy validation split is held out.
The corresponding config is `train/cfg/Franka/transformer_lift_mixed.yaml`.

For policy/mixed comparisons, each task uses identical network, optimizer, batch
size and iteration budgets. Reach uses 400 epochs and Lift uses 1000 epochs, both
with 1250 iterations per epoch and a 2e-4 to 2e-5 linear learning-rate schedule.
Compare the common policy/passive validation splits, not unlike `exp_trajectory`
labels (Reach mixed uses random actions for that additional validation split).

### CUDA-safe dataset loading

NeRD training creates the CUDA simulation before iterating its DataLoaders.
Multi-process loaders therefore use `spawn`, not `fork`, to avoid inheriting
Warp CUDA allocations that a worker cannot safely release. This does not change
the process-wide multiprocessing start method.

The formal Lift recipes use eager loading with `num_data_workers: 0` for both
training and validation. Their large NumPy arrays stay in the main rank process,
avoiding per-worker copies under `spawn`. Loading throughput and random sampling
order can differ from the former eight-worker recipes; dataset weights, batch
size, optimizer, physics and training budgets are unchanged. If enabling workers
for another recipe, account for dataset serialization and memory usage.

### Original-coordinate rollout evaluation

Lift presets use `algorithm.eval.preserve_source_origins: True` and 1024 scene
replicas, matching collection. The evaluator verifies `source_env_id` and
`env_origin` against the runtime grid, routes every window to its original slot,
and splits windows that share a source slot into separate batches. Unused slots
receive reset-state padding; only requested windows are validated and scored,
and results retain their original sample order. No recorded state or contact
point is translated. Training batch size and model architecture are unchanged.

This mode cannot be combined with `rebase_env_origins` or terrain-patch restoration.
It currently requires native contact tokens. Smaller local datasets must use
their own original replica count, not the formal 1024-grid. Merely increasing
`--num-envs` without source-slot routing does not preserve coordinates.

Strict contact tolerances remain unchanged. Original-coordinate routing removes
the rebasing error but does not establish cross-GPU collision reproducibility;
validate the recorded contact sets on the intended runtime before resubmission.

### Filtered finger normal-force prediction

Dataset-Gen records `finger_object_normal_forces_w` with shape `[B, T, 2, 3]`,
right finger then left finger, in world axes [N]. Each label is the sum of normal
contact-force vectors exerted by the cube on that finger during the transition
`states[t] -> next_states[t]`. Friction and contacts with the table or other bodies
are excluded. Labels are read after the solve, are not clipped, and carry explicit
order, frame, object-filter and timing metadata. This contract requires a fixed-base
robot so pre-step and post-step root axes agree.

`network.finger_object_normal_force_head.enabled` adds a separate 6-output force
head and two contact logits sharing the state model's trunk. Training rotates
labels into robot-root axes, computes separate force normalization statistics,
and adds normalized force MSE plus contact BCE, weighted by
`algorithm.loss.finger_object_normal_force.weight`. The BCE target is whether each
finger's measured force norm exceeds 0.01 N. At inference, non-positive contact
logits produce exactly zero force; otherwise the regressed vector is used. No
analytic contact flag replaces the predicted force. Logs include physical-unit
force RMSE, active-contact RMSE, per-finger RMSE and contact precision/recall at the
policy reward threshold of 0.01 N. Force labels are supervision only, never model inputs.

After each neural physics step the predicted vectors are rotated back to world
axes and published to the existing sensors' filtered normal-force buffers. The
unchanged observation term handles root-frame conversion, clipping to [-20, 20] N,
and five-frame history; rewards read the unclipped vectors. Partial reset clears
only the reset environments' predictions and history. Only the filtered normal-force
matrix and its history are supported; aggregate forces, friction and contact points
are not predicted. Feature/KV-cache inference is currently rejected for this head
rather than silently omitting force updates.

**Migration:** Regenerate all splits under `lift-core-policy-normal-force-v1` and
use the new `franka-lift-core-120hz-policy-normal-force-v1` OSMO cache. Old unlabeled
data is rejected for force-head training. Checkpoints save and validate the force
contract and normalization; state-only checkpoints cannot supply Lift force
observations and fail explicitly on deployment. The existing resultant body-force
head is separate and is not a substitute for filtered normal forces.

**Validation boundary:** A training smoke verifies connectivity, not usable force
accuracy. In particular, small nonzero regression errors can trigger the 0.01 N
reward threshold during no-contact phases. Validate force errors and contact
precision/recall on independent trajectories, then compare closed-loop behavior
against MJWarp before using a checkpoint for Lift policy evaluation or RL training.

## Lift control-step recipe (SO101/Allegro pipeline)

`franka_lift_control_step` records reset-safe control-step clips from `Isaac-Lift-Cube-Franka-Dataset-Gen-v0` with
the official policy, including filtered finger-object normal-force labels, and trains
`train/cfg/Franka/transformer_lift_control_step.yaml` (one 33 ms control step per prediction, joint-target input,
finger-force head, unique-transition sliding windows, GPU-resident data, FP16 autocast). The native pose success rate
from the OSMO teacher gate becomes the checkpoint-selection reference of the closed-loop `policy_rollout` evaluation.
Run `franka_lift_control_step_smoke` first.

## Registered task IDs

| Purpose | Reach | Lift |
| --- | --- | --- |
| Policy environment | `Isaac-Reach-Franka-Policy-v0` | core task adapted by Dataset-Gen |
| Dataset generation | `Isaac-Reach-Franka-Dataset-Gen-v0` | `Isaac-Lift-Cube-Franka-Dataset-Gen-v0` |
| NeRD inference/RL | `Isaac-Reach-Franka-NeRD-v0` | `Isaac-Lift-Cube-Franka-NeRD-v0` |
