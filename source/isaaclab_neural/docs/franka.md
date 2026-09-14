# Franka Tutorial

Franka NeRD covers two manipulation scenes that are **not** drop-in copies of
the stock Isaac Lab tasks.

- Reach is contact-free: one Panda per world, no table articulation, and
  `num_contacts_per_env=0`.
- Lift is contact-aware: the Panda and cube share one Newton world, the table
  is a static collider, and native contacts are packed like Anymal-C.

Do not generate data from `Isaac-Reach-Franka-v0` or `Isaac-Lift-Cube-Franka-v0`
for NeRD. Those scenes are not compatible with the solver layout used here.

## Franka Reach

The default `Isaac-Reach-Franka`, Dataset-Gen and NeRD configurations now use
Legacy Panda, matching the historical September 5 Reach policy. Reach pins the
Isaac 6.0 `Robots/FrankaEmika/Legacy/panda_instanceable.usd` revision used by that
checkpoint; it does not change the global asset root. Override the task's
`env.scene.robot.spawn.usd_path` explicitly if using an equivalent local mirror.
The arm uses Kp=80 N*m/rad, Kd=4 N*m*s/rad and armature=0.001 kg*m^2, with effort
limits 87 N*m on joints 1-4 and 12 N*m on joints 5-7. Both fingers use Kp=2000 N/m,
Kd=100 N*s/m and effort limits 200 N, without the Menagerie passive mimic drive.

The physics step is `dt=1/60 s`, decimation is 2, and native Newton MJWarp uses
one substep (`njmax=50`, `update_data_interval=1`). Control remains 30 Hz.
Dataset-Gen and NeRD still remove the table and disable self-collisions, with
`num_contacts_per_env=0`; this contact-free contract intentionally differs from
ordinary Reach. Lift, UR10 and the separate OSC task retain their existing
configurations. Ordinary Reach no longer exposes the temporary Menagerie
asset/drive experiment profiles.

The historical checkpoint's saved reward configuration already used continuous
tracking. Its active terms are position error (-0.2), position tanh reward
(+0.1, scale 0.1 m), orientation error (-0.1), action-rate and all-joint velocity
penalties. The latter two start at -0.0001 and change after the 4500-step
curriculum threshold to -0.005 and -0.001. There is no action-magnitude penalty,
terminal success reward, or success termination. Episodes end at 12 s; ordinary
Reach goals resample every 4 s. Do not enable upstream success termination when
reproducing this checkpoint. Pose success thresholds remain metrics only.
The joint-position action is `q_default + 0.5 * action`; the neural solver
replaces MJWarp, not the robot, drives or policy action contract.

**Migration:** Menagerie data/models and Legacy data recorded at another time
step or with zero `joint_f` must not be silently reused. The local generation
wrapper and both Reach training YAMLs use the isolated root
`data/datasets/legacy-60hz/Franka-Reach/`. All Reach OSMO recipes use new
`franka-reach-legacy-60hz-*` cache names. Generate all splits, recompute statistics,
and train a matching NeRD model. Existing data, checkpoints and experiment
logs are retained, not deleted. The bundled `model_999.pt` is a Legacy
control policy trained on this default Reach configuration (RSL-RL PPO,
`franka_reach_legacy/2026-09-13_23-01-37_legacy_baseline`); it is not a NeRD
dynamics model. Its `params/agent.yaml` must stay next to the checkpoint.

Run commands from this worktree's root. When reusing a uv environment from
another checkout, set `UV_PROJECT_ENVIRONMENT` to its absolute path and use
`--no-sync`. NeRD is not installed by the root uv project; select this source tree:

```bash
export PYTHONPATH="$PWD/source/isaaclab_neural:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_newton:$PWD/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync isaaclab play --rl_library rsl_rl --task Isaac-Reach-Franka \
  --checkpoint pretrained/control_policy/franka_reach/model_999.pt \
  --num_envs 1 --seed 0 --train_env_cfg --viz newton --real-time \
  physics=newton_mjwarp
```

For a short local smoke, generate the four splits below with `--num-envs 4`
and `--num-transitions 800`, using a separate `--dataset-dir`. Point all training
and evaluation dataset paths at that directory, then override
`algorithm.num_epochs`, `algorithm.num_iters_per_epoch`, and
`algorithm.num_valid_batches` for a short run. Such a checkpoint validates the
pipeline only; it is not a trained dynamics model for deployment.

Fingers stay in the USD (9 DoF); finger joint-force sampling limits are zero.
Random splits command ``q_target = clip(q + scale a, soft limits)`` with a
per-joint ``scale`` of ``effort_limit / kp`` so ``|a| = 1`` saturates
implicit PD at rest unless the target is clamped by a soft limit. With
`IncrementalJointPositionActionCfg.scale=None`, these steps are computed from
the resolved articulation drives at initialization, not from Legacy PD constants.
Explicit numeric/dict scales still work; `FRANKA_REACH_DATASET_INCREMENT_SCALE`
retains the old Legacy values for callers reproducing that recipe. Each
absolute target is held for 20 physics frames (1/3 s at the default 60 Hz). The
dataset-generation Panda uses ``soft_joint_pos_limit_factor=0.6`` as the
walk envelope; with the Legacy drives, joint 2 overshoots held targets by up to
about 0.45 rad, so the previous 0.8 factor let roughly 17% of random frames cross
the USD joint limits. The 0.6 factor reduces this to about 6% but does not
eliminate it, and the sampler still does not reject such frames.
Random splits independently sample all seven arm joint positions
uniformly within these soft limits at every reset, including joints whose default
position is zero. Joint velocities start at zero; fingers start at their default
open position clamped to the soft limits. Because every arm action term covers
only ``panda_joint.*``, the finger drive target stays at the zero-initialized
buffer, so the fingers close from 0.04 m toward 0 during each clip in all splits
and in the RL task alike. The local wrapper and random-data OSMO
presets use 200-physics-frame clips (about 3.33 s), giving ten held targets per
clip instead of five. The generic Python generator still defaults to 100 frames;
pass ``--trajectory-length 200`` when calling it directly.
Policy and zero-action splits restore both the ``q_default ± 0.5 rad`` action
and the original Reach reset (``q_default`` scaled by ``U(0.5, 1.5)``,
clamped to soft limits, with zero joint velocity). The policy-training OSMO
presets retain their 240-frame clips. Mapping random actions onto joint limits in one shot
saturates implicit PD on MJWarp. Dataset-Gen resamples the EE pose every
1.5 s. Policy-mode collection ticks command and observation managers each
control interval without calling ``env.step``, so the Reach policy sees
the current joints and a new goal mid-clip on trajectories longer than
1.5 s. ``Isaac-Reach-Franka-NeRD-v0``
keeps the 4 s Reach command for closed-loop eval. Partial env resets copy
the current post-reset state into Transformer history instead of zeros, and
direct ``joint_q`` writes invalidate Newton FK before ``forward``.
Implicit Dataset-Gen actuators still apply PD inside MJWarp, so Newton
``control.joint_f`` is zero during the analytic step. After that step,
recording copies ``applied_effort`` (clipped shadow PD, all 9 DoF including
fingers) into HDF5 ``joint_f``. ``Isaac-Reach-Franka-NeRD-v0`` copies the
same telemetry into ``control.joint_f`` after actuators for neural-model
input only. Regenerate datasets recorded with zero ``joint_f`` and retrain;
do not evaluate old zero-``joint_f`` checkpoints closed-loop.

Generate the training split:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_reach.sh \
  --dataset-name dataset_train.hdf5 \
  --num-transitions 100000
```

Generate validation data with a different name and seed:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_reach.sh \
  --dataset-name dataset_valid.hdf5 \
  --num-transitions 100000 \
  --seed 1
```

Generate the passive validation split used by the config:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_reach.sh \
  --dataset-name dataset_passive_valid.hdf5 \
  --num-transitions 100000 \
  --zero-actions \
  --seed 2
```

Generate the policy validation split on the Dataset-Gen scene (not
`Isaac-Reach-Franka-v0`):

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_reach.sh \
  --dataset-name dataset_policy_valid.hdf5 \
  --num-transitions 100000 \
  --seed 3 \
  --sample-mode policy \
  --policy-checkpoint pretrained/control_policy/franka_reach/model_999.pt
```

Train:

```bash
uv run --no-sync python -m isaaclab_neural.train.train \
  --task Isaac-Reach-Franka-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Franka/transformer_reach.yaml
```

Evaluate a checkpoint. Closed-loop policy eval needs both the NeRD
dynamics checkpoint and the Reach RSL-RL policy. Use a model trained after
shadow-PD ``joint_f`` recording; old zero-``joint_f`` checkpoints are
out of distribution once overlay is on.

```bash
PYTHONPATH="$(pwd)/source/isaaclab_neural:$(pwd)/source/isaaclab_newton:$(pwd)/source/isaaclab:$(pwd)/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}" \
  ./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Reach-Franka-NeRD-v0 \
  --checkpoint /path/to/best_eval_model.pt \
  --policy-checkpoint pretrained/control_policy/franka_reach/model_999.pt \
  --num-envs 16 \
  --num-steps 100 \
  presets=newton_mjwarp
```

Add ``--video`` to write an mp4 under ``./videos/eval`` (Isaac Lab 3
``VideoRecorderCfg``, not Gymnasium ``RecordVideo``). NeRD eval captures
Newton GL; do not rely on the RL ``--video`` Kit injection, which starts
Omniverse RTX and can SIGSEGV at Kit startup on hybrid NVIDIA+Intel
hosts. ``--video`` alone is headless (mp4 only). Pass ``--viz newton``
or ``--viz newton_gl`` to open a live window while recording; ``--vis``
is accepted as an alias of ``--viz``. Match ``--video-length`` to
``--num-steps`` for one clip of the whole rollout. Do not pass
``--viz none``. Requires ``moviepy`` (1.x or 2.x).

```bash
PYTHONPATH="$(pwd)/source/isaaclab_neural:$(pwd)/source/isaaclab_newton:$(pwd)/source/isaaclab:$(pwd)/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}" \
  ./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Reach-Franka-NeRD-v0 \
  --checkpoint /path/to/best_eval_model.pt \
  --policy-checkpoint pretrained/control_policy/franka_reach/model_999.pt \
  --num-envs 16 \
  --num-steps 100 \
  --video --video-length 100 --video-dir ./videos/eval \
  presets=newton_mjwarp
```

Replay collected HDF5 clips kinematically (no policy or physics re-sim).
The script writes ``data/states`` into Dataset-Gen Newton. Random, policy,
and zero-action splits all work. Clips do not store the EE pose command.

```bash
PYTHONPATH="$(pwd)/source/isaaclab_neural:${PYTHONPATH}" ./isaaclab.sh -p \
  -m isaaclab_neural.eval.replay_franka_dataset \
  --dataset ./data/datasets/legacy-60hz/Franka-Reach/dataset_valid.hdf5 \
  --num-envs 1 --loop --playback-hz 30 --viz newton
```

Use ``dataset_policy_valid.hdf5`` (or any other Franka split) the same way.
``--trajectory-index 0`` pins one clip. ``Franka-Lift`` files open
``Isaac-Lift-Cube-Franka-Dataset-Gen-v0``. Do not pass a NeRD task; the
script launches Dataset-Gen with ``presets=newton_mjwarp``. Replay turns
CUDA graph off and refreshes Newton FK after each written frame so the
arm pose matches the HDF5 clip.

The maintained training config expects datasets under `data/datasets/legacy-60hz/Franka-Reach/`.

Verify Dataset-Gen ``joint_f`` wiring (analytic Newton, no NeRD checkpoint):

```bash
./source/isaaclab_neural/isaaclab_neural/eval/scripts/verify_franka_joint_f_wiring.sh
```

For remote execution, use the `franka_reach_newton_native` OSMO preset:

```bash
./osmo_scripts/start.sh \
  --preset franka_reach_newton_native \
  --pool <osmo-pool>
```

The preset matches the Anymal-C rough OSMO schedule: 8 L40S GPUs, 80
CPUs, 512Gi memory, 512Gi storage, `presets=newton_mjwarp`, 1024
generation environments, 64 training environments, and 819200-transition
write chunks. Trajectory length is 200. Data scale is ten million training
transitions and one million transitions per validation split. Training launches with
`torchrun` on eight GPUs. Reach stays contact-free
(`num_contacts_per_env=0`). Training data are incremental joint-position
actions (per-joint steps that saturate implicit PD at rest, held for 20
physics frames) starting from uniformly sampled arm positions within the soft limits.
The random-data presets now use separate caches, ``franka-reach-legacy-60hz-wide-reset-200``
and ``franka-reach-legacy-60hz-wide-reset-200-smoke``. Regenerate random splits and recompute
training statistics rather than reusing the old reset-distribution cache. For local
runs, use a fresh dataset directory and update the training paths, or move the old
splits aside before regeneration. The preset also records a
policy validation split from
`pretrained/control_policy/franka_reach/model_999.pt` on the Dataset-Gen
scene using the Reach RL action map. Run it from this worktree so the
packaged source includes the Franka environments and that checkpoint.
Franka OSMO presets pull
`docker.io/rowany28070/isaac-lab-nerd:g1-develop` and overlay the uploaded
Isaac Lab tree, matching the G1 develop image that already has
``isaaclab_tasks.core`` and current Newton.

To train on Reach-policy trajectories instead of the incremental random
walk, use ``franka_reach_policy_train``. Train, same-distribution valid,
and the 10-step eval split all record ``model_999.pt`` with
``q_default ± 0.5``; the passive split stays zero-action. Clips are 240
physics frames (~4.0 s) with a 1.5 s EE resample, so each clip can record
multiple target changes. The training
config is ``Franka/transformer_reach_policy.yaml`` (400 epochs, Transformer
``n_embd=384`` matching Anymal). Use a
distinct dataset cache subdirectory so this data is not mixed with the
incremental walk:

```bash
./osmo_scripts/start.sh \
  --preset franka_reach_policy_train \
  --pool <osmo-pool>
```

### Legacy PPO training

Ordinary Reach retains only the historical Legacy asset/drive recipe. The
experimental Menagerie and response-tuning selectors, including the redundant
`legacy` selector, have been removed. Do not pass those old `presets=` values.
Historical experiment logs, checkpoints and datasets remain available, but a
Menagerie checkpoint is not compatible with this default simulation merely
because its tensor dimensions match. The standard controller and physics
selectors are independent and remain available.

Run a fresh PPO training job using the registered agent configuration:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka \
  --num_envs 4096 --max_iterations 1000 --seed 42 \
  --experiment_name franka_reach_legacy --run_name legacy_baseline \
  --logger tensorboard --viz none physics=newton_mjwarp
```

No drive or reward overrides are needed. This does not resume a checkpoint.
For a launch smoke, use 64 environments and 2 iterations under a separate
experiment name. Keep the standard `joint_pos` controller for this recipe;
IK controller presets change the action contract and may change gravity handling.

## Franka Lift

Lift uses `isaaclab_tasks.contrib.lift.config.franka.joint_pos_env_cfg.FrankaCubeLiftEnvCfg`
as its baseline, not the Reach task. Both Dataset-Gen and NeRD retain
the parent's Legacy Panda, implicit PD drives, self-collisions, cube, end-effector
frame sensor, observations, actions, rewards, commands, reset events and
terminations. The control period is 0.02 s (`dt=0.01 s`, decimation 2), and
episodes time out after 5 s or terminate when the cube falls below -0.05 m.
The arm action is `q_default + 0.5 * action`; the eighth action opens/closes the
gripper. Unlike Reach, successful lifting does not itself terminate an episode.

Lift concatenates the Panda and the free cube into one generalized-state vector
per world (16 position coordinates, 15 velocity coordinates and 15 force entries).
The table remains a static 0.9 × 1.3 × 0.05 m cuboid so Newton does not add a third
articulation. Its top surface is at z=-0.003 m, matching the standard
SeattleLabTable collision surface, with the center at z=-0.028 m and an XYZW
identity quaternion. This is a simplified support surface, not a replacement of
the standard table's full geometry. Contacts stay on (`num_contacts_per_env=64`),
using the existing Newton-native collision pipeline and 0.01 m shape margin.

**Migration:** The previous cuboid top was at z=+0.025 m. Under the configured
Newton solver, a resting cube then settled above the standard 0.04 m lifting
threshold and incorrectly earned continuous lifting reward without a grasp.
The support-height correction changes dynamics and reward outcomes: regenerate
Lift datasets in fresh directories, recompute statistics, retrain NeRD dynamics,
and revalidate or retrain existing Lift policies. Old checkpoints are not
automatically rejected. Robot/PD settings and the standard reward thresholds
were not adjusted to hide the geometry mismatch.

A Newton regression test checks that the cube remains supported but earns no
lifting reward after settling with a stationary arm and open gripper. This does
not establish successful grasping or trained NeRD closed-loop accuracy.

The local Legacy asset audit found authored link density (1000 kg/m³), but no
explicit link mass, diagonal inertia or center of mass. Newton derives these
from collision geometry. The cube authors mass 0.216 kg; its resolved diagonal
inertia is 8.2944e-5 kg·m², consistent with a uniform 0.048 m cube. With the same
Newton solver, the standard Lift scene and the adapted scene resolve the same
robot/cube masses and inertias. These are not measured Panda inertial parameters;
positive final values do not establish physical fidelity. The importer warns
while resolving incomplete USD mass properties and then applies its fallback.

A two-environment action audit confirmed `+1` opens the gripper to approximately
0.04 m per finger and `-1` closes it to approximately 0.00063 m with no object
between the fingers. After physics updates, the frame sensor agrees with
`panda_hand` plus its 0.1034 m local-Z offset to below 1e-6 m. Immediately after
the first reset, the native frame-transform output was still zero until physics
updated it; do not use that initial sample as a validated end-effector pose.
Dropping one cube below the threshold reset only that environment, and both
environments timed out after 250 control steps. A preliminary differential-IK
approach/close/lift/hold probe did not lift either cube and had substantial pose
tracking error. This does not isolate friction or contact quality from controller
error; scripted grasping and hold/slip validation remain required before a
large Lift training run.

Random joint-position actions are a coverage starting point. They do not
reproduce successful grasps. Use `--sample-mode policy` later if you have a
lift policy checkpoint.

Generate the training split:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_lift.sh \
  --dataset-name dataset_train.hdf5 \
  --num-transitions 100000
```

Generate validation and zero-action splits with different names and seeds:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_lift.sh \
  --dataset-name dataset_valid.hdf5 \
  --num-transitions 100000 \
  --seed 1
```

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_franka_lift.sh \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --num-transitions 100000 \
  --zero-actions \
  --seed 2
```

Train:

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Lift-Cube-Franka-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Franka/transformer_lift.yaml
```

If dataset generation raises `Expected limit with N entries`, the Newton
generalized-state size does not match the `Franka-Lift` arrays in
`isaaclab_neural.utils.commons`. Measure `dof_q_per_env`, `dof_qd_per_env`, and
`joint_f_dim` from the solver and update those arrays.

## Registered task ids

| Role | Reach | Lift |
| --- | --- | --- |
| Dataset generation | `Isaac-Reach-Franka-Dataset-Gen-v0` | `Isaac-Lift-Cube-Franka-Dataset-Gen-v0` |
| NeRD inference / RL | `Isaac-Reach-Franka-NeRD-v0` | `Isaac-Lift-Cube-Franka-NeRD-v0` |
