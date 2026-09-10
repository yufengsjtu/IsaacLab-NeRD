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

The dataset-generation and NeRD environments remove the Newton table
articulation so each world contains only the Panda. Self-collisions are off.
Fingers stay in the USD (9 DoF); finger joint-force sampling limits are zero.
Random splits command ``q_target = clip(q + scale a, soft limits)`` with a
per-joint ``scale`` of ``effort_limit / kp`` so ``|a| = 1`` saturates
implicit PD at rest (shoulder ``87/80`` rad, forearm ``12/80`` rad). Each
absolute target is held for 20 physics frames (1/6 s at 120 Hz). The
dataset-generation Panda uses ``soft_joint_pos_limit_factor=0.8`` as the
walk envelope. Policy and zero-action splits restore
``q_default ± 0.5 rad``. Each rollout starts from the Reach env reset
(``q_default`` scaled by ``U(0.5, 1.5)``, clamped to those soft limits,
zero joint velocity). Mapping random actions onto joint limits in one shot
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
./isaaclab.sh -p -m isaaclab_neural.train.train \
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
  --dataset ./data/datasets/Franka-Reach/dataset_valid.hdf5 \
  --num-envs 1 --loop --playback-hz 30 --viz newton
```

Use ``dataset_policy_valid.hdf5`` (or any other Franka split) the same way.
``--trajectory-index 0`` pins one clip. ``Franka-Lift`` files open
``Isaac-Lift-Cube-Franka-Dataset-Gen-v0``. Do not pass a NeRD task; the
script launches Dataset-Gen with ``presets=newton_mjwarp``. Replay turns
CUDA graph off and refreshes Newton FK after each written frame so the
arm pose matches the HDF5 clip.

The maintained training config expects datasets under `data/datasets/Franka-Reach/`.

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
write chunks. Trajectory length is 100. Data scale is ten million training
transitions and one million validation transitions. Training launches with
`torchrun` on eight GPUs. Reach stays contact-free
(`num_contacts_per_env=0`). Training data are incremental joint-position
actions (per-joint steps that saturate implicit PD at rest, held for 20
physics frames) starting from the Reach env reset. The preset also records a
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
physics frames (~2.0 s) with a 1.5 s EE resample, so each clip records
one start-to-stop and the first 0.5 s of the next Reach. The training
config is ``Franka/transformer_reach_policy.yaml`` (400 epochs, Transformer
``n_embd=384`` matching Anymal). Use a
distinct dataset cache subdirectory so this data is not mixed with the
incremental walk:

```bash
./osmo_scripts/start.sh \
  --preset franka_reach_policy_train \
  --pool <osmo-pool>
```

## Franka Lift

Lift concatenates the Panda and the free cube into one generalized-state vector
per world. The table is a static cuboid so Newton does not add a third
articulation. Contacts stay on (`num_contacts_per_env=64`).

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
