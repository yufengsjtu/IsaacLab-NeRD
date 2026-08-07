# IsaacLab-NeRD RL Policy Learning

Train and play RSL-RL policies on NeRD-backed Isaac Lab environments.

NeRD replaces the analytical Newton integrator. The Isaac Lab MDP managers
(observations, actions, rewards, terminations) still own the RL loop.

## Flat Anymal-C (v1)

Registered task:

```text
Isaac-Velocity-Flat-Anymal-C-NeRD-v0
```

This task is the single flat Anymal-C NeRD entry point for both dynamics
closed-loop evaluation and RSL-RL policy learning. Relative to the stock
upstream flat Anymal-C velocity task it:

- Disables ``push_robot``.
- Raises tracking reward weights to ``track_lin_vel_xy_exp=2.0`` and
  ``track_ang_vel_z_exp=1.0`` (stock flat uses ``1.0`` / ``0.5``).
- Geometry-contact terms (``feet_air_time``, ``undesired_contacts``, and
  ``base_contact``): **on** only
  when ``contact_mode=newton_native``. Flat defaults to ``fixed_ground`` (off);
  ``--contact-mode newton_native`` re-enables them after CLI overrides.
  They use signed native-contact separation directly; NeRD does not synthesize
  or report constraint-solver contact forces.
- Points ``rsl_rl_cfg_entry_point`` at ``AnymalCFlatNeRDPPORunnerCfg``
  (experiment ``anymal_c_flat_nerd``; ``clip_actions`` unset)
- Default physics is ``fixed_ground``

## Validated flat recipe

Reproduced successfully on ``contact_encoder_set`` (runs
``2026-08-07_15-38-10`` and ``2026-08-07_16-58-45``):

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path <native_dynamics.pt> \
  --contact-mode newton_native --num-contacts-per-env 64 \
  --num_envs 4096 --max_iterations 500 --headless \
  presets=newton_mjwarp
```

Required MDP (already set in ``NerdAnymalCFlatEnvCfg``):

- ``push_robot`` disabled
- tracking weights ``2.0`` / ``1.0``
- geometry contact rewards/termination on under ``newton_native``
- 48-D stock flat observations; ``clip_actions`` unset

Late-training metrics to expect (~iter 500):

| metric | typical |
|---|---:|
| ``Metrics/success_rate`` | ~0.99–1.0 |
| ``Metrics/base_velocity/error_vel_xy`` | ~0.11 |
| ``Metrics/base_velocity/error_vel_yaw`` | ~0.10 |
| mean episode length | ~980–1000 |
| ``Episode_Termination/base_contact`` | ~0.02 |

Play the final checkpoint:

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.play \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path <native_dynamics.pt> \
  --contact-mode newton_native --num-contacts-per-env 64 \
  --checkpoint logs/rsl_rl/anymal_c_flat_nerd/<run>/model_499.pt \
  --num_envs 16 --num_steps 1000 \
  presets=newton_mjwarp
```

## Prerequisites

1. Install the package:

```bash
./isaaclab.sh -p -m pip install -e source/isaaclab_neural --no-deps
```

2. Provide a trained NeRD dynamics checkpoint. For the **validated** flat RL
   recipe use a ``newton_native`` Transformer checkpoint (64 contacts). Example
   placeholder path:

```text
./data/trained_models/Anymal-C-Native/nn/final_model.pt
```

A ``fixed_ground`` dynamics checkpoint is fine for FG-only experiments, but it
is not the validated locomotion recipe below.

Train dynamics first with `isaaclab_neural.train.train` if needed (see
[train.md](train.md)).

## Train

Prefer the [Validated flat recipe](#validated-flat-recipe) above
(``newton_native``, 64 contacts, 4096 envs).

Default-task smoke (uses env default ``fixed_ground`` unless you override):

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path ./data/trained_models/Anymal-C/nn/final_model.pt \
  --num_envs 64 \
  --max_iterations 5 \
  --headless \
  presets=newton_mjwarp
```

Logs write under:

```text
logs/rsl_rl/anymal_c_flat_nerd/<timestamp>/
```

Optional: the unified CLI also imports NeRD tasks, so the following can work
once a NeRD dynamics checkpoint path is configured on the env (prefer the
dedicated module so `--neural-model-path` is available):

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  presets=newton_mjwarp
```

## Play

Match the dynamics ``contact_mode`` used at train time. Validated example:

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.play \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path ./data/trained_models/Anymal-C-Native/nn/final_model.pt \
  --contact-mode newton_native --num-contacts-per-env 64 \
  --checkpoint logs/rsl_rl/anymal_c_flat_nerd/<run>/model_499.pt \
  --num_envs 16 \
  --num_steps 200 \
  --headless \
  presets=newton_mjwarp
```

## Profiling contact modes

``newton_native`` RL steps are typically slower than ``fixed_ground`` because each
step runs Newton's full collision pipeline, packs up to 64 contact slots, and
feeds a wider transformer input. Segmented timing is opt-in:

```bash
NERD_STEP_PROFILE=1 ./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path <dynamics.pt> \
  --contact-mode newton_native --num-contacts-per-env 64 \
  --num_envs 256 --max_iterations 2 --headless \
  presets=newton_mjwarp
```

Compare modes end-to-end (isolated workers; writes JSON). ``--compare`` runs
stock ground-truth MJWarp (``Isaac-Velocity-Flat-Anymal-C-v0``), NeRD
``fixed_ground``, and NeRD ``newton_native``. Use ``--skip-gt`` to omit GT.
GT only reports wall-clock / ``env_step_total`` (NeRD section timers do not
apply).

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_nerd_contact_modes \
  --compare \
  --neural-model-path-fg <fg_final_model.pt> \
  --neural-model-path-native <native_final_model.pt> \
  --num_envs 256 --warmup 20 --steps 100 --headless \
  --output-json /tmp/nerd_contact_bench.json \
  presets=newton_mjwarp
```

Ground-truth only:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_nerd_contact_modes \
  --contact-mode ground_truth \
  --num_envs 256 --warmup 20 --steps 100 --headless \
  presets=newton_mjwarp
```

Reported NeRD sections include ``contact_prepare``, ``adapter_update``,
``to_neural_inputs``, ``model_forward``, ``eval_fk``, and ``env_step_total``.

Example (256 envs, Anymal flat, 100 timed steps; wall-clock ms/step):

| section | ground_truth | fixed_ground | newton_native |
|---|---:|---:|---:|
| env_step_total | ~12 | ~91 | ~209 |
| adapter_update | n/a | 0 | ~123 |
| model_forward | n/a | ~17 | ~17 |
| contact_prepare | n/a | ~3 | ~5 |

Native is dominated by ``adapter_update`` (Newton collision + contact packing),
not by ``model_forward``.

Native flat packing is vectorized (no Python contact loop). Do not change contact
slot count or packing policy solely for speed without re-checking dynamics.
Use a native checkpoint whose contact feature schema matches the runtime
(older native ckpts may fail shape checks under current packing).

## Notes and limitations

- NeRD physics steps are eager (CUDA graphs disabled).
- Geometry-contact rewards and ``base_contact`` follow ``contact_mode`` (on for
  ``newton_native``, off for ``fixed_ground``).
- Dynamics models are often trained on simple-PD dataset trajectories while this
  RL task uses the LSTM actuator. Treat domain gap as expected.
- The policy observation follows the stock 48-D flat Anymal layout.
