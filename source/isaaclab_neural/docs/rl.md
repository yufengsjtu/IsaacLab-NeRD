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
- Validated flat training uses ``--contact-mode newton_native
  --num-contacts-per-env 64`` with ``push_robot`` disabled

## Prerequisites

1. Install the package:

```bash
./isaaclab.sh -p -m pip install -e source/isaaclab_neural --no-deps
```

2. Provide a trained NeRD dynamics checkpoint (Transformer / fixed_ground for
   the default flat task), for example:

```text
./pre-trained_models/Anymal-C/nn/final_model.pt
```

Train dynamics first with `isaaclab_neural.train.train` if needed (see
[train.md](train.md)).

## Train

Preferred entry point:

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path ./pre-trained_models/Anymal-C/nn/final_model.pt \
  --num_envs 4096 \
  --max_iterations 300 \
  --headless \
  presets=newton_mjwarp
```

Smoke (short) run:

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path ./pre-trained_models/Anymal-C/nn/final_model.pt \
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

```bash
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.play \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path ./pre-trained_models/Anymal-C/nn/final_model.pt \
  --checkpoint logs/rsl_rl/anymal_c_flat_nerd/<run>/model_<step>.pt \
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

Compare modes end-to-end (isolated workers; writes JSON):

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_nerd_contact_modes \
  --compare \
  --neural-model-path-fg <fg_final_model.pt> \
  --neural-model-path-native <native_final_model.pt> \
  --num_envs 256 --warmup 20 --steps 100 --headless \
  --output-json /tmp/nerd_contact_bench.json \
  presets=newton_mjwarp
```

Reported sections include ``contact_prepare``, ``adapter_update``,
``to_neural_inputs``, ``model_forward``, ``eval_fk``, and ``env_step_total``.

Example (64 envs, Anymal flat, after vectorized packing / view-based
``to_neural_inputs``):

| section | fixed_ground | newton_native |
|---|---:|---:|
| env_step_total | ~22 ms | ~33 ms |
| contact_prepare | ~0.2 ms | ~2.6 ms |
| adapter_update | n/a | ~8 ms |
| to_neural_inputs | n/a | ~0.05 ms |
| model_forward | ~6 ms | ~6 ms |

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
