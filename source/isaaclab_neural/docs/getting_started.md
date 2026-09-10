# Getting Started

## Requirements

IsaacLab-NeRD follows the Isaac Lab and Isaac Sim versions documented in the
repository root. Complete the upstream binary installation first, then install
the repository extensions:

```bash
./isaaclab.sh -i
```

Run Python through `./isaaclab.sh -p` so Isaac Lab, Newton, Warp, and the
extension packages use the same environment.

## Workflow

A NeRD experiment has four stages:

1. Generate HDF5 trajectories with a ground-truth Newton environment.
2. Train a neural dynamics checkpoint from those trajectories.
3. Evaluate the checkpoint in a registered NeRD environment.
4. Optionally train an RSL-RL control policy against that environment.

Start with [Cartpole](cartpole.md) to verify the installation. Continue with
[Anymal-C](anymal.md) for contact-aware locomotion, or [Franka](franka.md) for
manipulation.

## Dataset collection intervals

Physics-frame action collection holds each random action for the environment's
control decimation by default. `--action-hold-frames N` extends that hold to at
least `N` physics frames. Environment-step collection ignores this option.
Policy collection refreshes commands and observations once per control interval
without running episode terminations or resets inside a fixed-length clip.
Regenerate policy datasets collected before this refresh was implemented rather
than reusing cached trajectories with stale observations.

## Contact configuration

Use `contact_mode: newton_native` for every new configuration.

- Cartpole is contact-free and uses `num_contacts_per_env: 0`.
- Franka Reach is also contact-free; Franka Lift uses a positive contact
  capacity like Anymal-C.
- Anymal-C requires a positive contact capacity; maintained configurations use
  64.
- `fixed_ground` is deprecated and rejected because its representation is not
  compatible with native Newton contacts.

Generated datasets, checkpoints, logs, and videos are local artifacts and are
ignored by Git.
