# Rough Anymal Per-Body Contact Force

## Contract

- Branch: `rowany/per-body-net-contact-force`
- Worktree: `/home/rowany/IsaacLab-NeRD/.worktree/feature-per-body-net-contact-force`
- Base: `contact_encoder_set` at `6ae58df788`
- Input contact representation: `contact_tokens` with `shared_per_body`
- Stage 1 output: per-primary-body resultant/net linear contact force `[N]`,
  including normal and tangential/friction components
- Dataset tensor: `body_net_contact_forces_w`, shape `[trajectory, time, body, 3]`
- Dataset frame: world; timing: post-solve transition from `t` to `t+1`
- Model output frame: current root-body anchor frame
- Runtime shape for Anymal-C: `[num_envs, 17, 3]`
- The auxiliary force output does not feed back into dynamics.
- Existing state-only models, datasets, and checkpoints must remain compatible.

## Tasks

1. Record and validate Newton ContactSensor labels.
2. Add a backward-compatible auxiliary force head and solver cache.
3. Add force normalization, loss, metrics, and checkpoint support.
4. Add rough shared-per-body smoke/formal recipes, docs, and tests.
5. Run unit, runtime smoke, and repository pre-commit validation.

## Deferred

Contact moments/torques, full contact-wrench prediction, and joint-wrench
reconstruction are Stage 2 and are not part of this implementation.
