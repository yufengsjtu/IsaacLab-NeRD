# OSMO Reference

OSMO support is optional. It packages the local repository, restores or
generates datasets, trains a model, and uploads outputs. NV-Datasets is the
default storage backend; Swift remains supported.

## Public presets

- `cartpole_newton_native`
- `franka_reach_newton_native`
- `anymal_newton_native`
- `anymal_rough_newton_native`
- `anymal_newton_native_contact_tokens`
- `anymal_rough_newton_native_contact_tokens`
- `anymal_newton_native_shared_per_body`
- `anymal_rough_formal`
- `anymal_rough_newton_native_shared_per_body_contact_force_formal`
- `anymal_rough_newton_native_shared_per_body_rope_formal`
- `anymal_rough_newton_native_shared_per_body_contact_force_rope`

Validate and submit a preset with:

```bash
./osmo_scripts/start.sh \
  --preset anymal_newton_native \
  --pool <osmo-pool>
```

Select Swift explicitly:

```bash
./osmo_scripts/start.sh \
  --preset anymal_newton_native \
  --storage-backend swift \
  --pool <osmo-pool>
```

Each preset owns one training-config mapping and declares its task names,
dataset files, contact representation, and resource request. Dataset cache
modes are `auto`, `require`, and `off`.

See [`osmo_scripts/README.md`](../../../osmo_scripts/README.md) for credentials,
storage layout, cache behavior, and launcher options. Validate workflows
locally before submission; repository release checks do not submit real OSMO
jobs.
