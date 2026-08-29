# OSMO Workflows

This directory contains the optional preset-driven OSMO launcher for
IsaacLab-NeRD. It packages the repository, restores or generates HDF5 datasets,
trains a model, and uploads logs and checkpoints.

No real OSMO job is submitted by repository tests.

## Storage backends

NV-Datasets is the default backend:

- `IsaacLab-NeRD-Code` stores the packaged repository.
- `IsaacLab-NeRD-Datasets` stores generated HDF5 datasets under
  `<dataset_subdir>/<env_name>/`.
- `IsaacLab-NeRD-Output` stores logs and checkpoints under the workflow name.

Select OpenStack Swift with `--storage-backend swift`. The equivalent default
containers are `isaaclab-nerd-code`, `isaaclab-nerd-datasets`, and
`isaaclab-nerd-output`.

## Public presets

- `cartpole_newton_native`: contact-free Cartpole using
  `Cartpole/transformer.yaml`.
- `anymal_newton_native`: recommended flat baseline with flat native contacts,
  using `Anymal/transformer_native.yaml`.
- `anymal_rough_newton_native`: recommended rough baseline with flat native
  contacts, using `Anymal/transformer_rough_native.yaml`.
- `anymal_newton_native_contact_tokens`: flat contact-token model using
  `Anymal/transformer_native_contact_tokens.yaml`.
- `anymal_rough_newton_native_contact_tokens`: rough contact-token model using
  `Anymal/transformer_rough_native_contact_tokens.yaml`.
- `anymal_newton_native_shared_per_body`: flat shared-per-body token encoder
  using `Anymal/transformer_native_shared_per_body.yaml`.
- `anymal_rough_formal`: maintained rough shared-per-body baseline using
  `Anymal/transformer_rough_native_shared_per_body_formal.yaml`.
- `anymal_rough_newton_native_shared_per_body_contact_force_formal`: rough
  model with auxiliary per-body contact-force prediction, using the matching
  `contact_force_formal` training config.
- `anymal_rough_newton_native_shared_per_body_rope_formal`: rough
  shared-per-body RoPE/KV-cache model using the matching `rope_formal` config.
- `anymal_rough_newton_native_shared_per_body_contact_force_rope`: rough
  model combining the force head and RoPE.

The release does not expose ablation, diagnostic, or remote smoke presets. A
small local smoke training YAML remains under the package configuration tree.

## Submit a workflow

Required credentials depend on the selected backend:

- NV-Datasets: `NGC_API_KEY` and `NVDATASET_TENANTID`.
- Swift: `SWIFT_AUTH_KEY`.
- Weights & Biases: `WANDB_API_KEY` when `--enable-wandb` is used.

Example:

```bash
./osmo_scripts/start.sh \
  --preset anymal_newton_native \
  --dataset-cache-mode auto \
  --pool <osmo-pool>
```

Use `--code-only` to upload the packaged source without submitting a workflow.
Run `./osmo_scripts/start.sh --help` for resource, storage, cache, and W&B
options.

`start.sh` validates the rendered workflow before submission. Extra arguments
after `--` are forwarded to OSMO.

## Dataset cache

`run_experiment.py` supports:

- `auto`: reuse a compatible cache and generate missing data.
- `require`: fail when required cached files are absent or incompatible.
- `off`: always regenerate.

Cached files are validated against the preset's environment, trajectory shape,
transition count, token schema, terrain context, and optional contact-force
schema. Policy-driven datasets are skipped when their configured policy
checkpoint is unavailable.

## Contact settings

All presets use `contact_mode: newton_native`.

- Cartpole uses `num_contacts_per_env: 0` and has no contact tensors.
- Anymal-C uses a positive contact capacity.
- Token presets must keep generation, training, and inference token packing
  settings aligned.
- Force presets add `--record-body-net-contact-forces` during generation.

The deprecated `fixed_ground` input emits a warning and then exits with an
error; it is not translated to native contacts.

## Files

- `start.sh`: package, upload, validate, and submit.
- `osmo_workflow.yaml`: workflow template and defaults.
- `entrypoint.sh`: pod entrypoint and final artifact upload.
- `run_experiment.py`: cache validation, generation, training, and evaluation.
- `presets/`: maintained experiment definitions.
- `lib/nvdataset_io.py`: NV-Datasets transfers.
- `lib/swift_io.py`: Swift transfers.

For the overall learning path, see the
[IsaacLab Neural documentation](../source/isaaclab_neural/docs/README.md).
