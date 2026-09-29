# OSMO Workflows

This directory contains the optional preset-driven OSMO launcher for
IsaacLab-NeRD. It packages the repository, restores or generates HDF5 datasets,
trains a model, and uploads logs and checkpoints.

No real OSMO job is submitted by repository tests.

## Storage backends

NV-Datasets is the default backend:

- `IsaacLab-NeRD-Code-<run-id>` stores a run-specific packaged repository.
- `IsaacLab-NeRD-Datasets` stores generated HDF5 datasets under
  `<dataset_subdir>/<env_name>/`.
- `IsaacLab-NeRD-Output` stores logs and checkpoints under the workflow name.

Dataset names do not guarantee the same backing resource or write access. After
an object PUT 403 on the old output storage, the default names were switched on
2026-09-24 to newly created datasets on `team-isaaclab`:

- `IsaacLab-NeRD-Datasets`: `8773b921-14d1-49d4-9b81-7809239e7b22`.
- `IsaacLab-NeRD-Output`: `e50afa3d-3ad7-4e18-a080-aa834738216b`.

This was a name replacement, not an in-place migration of old dataset IDs or files.
The previous datasets and their contents were retained on `team-isaac` under:

- `IsaacLab-NeRD-Datasets-legacy-team-isaac-20260924-115115`.
- `IsaacLab-NeRD-Output-legacy-team-isaac-20260924-115115`.

Their original IDs, snapshot IDs, file counts and byte counts were checked before
and after the switch. Existing consumers needing the old contents must use the
legacy names or original IDs. No legacy storage permission was changed.

Cloud workflow `nerd-default-storage-check-20260924-115115-1` verified uploads,
downloads, nested paths, empty files and matching SHA-256 hashes for both new
default datasets. Only after it completed successfully, the temporary
`SO101-Smoke-Data-20260924-025044` and `SO101-Smoke-Output-20260924-025044` datasets
and their current physical files were deleted as authorized. Local experiment
artifacts and code datasets were retained. The new defaults do not contain the
old smoke cache; their initial contents are only the small acceptance fixtures.

Use the normal defaults now. Remove temporary dataset overrides from your shell
if they were set, and omit the old SO101 `--data-dataset` / `--output-dataset` flags:

```bash
unset NVDATASET_DATA_DATASET NVDATASET_OUTPUT_DATASET
```

For another tenant, verify its authorized resource and perform a bounded roundtrip
before a full pipeline. SDK `storage_type` and `storage_service_id` are mutually
exclusive when creating a dataset. Standalone NV-Datasets maintenance scripts must
protect side effects with `if __name__ == "__main__":`; MSC telemetry can spawn a
process that imports the entry script again.

Select OpenStack Swift with `--storage-backend swift`. The equivalent default
containers are `isaaclab-nerd-code`, `isaaclab-nerd-datasets`, and
`isaaclab-nerd-output`.

The workflow image still supplies Isaac Sim and third-party runtime
dependencies. After the uploaded tarball is extracted, `init.sh` editable-installs
every local `source/isaaclab*` package (not only `isaaclab_neural`) and installs
a `sys.path` hook so Python imports the submitted repository instead of the
Isaac Lab distribution bundled in the image. Pods prefer the image uv
environment at `/opt/isaaclab-venv` when it exists, and fall back to Isaac Sim
`python.sh` on older images that only expose Kit Python.

Code archives include untracked source and required G1/Franka policy assets, but
exclude local workflow outputs, nested worktrees, and viewer state. Policy caches
from before command/observation refresh are not valid for the updated collector.
Use `--dataset-cache-mode off` with a new `--dataset-subdir` to regenerate them
without reusing the old cache. Cache shape checks alone cannot detect stale
policy observations.

## Private Nucleus assets

SO101 original-task assets require read access on `isaac-dev.ov.nvidia.com`.
For API tokens, `OMNI_USER` must be the literal `$omni-api-token`, while `OMNI_PASS`
contains a token issued by that Nucleus server. Do not use an NGC storage key or
embed either secret in a Dockerfile, workflow YAML, command argument or source file.

Register a fresh, user-owned OSMO credential from your local environment:

```bash
export OMNI_USER='$omni-api-token'
read -rsp 'Nucleus API token: ' OMNI_PASS
printf '\n'
export OMNI_PASS
export NUCLEUS_CREDENTIAL="$(python3 osmo_scripts/lib/nucleus_auth.py register)"
```

Registration uses private temporary payload files, removes them afterwards, and
prints only the new credential name. It never replaces an existing credential.
SO101 presets contain the current local credential reference; other users or token
rotations can override it with `--nucleus-credential "$NUCLEUS_CREDENTIAL"`.
The launcher attaches `OMNI_USER: omni_user` and `OMNI_PASS: omni_pass` under that
managed credential and preserves absolute paths to the bootstrap scripts. Presets
without a Nucleus credential retain their previous credential mappings. Local
`export` alone does not change an already-running pod. This does not grant writes
to NV-Datasets output datasets; those permissions are independent.

Collection/replay failures retain their original exit status and full log artifact,
and echo a bounded, credential-redacted tail to workflow stderr. This preserves
useful diagnostics even when final artifact upload fails.

## Preinstalled NV-Datasets runtime

The NV-Datasets path requires SDK >= 0.95.0. Submission checks the selected local
interpreter before packaging. Service errors are not treated as missing datasets:
only the SDK's explicit missing-name result permits creation. Existing code
datasets are never deleted by default; choose a fresh name. The low-level
`replace-files --allow-replace` option explicitly authorizes deletion and should
not be used for shared or in-use datasets.

The SO101 image is
`docker.io/rowany28070/isaac-lab-nerd:so101-newton160-cu130-sdk095`, published with
manifest digest `sha256:0362671e11f48e9a49186295d26d8e37be1e9cbd20cf47f00742d78324a6a651`.
It retains the original `g1-develop` image and installs the validated lab dependency
closure into `/opt/isaaclab-venv`, with NV-Datasets 0.95.0 and its dependencies in
`/opt/nvdataset-venv`. Storage subprocesses clear simulator Python/library paths;
storage dependencies do not upgrade the training environment. Credentials are
provided at runtime, never baked into the build context or image.

`entrypoint.sh` uses the image's `NVDATASET_PYTHON` and checks the preinstalled SDK.
It no longer runs an unconditional network `pip install -U nvdataset`. This avoids
the measured `isaac-lab-l40s-03` startup failures: PDX package-source DNS lookup
failed and unauthenticated access to the documented URM package source returned
HTTP 403. The image fixes package installation, not authorization or connectivity
to the NV-Datasets service or private asset servers.

Build a fresh minimal context from the already validated Python 3.12 environments:

```bash
uv run python docker/scripts/prepare_so101_image.py prepare \
  --lab-python "$PWD/.venv-develop/bin/python" \
  --sdk-python "$CONDA_PREFIX/bin/python" \
  --output logs/so101_image_context_new

docker build --network=host \
  -t rowany28070/isaac-lab-nerd:YOUR_NEW_TAG \
  logs/so101_image_context_new
```

The SDK environment must already contain the versions in
`osmo_scripts/nvdataset_requirements.in`. The script verifies the installed
transitive closure (using the repository's torch-stack override where needed) and
exports exact pins plus provenance, without installing into either source
environment. `docker/Dockerfile.so101` uses a digest-pinned parent and only the
exported requirements; no credentials, datasets or host home directories are copied.
`--no-deps` in the image installs this explicit full closure rather than resolving
new transitive versions. Image publication is a separate, explicitly authorized
`docker push`; do not overwrite a tag used by running jobs.

## Public presets

- `cartpole_newton_native`: contact-free Cartpole using
  `Cartpole/transformer.yaml`.
- `franka_reach_newton_native`: contact-free Franka Reach using
  `Franka/transformer_reach.yaml`.
- `franka_reach_policy_train`: contact-free Franka Reach trained on
  Reach-policy rollouts using `Franka/transformer_reach_policy.yaml`.
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

Additional ablation and smoke presets depend on the checked-out branch; see
`presets/` for the available recipes. Keep their dataset caches separate when
collection semantics differ.

## Submit a workflow

Required credentials depend on the selected backend:

- NV-Datasets: `NGC_API_KEY` and `NVDATASET_TENANTID`.
- Swift: `SWIFT_AUTH_KEY`.
- Weights & Biases: `WANDB_API_KEY` when `--enable-wandb` is used, or an existing
  `--wandb-credential NAME` with an `api_key` field. A new key is registered through
  private temporary payload files, and only the managed credential reference is
  attached to the workflow. Keys are never passed through `osmo --set` or stored
  as literal workflow environment values. Reuse a credential reference for later
  submissions rather than recreating it; existing credentials are not overwritten.

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

Franka Reach presets use the public image
`docker.io/rowany28070/isaac-lab-nerd:g1-develop` (no OSMO registry
credential). Anymal and Cartpole still default to
`nvcr.io/nvidia/isaac-lab:3.0.0-beta2`.

## SO101 Keyboard (fixed keyboard)

SO101 Keyboard NeRD uses one fixed scene: keyboard variant 0 at its authored root
pose. Both keyboard reset paths use zero pose offsets. Robot resets, the typing
curriculum, rewards, terminations and timing (100 Hz physics, 25 Hz control) are
unchanged. The randomized 32-variant scene, its presets and its data schema were
removed. Original-scene recordings and checkpoints are rejected; collect new data
and train a new model.

Two presets remain:

- `so101_keyboard_resident_unique`: eight resident 1024-world collectors, then
  eight-rank training with the K recipe (`SO101/transformer_keyboard_throughput.yaml`:
  shared key head, balanced robot/key loss groups, eager loading) on `ovx-l40`.
  Cache prefix: `so101-keyboard-fixed-v2`.
- `so101_keyboard_resident_unique_smoke`: the same model and entry point on two
  GPUs, with 16,384/8,192 unique train/validation targets and two epochs of two
  updates. Qualify it before launching the full recipe.

Both train `IsaacContrib-Keyboard-SO101-NeRD` (`SO101-Keyboard-Fixed`). Model inputs
are `states_embedding`, `joint_f`, `gravity_dir` and 128 owner-frame contact tokens
filtered with `solver_active_movable`. Physical-parameter and articulation-pose
conditioning were removed together with the randomized scene. The model also has
the seven-body world-frame robot normal-force head.

### Pipeline

The pipeline is: policy checkpoint -> bounded native collection -> data/seed/size
validation -> cache publication -> NeRD training -> checkpoint action-rollout
diagnostics. It does not retrain the policy. Keyboard collection keeps its task-aware
recording hook: ordinary `env.step()` executes the policy action while every physics
frame, typing event and reset boundary is recorded.

Stage the trusted policy locally before packaging (no upload):

```bash
uv run python osmo_scripts/lib/keyboard_data.py stage-policy \
  --preset so101_keyboard_resident_unique_smoke \
  --checkpoint /home/rowany/Downloads/SO101/model_14999.pt
```

The destination is `pretrained/control_policy/so101_keyboard/model_14999.pt`. The
collector uses the registered SO101 RSL-RL configuration and pins the checkpoint hash.

Each resident worker creates its environment and loads the policy once, then serves
independently seeded chunks on an exclusive GPU UUID (logical `cuda:0`). This is
process-isolated collection, not DDP. Seeds advance by 1000 per chunk from bases 42
(train) and 43 (validation). Collection stops when the usable unique-transition
quota is reached. Deterministic eager physics uses the explicit
`--bounded-deterministic-workspace` profile, whose audited MJWarp/Warp source
fingerprints must match the pinned image. It bounds scratch records; it does not
reduce physical contact or constraint capacities. Recordings include the derived
MJWarp model coefficients needed to reproduce warm workers.

The presets use `validation_mode: basic` and `require_keyboard_replay: false`: no
per-chunk physics replay is launched. Finite values, reset boundaries, shapes,
fixed-keyboard identity, provenance, contact capacity, seed independence, data volume
and hashes are checked. Collection fails on dropped or truncated contacts. The
`keyboard_basic_collection_bundle_v1` manifest records `replay_performed: false`.
Replay-qualified collection (`validation_mode: full_replay`) remains supported with at
most eight worlds per process. Qualify changed collector profiles locally with cold
and warm-worker replay.

Each logical training source is a portable HDF5 virtual view of its chunks. Dense
tensors and contact pairs are not duplicated; native CSR offsets are rebased. Replay
timelines stay separate, and the checkpoint evaluator consumes the original chunks.

Cache modes:
- `auto`: validate/reuse a qualified cache; otherwise collect and validate new data.
- `require`: fail on a cache miss; never collect.
- `off`: collect fresh data in a new local directory; never overwrite an existing run.

`export_collection_resume` snapshots only accepted chunks plus their receipts;
`validate_collection_resume` verifies a downloaded copy before a producer is stopped.
Setting `collection.resume_from_input: true` requires `--dataset-cache-mode auto` with
that recovery prefix as dataset input. A missing or corrupted checkpoint fails instead
of recollecting, and recovered chunks are not a training-ready bundle.

After access is authorized and upload/submission is approved:

```bash
uv run bash osmo_scripts/start.sh --preset so101_keyboard_resident_unique_smoke \
  --dataset-cache-mode auto --pool "$OSMO_POOL" --image "$SO101_IMAGE" \
  --code-dataset "SO101-Code-$(date -u +%Y%m%d-%H%M%S)"
```

The SO101 image `docker.io/rowany28070/isaac-lab-nerd:so101-newton160-cu130-sdk095`
pins Newton 1.6.0, MJWarp 3.12.0, Warp 1.17.0, PyTorch 2.12.0+cu130 and RSL-RL 5.5.1;
the runtime fails on mismatches. The worker needs authorized access to the private
SO101 Nucleus USD. `--num-gpu` sets both allocated GPUs and training processes.
`keyboard_run_status.json` separates finite execution from `task_fidelity_accepted`,
which remains false pending physical-error and task-event review.

### Training recipe

| Setting | SO101 Keyboard | Franka Lift mixed |
| --- | --- | --- |
| Epochs x iterations | 1000 x 1250 | Same |
| Batch per rank / global sequences | 512 / 4096 with eight ranks | Same |
| Sequence/history length | 10 physics frames | Same |
| Adam, learning rate, clipping | (0.9, 0.999), 2e-4 to 2e-5 linear, norm 1.0 | Same |
| Transformer layers / width / heads | 6 / 384 / 12 | Same |
| Contact encoder | Shared-per-body, hidden 64, latent 16 per body | Same |
| Contact tokens / frame | 128 filtered, owner | 64, root |
| Extra conditioning | None | None |
| Force head | Seven robot bodies, world-frame normal force | Two fingers, root-frame |
| Data volume | At least 10M train + 3M validation unique transitions | 10M train + three 1M validation |

The training dataset declares `capacity_unit: unique_transition` and
`capacity_scope: global`: `max_capacity` counts distinct recording/world/frame targets
across all ranks, not raw clip slots or optimizer visits. `window_sampling: sliding`
enumerates stride-one windows, deduplicates identical source windows introduced by clip
overlap, and supervises every selected position; only capacity-boundary context is
masked. RMS counts follow window exposures; `unique_training_targets_seen` tracks
distinct source IDs. Sliding DDP assigns each recording/world timeline to one rank.
Resolved configs and checkpoints record the sampling schema, normalization convention,
selected counts and selection fingerprint.

The fixed-v2 recording contract stores authored robot/keyboard root poses and rejects
nonzero reset offsets in either reset path. Recorded pose tensors are checked against
that contract, and training/evaluation bind the poses and physical parameters to the
live model. Fixed-v1 and randomized-scene data/checkpoints require recollection rather
than silent relabeling. Raw native-pair storage and physical collision solving are unchanged.

Resuming compares the requested YAML plus CLI overrides with the checkpoint before
merging it. Different solver input/target semantics, input fields, model architecture
or environment are rejected rather than silently overwritten. Reusing normalization
also requires matching training-pool fingerprints, capacity and sampling contracts;
otherwise pass `--update-dataset-statistics`.

Sliding DDP still keeps source timelines on one rank. Each batch uses population
importance weights (`rank_window_count / local_batch_windows`) so unequal rank pools
do not change the global window objective. This aligns objective weights, not the
order or variance of global IID sampling. RMSE is reduced from squared errors; active
RMSE and contact precision/recall use summed numerators and their own denominators
across batches/ranks. Maximum metrics retain MAX reduction.

The `solver_active_movable` filter keeps negative recorded separation and rejects only
pairs whose two sides are immovable, using joint-tree mobility saved in the recording
contract and checked against the live NeRD model. Native pairs and collision solving
are unchanged. A qualified 1024-world probe needed at most 72 directed tokens, which
supports 128 with margin but is not a guarantee for every seed.

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
- `lib/local_isaaclab.py`: overlay the uploaded Isaac Lab tree on the image.

For the overall learning path, see the
[IsaacLab Neural documentation](../source/isaaclab_neural/docs/README.md).
