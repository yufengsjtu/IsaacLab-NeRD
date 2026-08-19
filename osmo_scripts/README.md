# OSMO Training Scripts

This directory contains the OSMO entrypoints for running IsaacLab-NeRD training
jobs with NV-Datasets or OpenStack Swift. NV-Datasets remains the default;
select Swift with `--storage-backend swift`.

`--storage-backend osmo_data` uses OSMO's existing DATA credentials instead of
raw Swift user/key environment variables. Its default standalone prefix is
`swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan`; override
it with `--osmo-swift-base` or `OSMO_SWIFT_BASE`.

## Storage Layout

The default workflow uses three long-lived NV-Datasets datasets:

- `IsaacLab-NeRD-Code`: packaged source code. `start.sh` force-replaces this
  dataset before each submit.
- `IsaacLab-NeRD-Datasets`: generated HDF5 datasets. Files are stored under
  `<dataset_subdir>/<env_name>/`.
- `IsaacLab-NeRD-Output`: logs, TensorBoard files, checkpoints, and staged
  outputs. Files are stored under `<workflow_name>/`, where `workflow_name`
  includes the run id.

The opt-in Swift backend uses equivalent `isaaclab-nerd-code`,
`isaaclab-nerd-datasets`, and `isaaclab-nerd-output` containers. Code is stored
as `IsaacLab-NeRD.tar.gz`; datasets use `<dataset_subdir>/<env_name>/`; outputs
use `<workflow_name>/`.

## Supported Features

This folder provides a preset-driven OSMO launcher for end-to-end IsaacLab-NeRD
experiments. The supported flow is:

1. package the local repository,
2. upload/replace the packaged source in the selected storage backend,
3. submit an OSMO workflow,
4. download source and optional cached HDF5 datasets inside the pod,
5. generate missing datasets,
6. start TensorBoard,
7. train the NeRD model,
8. upload generated datasets to the data dataset and upload
   logs/checkpoints to the output dataset.

### Experiment Presets

The launcher supports declarative presets in `osmo_scripts/presets/`:

- `anymal_newton_native`: Anymal-C with Newton native contacts,
  flat representation, `num_contacts_per_env: 64`, and
  `penetration_priority` contact packing.
- `anymal_newton_native_contact_tokens`: Anymal-C with Newton native
  directed contact tokens, `max_contact_tokens: 64`, and pair-atomic
  body-round-robin packing.
- `anymal_rough_newton_native`: Anymal-C rough terrain with Newton native
  contacts, rough dataset-generation/deployment/NeRD task IDs, and separate
  `Anymal-C-Rough-Native` dataset paths.
- `anymal_rough_newton_native_contact_tokens`: rough-terrain equivalent with
  token HDF5 datasets and the contact-token transformer config.
- `anymal_rough_newton_native_body_routed`: rough-terrain body-routed Deep Sets
  model that reuses the same canonical contact-token HDF5 datasets.
- `anymal_rough_newton_native_active15_dataset`: generate and upload the
  owner-frame Active15 dataset without starting training.
- `anymal_rough_newton_native_active15` and
  `anymal_rough_newton_native_active15_lr_1e-3`: train Active15 with the two
  requested learning-rate schedules on the dedicated Active15 HDF5 dataset.
- `anymal_fixed_ground`: Anymal-C with fixed-ground abstract contacts.
- `cartpole_fixed_ground`: Cartpole with fixed-ground contacts.

Each preset declares OSMO resources, task names, contact mode, dataset
generation scale, train config path, optional policy checkpoint, and the list of
HDF5 datasets to generate or load.

### Dataset Cache Modes

`run_experiment.py` can reuse generated HDF5 datasets from the configured
NV-Datasets dataset or Swift container.

- `auto`: try to load cached datasets; generate and upload them if missing.
- `require`: fail if the required cached datasets are not present.
- `off`: ignore cache and always regenerate datasets.
- `local_require`: require a complete local/Lustre cache and never download or
  regenerate it.

Set this through `start.sh`:

```bash
./osmo_scripts/start.sh \
  --preset anymal_newton_native \
  --dataset-cache-mode off \
  --pool <osmo-pool>
```

For contact-token training:

```bash
./osmo_scripts/start.sh \
  --preset anymal_newton_native_contact_tokens \
  --dataset-cache-mode off \
  --pool <osmo-pool>
```

Use `anymal_rough_newton_native_contact_tokens` for rough terrain. Token
presets use distinct workflow names and dataset cache subdirectories, so they
cannot accidentally reuse flat HDF5 files.

Cached datasets are searched under several compatible layouts, including
`<dataset_subdir>/<env_name>/` and `<dataset_subdir>/`. When
`AMLFS_DATA_ROOT` is available, the entrypoint links `data/datasets` to that
pool-wide Lustre directory. `--dataset-cache-mode local_require` validates and
uses this cache without mounting or copying the Swift dataset.

### Dataset Generation Features

Preset `dataset_specs` support multiple dataset roles:

- train and validation splits,
- zero-action validation datasets,
- deployment-task validation datasets,
- optional policy-driven validation datasets.

If a policy dataset is requested but the configured policy checkpoint is not
available in the OSMO pod, that dataset is skipped instead of failing the whole
workflow.

Contact-related generation options come from each preset:

- `contact_mode`: `fixed_ground` or `newton_native`.
- `contact_representation`: `flat`, `contact_tokens`, or `active15_tokens`.
- `num_contacts_per_env`: required for native contacts.
- `max_contact_tokens`: padded capacity for either token representation.
- `contact_packing_policy`: native packing policy such as
  `penetration_priority`; token presets use
  `body_round_robin_pair_atomic`.
- `states_frame`: optional override for dataset generation, used by Cartpole
  fixed-ground to collect world-frame data.

### Active15 Dataset and Training

Generate once to standalone Swift while retaining the same files on Lustre:

```bash
./osmo_scripts/start.sh \
  --preset anymal_rough_newton_native_active15_dataset \
  --storage-backend osmo_data \
  --dataset-cache-mode off \
  --amlfs-data-root /mnt/amlfs-04/home/jiex/data/datasets \
  --pool isaac-srl-l40-04 \
  --priority NORMAL
```

After the generation workflow validates all five files, training jobs should use
`--dataset-cache-mode local_require`. This avoids the Swift dataset input
mount and reads directly through the pool-wide Lustre symlink.

### Training Features

Training is launched from the preset's `train_task` and `train_cfg`.

- Single-GPU presets run `python -m isaaclab_neural.train.train`.
- Multi-GPU presets run `torch.distributed.run --nproc_per_node <num_gpus>`.
- `--train-seed N` sets the training seed independently of the fixed dataset-generation seeds.
- `train_preset` is forwarded as an Isaac Lab Hydra preset, for example
  `presets=newton_mjwarp` for Anymal Newton/MJWarp jobs.
- `update_dataset_statistics: true` adds `--update-dataset-statistics` before
  training.

Training hyperparameters live in the YAML files under
`source/isaaclab_neural/isaaclab_neural/train/cfg/`. Presets select the config
file and runtime scale, but they do not mutate the training YAML.

### Output and Logging

The workflow writes logs and artifacts under `OUTPUT_LOCAL_PATH`, defaulting to
`/tmp/runs/output` in OSMO.

Generated artifacts include:

- `main_scripts.log` from `run_experiment.py`,
- `entry.log` from the OSMO entrypoint,
- TensorBoard logs and `tensorboard.pid`,
- training checkpoints under `<env_name>/`.

`entrypoint.sh` uploads the full output directory to `IsaacLab-NeRD-Output` at
exit. If dataset generation ran, `run_experiment.py` uploads the generated HDF5
datasets to `IsaacLab-NeRD-Datasets` only, not to the output dataset, to avoid
duplicating large datasets in run outputs.

#### Weights & Biases (optional)

TensorBoard is always enabled. To also log to W&B, bind an existing OSMO
GENERIC credential whose payload contains `wandb_api_key`:

```bash
osmo credential list
./osmo_scripts/start.sh --enable-wandb --wandb-credential wandb --wandb-project nerd-newton
```

Optional overrides:

```bash
./osmo_scripts/start.sh \
  --enable-wandb \
  --wandb-project nerd-newton \
  --wandb-exp-name anymal-native-v6 \
  --wandb-entity <team-or-user>
```

What W&B receives when enabled:

- the same scalar metrics as TensorBoard,
- hyperparameters from the trainer config panel,
- `cfg.yaml` as an artifact,
- **best** checkpoints only (`best_valid_*`, `best_eval_model`) via `wandb.save`,
- stdout/stderr from the **training subprocess** (epoch logs, grad info, etc.).

What W&B does **not** receive automatically:

- periodic `model_epoch{N}.pt` checkpoints,
- dataset-generation logs from earlier in `run_experiment.py`,
- `entry.log` / `main_scripts.log` (those stay in the configured output storage).

Disable checkpoint uploads with `--no-wandb-save-checkpoints` on the training CLI,
or pass that through `run_experiment.py` when invoking it directly.

### Local Utility Scripts

This folder also includes local helpers for NV-Datasets:

- `list_nvdataset_files.py`: list files in an NV-Datasets dataset, optionally
  by downloading when the SDK has no remote listing API.
- `pull_nvdataset.py`: download an NV-Datasets dataset or prefix locally.
- `delete_nvdataset.py`: dry-run and delete a whole NV-Datasets dataset, or
  selected files by exact key, prefix, or regex when the installed SDK exposes
  file-delete APIs.
- `lib/nvdataset_io.py`: shared upload/download/replace implementation.
- `lib/package_code.py`: package the local repository for upload.
- `lib/simple_yaml.py`: minimal YAML parser used by OSMO bootstrap scripts.

## Submit Existing Experiments

Set NV-Datasets credentials first:

```bash
export NGC_API_KEY=<your-nvapi-key>
export NVDATASET_TENANTID=<your-tenant-id>
```

Use the credential-backed standalone Swift prefix without local Swift secrets:

```bash
./osmo_scripts/start.sh \
  --storage-backend osmo_data \
  --osmo-swift-base swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan \
  --osmo-data-credential css_team-nvr-srl
```

To use Swift instead, provide legacy name/key authentication only through the
environment:

```bash
export SWIFT_AUTH_URL=https://pdx.s8k.io
export SWIFT_AUTH_VERSION=1
export SWIFT_USER=team-isaac-lab
read -rsp "Swift auth key: " SWIFT_AUTH_KEY && echo
export SWIFT_AUTH_KEY
./osmo_scripts/start.sh --storage-backend swift --pool <osmo-pool>
```

Override the default containers with `--code-container`, `--data-container`,
and `--output-container`. Objects larger than 1 GiB use Swift static large
objects; set `SWIFT_SEGMENT_SIZE_BYTES` to change the segment size.

Submit the default Anymal Newton-native job:

```bash
./osmo_scripts/start.sh --pool <osmo-pool>
```

Submit Anymal fixed-ground:

```bash
./osmo_scripts/start.sh --preset anymal_fixed_ground --pool <osmo-pool>
```

Submit the full rough Newton-native baseline:

```bash
./osmo_scripts/start.sh \
  --preset anymal_rough_newton_native \
  --dataset-cache-mode off \
  --pool <osmo-pool>
```

Submit Cartpole fixed-ground:

```bash
./osmo_scripts/start.sh --preset cartpole_fixed_ground --pool <osmo-pool>
```

Upload code without submitting:

```bash
./osmo_scripts/start.sh --code-only
```

## Script Structure

- `start.sh`: thin submit CLI. It resolves a preset, packages code, replaces
  `IsaacLab-NeRD-Code`, sets/reuses the OSMO credential, validates the workflow,
  and submits it.
- `osmo_workflow.yaml`: thin OSMO bootloader. It declares resources,
  credentials, environment variables, and the files needed before code download.
- `entrypoint.sh`: task entrypoint. It installs the selected storage client,
  downloads code/data, extracts the code tree, runs init, runs the selected
  experiment preset, and uploads outputs on exit.
- `run_experiment.py`: structured runner for cache lookup, dataset generation,
  generated dataset upload, TensorBoard, and training.
- `lib/nvdataset_io.py`: shared NV-Datasets download/upload/replace helper.
- `lib/swift_io.py`: shared Swift authentication, upload, and download helper.
- `lib/package_code.py`: local source packaging helper.
- `lib/preset.py`: preset resolver used by submit-time code.
- `presets/*.yaml`: declarative experiment presets.

## Presets

Each preset contains:

- `workflow`: default workflow base name and dataset subdirectory.
- `resources`: OSMO resource defaults (`num_gpu`, `num_cpu`, `memory`,
  `storage`, and `platform`).
- `experiment`: task names, environment names, train config path, runtime env
  counts, contact settings, dataset specs, optional policy checkpoint, and
  optional training preset.

Training hyperparameters remain in the training YAML files under
`source/isaaclab_neural/isaaclab_neural/train/cfg/`; presets select which config
to run and how much data/runtime resource to use.

### Rough Shared-Per-Body Ablation

The rough shared-per-body ablation compares large, medium, and small models
using either 20 million or 10 million training transitions. All six conditions
reuse one complete 20-million-transition contact-token dataset. The 10-million
conditions load a deterministic trajectory prefix and use half as many
iterations per epoch to preserve the number of passes over the training data.

Set the existing Swift dataset prefix and common submission options:

```bash
export ABLATION_DATASET_SUBDIR=contact-ablation-20260812-020518-rough-token
export ABLATION_TAG=rough-shared-ablation
export POOL=<osmo-pool>

COMMON_ARGS=(
  --storage-backend swift
  --dataset-subdir "$ABLATION_DATASET_SUBDIR"
  --dataset-cache-mode require
  --num-cpu 80
  --memory 512Gi
  --storage 512Gi
  --enable-wandb
  --wandb-project nerd-contact-ablation
  --pool "$POOL"
)
```

Submit the six conditions sequentially; their workflows may run concurrently:

```bash
./osmo_scripts/start.sh "${COMMON_ARGS[@]}" \
  --preset anymal_rough_newton_native_shared_per_body \
  --wandb-exp-name "$ABLATION_TAG-large-20m"

./osmo_scripts/start.sh "${COMMON_ARGS[@]}" \
  --preset anymal_rough_newton_native_shared_per_body_10m \
  --wandb-exp-name "$ABLATION_TAG-large-10m"

./osmo_scripts/start.sh "${COMMON_ARGS[@]}" \
  --preset anymal_rough_newton_native_shared_per_body_medium \
  --wandb-exp-name "$ABLATION_TAG-medium-20m"

./osmo_scripts/start.sh "${COMMON_ARGS[@]}" \
  --preset anymal_rough_newton_native_shared_per_body_medium_10m \
  --wandb-exp-name "$ABLATION_TAG-medium-10m"

./osmo_scripts/start.sh "${COMMON_ARGS[@]}" \
  --preset anymal_rough_newton_native_shared_per_body_small \
  --wandb-exp-name "$ABLATION_TAG-small-20m"

./osmo_scripts/start.sh "${COMMON_ARGS[@]}" \
  --preset anymal_rough_newton_native_shared_per_body_small_10m \
  --wandb-exp-name "$ABLATION_TAG-small-10m"
```

The large, medium, and small models use approximately 10.94M, 3.31M, and
1.42M parameters, respectively. Keep the training seed fixed when comparing
the six conditions.

## Add a New Environment

1. Create a preset file in `osmo_scripts/presets/`, for example
   `my_env_fixed_ground.yaml`.

```yaml
workflow:
  base_name: my-env-fixed-ground
  dataset_subdir: my-env-fixed-ground

resources:
  num_gpu: 1
  num_cpu: 15
  memory: 64Gi
  storage: 256Gi
  platform: ovx-l40s

experiment:
  env_name: MyEnv
  robot_name: MyRobot
  data_gen_task: Isaac-MyEnv-Dataset-Gen-v0
  deployment_task: Isaac-MyEnv-v0
  train_task: Isaac-MyEnv-NeRD-v0
  train_cfg: ./source/isaaclab_neural/isaaclab_neural/train/cfg/MyEnv/transformer.yaml
  train_preset: ""
  num_gpus: 1
  contact_mode: fixed_ground
  sample_mode: action
  initial_states_source: env
  train_transitions: 1000000
  valid_transitions: 100000
  data_gen_num_envs: 256
  train_num_envs: 256
  trajectory_length: 100
  dataset_specs:
    - filename: dataset_train.hdf5
      seed: 0
      split: train
      task_kind: default
      zero_actions: false
      use_policy: false
    - filename: dataset_valid.hdf5
      seed: 1
      split: valid
      task_kind: default
      zero_actions: false
      use_policy: false
```

2. Validate the workflow:

```bash
osmo workflow validate \
  --set num_gpu=1 num_cpu=15 \
  --set-string memory=64Gi storage=256Gi resource_platform=ovx-l40s \
  --set-string nvdataset_credential=nvdataset_cred \
  --set-string workflow_name=my-env-fixed-ground-test \
  --set-string workflow_base_name=my-env-fixed-ground \
  --set-string dataset_subdir=my-env-fixed-ground \
  --set-string osmo_experiment_preset=my_env_fixed_ground \
  -- osmo_scripts/osmo_workflow.yaml
```

3. Submit:

```bash
./osmo_scripts/start.sh --preset my_env_fixed_ground --pool <osmo-pool>
```

## Useful Overrides

Override the generated workflow base name:

```bash
./osmo_scripts/start.sh \
  --preset anymal_fixed_ground \
  --workflow-name anymal-c-fixed-ground-debug \
  --pool <osmo-pool>
```

Override OSMO resources:

```bash
./osmo_scripts/start.sh \
  --preset cartpole_fixed_ground \
  --num-gpu 1 \
  --num-cpu 15 \
  --memory 64Gi \
  --storage 256Gi \
  --pool <osmo-pool>
```

List files in an NV-Datasets dataset:

```bash
./isaaclab.sh -p osmo_scripts/list_nvdataset_files.py IsaacLab-NeRD-Datasets
```

Show soft-deleted files too:

```bash
./isaaclab.sh -p osmo_scripts/list_nvdataset_files.py IsaacLab-NeRD-Datasets --include-deleted
```

Download an NV-Datasets dataset locally:

```bash
./isaaclab.sh -p osmo_scripts/pull_nvdataset.py \
  IsaacLab-NeRD-Datasets \
  --prefix anymal-c-fixed-ground/Anymal-C \
  --output-dir /tmp/IsaacLab-NeRD-Datasets \
  --clean \
  --list
```

Dry-run deleting a dataset:

```bash
./isaaclab.sh -p osmo_scripts/delete_nvdataset.py IsaacLab-NeRD-Anymal-C-Native --delete-dataset
```

Actually delete it:

```bash
./isaaclab.sh -p osmo_scripts/delete_nvdataset.py IsaacLab-NeRD-Anymal-C-Native --delete-dataset --yes
```

Dry-run deleting files under a prefix:

```bash
./isaaclab.sh -p osmo_scripts/delete_nvdataset.py \
  IsaacLab-NeRD-Output \
  --prefix anymal-c-newton-native-20260708-201234
```

Actually delete the matched files:

```bash
./isaaclab.sh -p osmo_scripts/delete_nvdataset.py \
  IsaacLab-NeRD-Output \
  --prefix anymal-c-newton-native-20260708-201234 \
  --yes
```

Dry-run deleting all HDF5 files in a dataset:

```bash
./isaaclab.sh -p osmo_scripts/delete_nvdataset.py \
  IsaacLab-NeRD-Anymal-C-Newton-Native-Output \
  --regex '\.hdf5$'
```
