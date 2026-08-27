# Contact encoder review and reproduction guide

This guide is for reviewing the contact encoders and data pipeline without
using the internal experiment labels A, B, C, and D. It covers:

1. solver-active filtering with the shared per-body and global-attention
   encoders;
2. the owner-routed native15 encoder with and without robot self-collisions;
3. generation and fixed-checkpoint evaluation of the current dataset sampling
   strategy.

The commands below are examples for OSMO. They never intentionally regenerate
data during a training submission: training uses `DATASET_CACHE_MODE=require`
and fails closed if the requested HDF5 package is missing or incompatible.

## Architecture and data matrix

| Review case | Stored representation | Model encoder | Runtime selection |
|---|---|---|---|
| Shared per-body, solver-active | Canonical 17-D ContactTokens | `shared_per_body` | Newton solver-active sidecar |
| Global attention, solver-active | Canonical 17-D ContactTokens | `global_attention` | Newton solver-active sidecar |
| Owner-routed native15, solver-active, no self-collision | Raw15 | `body_routed_active15` | Geometric solver-active view; self-collisions absent from storage |
| Owner-routed native15, solver-active, with self-collision | Raw15Self | `body_routed_active15` | Geometric solver-active view; directed self-collision rows retained |

The canonical ContactTokens dataset is shared by the first two cases. The two
owner-routed cases use two physical dataset families:

- Raw15 supports raw and solver-active views without robot self-collisions.
- Raw15Self supports raw and solver-active views with directed robot
  self-collision rows.

Raw15 and Raw15Self are deliberately kept separate. Do not describe the two
owner-routed training runs as a strictly paired self-collision ablation unless
the underlying trajectory and sampling identities have also been verified.

## Relevant implementation and configuration files

Core implementations:

- `source/isaaclab_neural/isaaclab_neural/models/shared_per_body_contact_encoder.py`
- `source/isaaclab_neural/isaaclab_neural/models/contact_set_model.py`
- `source/isaaclab_neural/isaaclab_neural/models/body_routed_active15_model.py`
- `source/isaaclab_neural/isaaclab_neural/contacts/active15_contact_encoder.py`
- `source/isaaclab_neural/isaaclab_neural/contacts/native15_self_contact_encoder.py`
- `source/isaaclab_neural/isaaclab_neural/contacts/tensor_utils.py`

Training configurations:

- `transformer_rough_native_shared_per_body_active_filter.yaml`
- `transformer_rough_native_contact_tokens_active_filter.yaml`
- `transformer_rough_native_body_routed_active15.yaml`
- `transformer_rough_native_body_routed_active15_self.yaml`

The files are under
`source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/`. The matching OSMO
presets are under `osmo_scripts/presets/`.

## Review tests

Run the focused implementation and configuration tests before launching:

```bash
./isaaclab.sh -p -m pytest -q \
  source/isaaclab_neural/test/test_contact_set_encoder.py \
  source/isaaclab_neural/test/test_shared_per_body_contact_encoder.py \
  source/isaaclab_neural/test/test_active15_contact_encoder.py \
  source/isaaclab_neural/test/test_contact_active_filter.py \
  source/isaaclab_neural/test/test_native15_self_collision.py \
  source/isaaclab_neural/test/test_native15_self_model.py \
  source/isaaclab_neural/test/test_raw15_contact_encoder.py \
  osmo_scripts/test_a_active_self_collision_cache.py \
  osmo_scripts/test_a_active_self_collision_campaign.py
```

Run repository formatting and static checks before packaging code:

```bash
./isaaclab.sh -f
git status --short
```

`osmo_scripts/lib/package_code.py` includes tracked and untracked non-ignored
files. Submit only from a clean checkout. Record `git rev-parse HEAD` with every
workflow.

## OSMO prerequisites

The examples use OSMO DATA so they work on a pool without Lustre. Set these for
the reviewing account and project:

```bash
export REVIEW_POOL="<l40-pool>"
export REVIEW_OSMO_DATA_BASE="<swift-prefix-owned-by-the-reviewing-account>"
export REVIEW_OSMO_DATA_CREDENTIAL="<osmo-data-credential>"
export REVIEW_WANDB_ENTITY="<wandb-entity>"
export REVIEW_WANDB_PROJECT="<wandb-project>"
export REVIEW_WANDB_CREDENTIAL="<wandb-credential>"
export REVIEW_TAG="<unique-review-tag>"
```

Verify access and current capacity before submission:

```bash
osmo profile list
osmo pool list --mode free
osmo credential list
```

The following helper keeps the storage backend, cache contract, resources,
priority, code namespace, and W&B naming explicit. It uses only NORMAL priority.

```bash
submit_contact_review() {
  case_name="$1"
  preset="$2"
  dataset_subdir="$3"
  seed="$4"

  env -u OSMO_SWIFT_BASE -u AMLFS_DATA_ROOT \
    RUN_ID="${case_name}-s${seed}-${REVIEW_TAG}" \
    ./osmo_scripts/start.sh \
      --preset "${preset}" \
      --workflow-name contact-review \
      --dataset-subdir "${dataset_subdir}" \
      --storage-backend osmo_data \
      --osmo-swift-base "${REVIEW_OSMO_DATA_BASE}" \
      --osmo-data-credential "${REVIEW_OSMO_DATA_CREDENTIAL}" \
      --dataset-cache-mode require \
      --train-seed "${seed}" \
      --num-gpu 8 \
      --num-cpu 96 \
      --memory 512Gi \
      --storage 512Gi \
      --platform ovx-l40 \
      --pool "${REVIEW_POOL}" \
      --priority NORMAL \
      --enable-wandb \
      --wandb-entity "${REVIEW_WANDB_ENTITY}" \
      --wandb-project "${REVIEW_WANDB_PROJECT}" \
      --wandb-credential "${REVIEW_WANDB_CREDENTIAL}" \
      --wandb-exp-name "contact-review-${case_name}-s${seed}-${REVIEW_TAG}"
}
```

Before using an existing dataset, recursively list its exact OSMO DATA prefix.
`osmo data check` alone is insufficient because it can succeed for an empty
prefix. A complete package contains exactly these five HDF5 files:

- `dataset_train.hdf5`
- `dataset_valid.hdf5`
- `dataset_zero_action_valid.hdf5`
- `dataset_lstm_actuator_zero_action_valid.hdf5`
- `dataset_lstm_actuator_policy_valid.hdf5`

## Pipeline 1: solver-active canonical ContactTokens

Both encoders reuse the same immutable ContactTokens package. Substitute the
reviewing account's uploaded dataset subdirectory if it differs from the frozen
experiment name below.

```bash
CONTACT_TOKENS_DATASET=anymal-c-rough-newton-native-contact-tokens-sampling-v2-20260821-0630

submit_contact_review \
  shared-per-body-active \
  anymal_rough_newton_native_shared_per_body_active_filter \
  "${CONTACT_TOKENS_DATASET}" \
  0

submit_contact_review \
  global-attention-active \
  anymal_rough_newton_native_contact_tokens_active_filter \
  "${CONTACT_TOKENS_DATASET}" \
  0
```

The expected single-variable difference is the model encoder. Both runs must
resolve `contact_representation=contact_tokens` and
`contact_filter=solver_active` and must load the same five HDF5 objects.

## Pipeline 2: owner-routed solver-active native15

The no-self and self-inclusive cases intentionally use separate dataset
families.

```bash
RAW15_DATASET=anymal-c-rough-newton-native-raw15-sampling-v2-20260821-0630
RAW15_SELF_DATASET=anymal-c-rough-newton-native-raw15-self-sampling-v1-20260826-0447

submit_contact_review \
  owner-routed-active-no-self \
  anymal_rough_newton_native_active15 \
  "${RAW15_DATASET}" \
  0

submit_contact_review \
  owner-routed-active-with-self \
  anymal_rough_newton_native_active15_self \
  "${RAW15_SELF_DATASET}" \
  0
```

Expected resolved contracts:

| Case | Dataset representation | Model representation | Filter |
|---|---|---|---|
| No self-collision | `raw15_tokens` | `active15_tokens` | `solver_active` |
| With self-collision | `raw15_self_tokens` | `active15_self_tokens` | `solver_active` |

The self-inclusive dataset generation preset requires both zero token overflow
and observed self-collision tokens. Those are dataset validity gates, not model
quality claims.

## Pipeline 3: current sampling strategy

The current branch generates the new sampling strategy. The old strategy is
not selected by a YAML flag: reproduce an old/new comparison with frozen old
data and fixed-epoch checkpoints, rather than silently regenerating the old
side with current code.

Always generate into a new, previously unused dataset subdirectory. Never use
`off` or `auto` with one of the frozen dataset names above.

```bash
NEW_RAW15_DATASET="<new-unique-raw15-dataset-subdirectory>"
NEW_CONTACT_TOKENS_DATASET="<new-unique-contact-token-dataset-subdirectory>"

env -u OSMO_SWIFT_BASE -u AMLFS_DATA_ROOT \
  RUN_ID="generate-raw15-${REVIEW_TAG}" \
  ./osmo_scripts/start.sh \
    --preset anymal_rough_newton_native_raw15_dataset \
    --workflow-name contact-review-dataset \
    --dataset-subdir "${NEW_RAW15_DATASET}" \
    --storage-backend osmo_data \
    --osmo-swift-base "${REVIEW_OSMO_DATA_BASE}" \
    --osmo-data-credential "${REVIEW_OSMO_DATA_CREDENTIAL}" \
    --dataset-cache-mode off \
    --pool "${REVIEW_POOL}" \
    --priority NORMAL

env -u OSMO_SWIFT_BASE -u AMLFS_DATA_ROOT \
  RUN_ID="generate-contact-tokens-${REVIEW_TAG}" \
  ./osmo_scripts/start.sh \
    --preset anymal_rough_newton_native_contact_tokens_dataset \
    --workflow-name contact-review-dataset \
    --dataset-subdir "${NEW_CONTACT_TOKENS_DATASET}" \
    --storage-backend osmo_data \
    --osmo-swift-base "${REVIEW_OSMO_DATA_BASE}" \
    --osmo-data-credential "${REVIEW_OSMO_DATA_CREDENTIAL}" \
    --dataset-cache-mode off \
    --pool "${REVIEW_POOL}" \
    --priority NORMAL
```

After generation, require all five HDF5 files and record their sizes, schema
metadata, and content digests. Train with `require`, never `auto`, by passing
the new subdirectory to the corresponding training command above.

### Fixed-checkpoint old/new evaluation

`osmo_scripts/sampling_strategy_eval/` provides a standalone evaluator at
exactly `model_epoch199.pt`. It runs old and new three-seed checkpoint sets on
both validation distributions, using identical deterministic windows and
policy rollouts.

The legacy workflow filenames map as follows:

- `eval_a_osmo_data_workflow.yaml`: owner-routed native15 encoder;
- `eval_d_osmo_data_workflow.yaml`: global-attention encoder.

Read `osmo_scripts/sampling_strategy_eval/README.md` and verify the checkpoint
and dataset manifests before submission. In particular, the four old
global-attention dataset entries currently have no frozen size or SHA-256 in
`dataset_manifest_d.json`; fill those fields before describing that evaluation
as fully immutable. A fixed Epoch-199 checkpoint is unavailable for the old
shared per-body run, so the existing evaluator does not make a controlled
old/new claim for that encoder.

## Runtime acceptance checklist

For every training workflow, verify all of the following before comparing
metrics:

1. OSMO accepted the intended pool, NORMAL priority, and requested resources.
2. The log shows the expected cached dataset, not a generation fallback.
3. The resolved config reports the intended stored representation, model
   representation, encoder type, activity filter, and self-collision policy.
4. All eight distributed ranks initialize.
5. Dataset statistics complete and W&B initializes under the unique run name.
6. Epoch 0 finishes with finite train, validation, and rollout metrics.
7. Comparisons use the same completed epoch or the same sample/update budget;
   never compare each run's latest point when start times differ.
8. After 100 completed epochs, `model_epoch99.pt` is visible and can be fully
   downloaded and deserialized before any workflow is canceled.

Do not infer model quality from a dataset-generation success, cache hit, or
single Epoch-0 value. Use multiple seeds and a shared evaluation distribution
for architectural conclusions.
