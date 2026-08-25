# Per-Body Contact Force Progress

## 2026-08-19

- Created `rowany/per-body-net-contact-force` from `contact_encoder_set` at
  `6ae58df788`.
- Frozen Stage 1 contract in `memory/task.md`.
- Added opt-in Newton ContactSensor recording, HDF5 schema validation, dataset
  loading, world-to-current-root label transforms, normalized auxiliary loss
  and metrics, checkpoint reconstruction, and runtime solver force caching.
- The labels are per-primary-body resultant linear contact forces [N], including
  normal and tangential/friction components and excluding moments/torques.
- Added rough Anymal-C shared-per-body contact-force smoke and formal configs.
  Both retain `contact_tokens`, 64-token capacity, body-frame state settings,
  and 17 runtime primary bodies. The auxiliary MLP is `[64]`; force loss weight
  is `0.1` and active-contact threshold is `1.0 N`.
- Added matching OSMO presets with isolated dataset subdirectories.
  `run_experiment.py` adds `--record-body-net-contact-forces` only when
  `record_body_net_contact_forces: true`; old presets remain unchanged.
  Force presets reject cached HDF5 files without compatible `[17, 3]` labels
  and metadata.
- Added user documentation and a changelog fragment. Documentation records
  label/output frames, post-solve timing, old-dataset incompatibility, Stage 2
  deferral, and exact local/OSMO smoke commands.
- Validation: 89 relevant force, shared-per-body, contact-token, Newton
  adapter, HDF5, dataset-loading, masking, and config tests passed. Three
  environment warnings were non-fatal (one unavailable-CUDA warning and two
  multiprocessing `fork` deprecation warnings).
- Validation: changed-file pre-commit passed all hooks after `ruff format`
  reformatted four changed Python files.
- Local runtime smoke generated all five rough Anymal datasets with
  `body_net_contact_forces_w` shape `[trajectory, time, 17, 3]` and
  matching schema metadata.
- The first local training attempt exposed that eager trajectory datasets did
  not publish `data_keys`; added the same metadata interface used by lazy
  datasets and regression coverage for eager transition and trajectory
  loaders.
- The retried one-epoch dual-head training completed. Validation force RMSE was
  `77.40 N` on `exp_trajectory` and `27.01 N` on
  `zero_action_trajectory`; checkpoints were saved.
- A four-step, two-environment rough NeRD evaluation loaded the force-enabled
  checkpoint and completed successfully, exercising the online auxiliary cache
  without changing state stepping.
- Full pre-commit passed. The final focused regression suite passed 85 tests
  before the eager `data_keys` fix; the focused post-fix suite passed 15 tests.
- Runtime label plausibility on the local zero-action dataset:
  `[64, 400, 17, 3]`, all finite, mean summed force
  `[0.19, -0.26, 510.24] N`, and `23.0%` active body slots above `1 N`.
- The historical OSMO smoke run ID
  `anymal-c-rough-shared-per-body-normal-force-smoke-20260819-080142-1`
  failed before starting because its assigned node had an unhealthy GPU.
  This was a backend allocation failure, not a code failure.
- Retried on `isaac-dev-l40s-04` under the historical run ID
  `anymal-c-rough-shared-per-body-normal-force-smoke-retry-20260819-081158-1`.
  It completed with exit code 0 using Swift storage: generated and uploaded all
  five force-label datasets, trained one epoch, saved checkpoints, and uploaded
  13 output files. Validation force RMSE was `76.93 N` on `exp_trajectory` and
  `28.29 N` on `zero_action_trajectory`.
- Final regression suite passed 85 tests with two pre-existing multiprocessing
  deprecation warnings. Final full `./isaaclab.sh -f` passed every hook.
- Added explicit per-body RMSE keys and a ContactSensor reconstruction RMSE
  alias; the focused metric/model suite passed 11 tests after this final
  addition, and a second full pre-commit run passed without modifications.
- All Stage 1 implementation and validation tasks are complete.
- No commit was created and no formal OSMO workflow was submitted.

## 2026-08-25

- Rejected active/inactive balanced force loss after formal OSMO training did
  not improve policy-trajectory force metrics over normalized MSE.
- Rejected the shared per-body force decoder after the formal OSMO ablation
  `anymal-c-rough-shared-per-body-contact-force-formal-20260824-051930-1`.
  At epoch 359, policy-trajectory RMSE was `13.59 N` (`32.91 N` active)
  versus `12.63 N` (`31.31 N` active) for the global decoder. Zero-action
  RMSE was `11.25 N` (`20.45 N` active) versus `9.80 N` (`17.55 N` active).
- Retained the global force decoder and normalized MSE objective.
