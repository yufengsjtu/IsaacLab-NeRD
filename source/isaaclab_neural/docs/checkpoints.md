# Archived NeRD checkpoints

The NV-Datasets dataset `IsaacLab-NeRD-Checkpoints` holds one selected NeRD dynamics
checkpoint per task, archived on 2026-10-15 from the OSMO
outputs (NV-Datasets `IsaacLab-NeRD-Output`, Swift `isaaclab-nerd-output`) and local
retrievals. Each directory holds `model.pt`, the training config `train_cfg.yaml` and
`metadata.json` (source workflow, preset, branch, selection rule, metrics, epoch, SHA-256).
`manifest.json` lists all entries. Teacher policies are in git under `pretrained/`.

| Path | Epoch | Source workflow | Key metric |
|---|---:|---|---|
| so101_keyboard/fixed_physics_frame | 249 | so101-keyboard-stream-20260930-034532-1 | closed loop 95.5% offline / 94.8% (6 seeds); native 95.3% |
| so101_keyboard/control_step | 299 | so101-keyboard-control-step-20261002-183811-1 | closed loop 95.3% (6 seeds); native 95.3% |
| so101_keyboard/randomized_control_step | 199 | so101-keyboard-randomized-20261005-035103-1 | closed loop 92.9% (6 seeds); native 94.1% |
| allegro_reorient/control_step | 299 | allegro-reorient-control-step-20261007-004031-1 | 16.9 goals/episode, 1.8% drops (MJWarp 15.8, 3.8%); 482.8k vs 127.2k control steps/s at 8192 envs |
| franka/reach | 359 | franka-reach-mixed-formal-20260916-155222-2ce27c-1 | 10-step rollout MSE 3.66e-6 |
| franka/lift | 234 | franka-lift-mixed-warpfix-wandb-20260917-131842-1 | 10-step rollout MSE 0.0441 |
| anymal_c/rough | 579 | anymal-c-rough-active-filter-owner-body-20260903-063515-1 | 10-step rollout MSE 0.0148 |
| anymal_c/flat_legacy | 269 | newton_native_abl_new_collision_pipeline-20260713-053718-1 | 10-step rollout MSE 0.00154 (July 2026 code) |
| g1/flat | 974 | g1-mixed-jointf-baseline-20260913-140349-1 | 10-step rollout MSE 0.00411 |

Rollout MSE values are only comparable within one task and data recipe. SO101 and Allegro
closed-loop numbers use the evaluation protocol recorded in `metadata.json`; Allegro NeRD is
more optimistic than MJWarp (fewer drops), so validate fidelity before trusting rewards.

Training these models from scratch uses the listed preset on the listed branch; the
checkpoints let you evaluate the archived results without retraining. Download with the
repository helper (needs `NGC_API_KEY` and `NVDATASET_TENANTID`):

    python osmo_scripts/pull_nvdataset.py IsaacLab-NeRD-Checkpoints --prefix <path> --output-dir <dir>

Verify `sha256sum model.pt` against `metadata.json`, then evaluate with the `eval_command` in
`metadata.json` (where given) from the listed branch.
