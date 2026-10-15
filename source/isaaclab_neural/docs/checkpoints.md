# Archived NeRD checkpoints

The NV-Datasets dataset `IsaacLab-NeRD-Checkpoints` holds one selected NeRD dynamics
checkpoint per task, archived on 2026-10-15 (see [release.md](release.md) for which entries the
control-step release keeps or replaces) from the OSMO
outputs (NV-Datasets `IsaacLab-NeRD-Output`, Swift `isaaclab-nerd-output`) and local
retrievals. Each directory holds `model.pt`, the training config `train_cfg.yaml`,
`metadata.json` (source workflow, preset, branch, selection rule, metrics, epoch, SHA-256)
and `rollout_mjwarp_vs_nerd.mp4`. `manifest.json` lists all entries. Teacher policies are in
git under `pretrained/`.

| Path | Epoch | Source workflow | Key metric | Rollout video |
|---|---:|---|---|---|
| `so101_keyboard/control_step` | 299 | `so101-keyboard-control-step-20261002-183811-1` | closed loop 95.3% (6 seeds); MJWarp 95.3% | [so101_keyboard_control_step.mp4](media/checkpoints/so101_keyboard_control_step.mp4) |
| `allegro_reorient/control_step` | 299 | `allegro-reorient-control-step-20261007-004031-1` | 16.9 goals/episode, 1.8% drops (MJWarp 15.8, 3.8%); 482.8k vs 127.2k control steps/s at 8192 envs | [allegro_reorient_control_step.mp4](media/checkpoints/allegro_reorient_control_step.mp4) |
| `franka/reach` | 359 | `franka-reach-mixed-formal-20260916-155222-2ce27c-1` | 10-step rollout MSE 3.66e-6 | [franka_reach.mp4](media/checkpoints/franka_reach.mp4) |
| `franka/lift` | 234 | `franka-lift-mixed-warpfix-wandb-20260917-131842-1` | 10-step rollout MSE 0.0441; not closed-loop faithful (see video) | [franka_lift.mp4](media/checkpoints/franka_lift.mp4) |
| `anymal_c/rough` | 579 | `anymal-c-rough-active-filter-owner-body-20260903-063515-1` | 10-step rollout MSE 0.0148 | [anymal_c_rough.mp4](media/checkpoints/anymal_c_rough.mp4) |
| `anymal_c/flat_legacy` | 269 | `newton_native_abl_new_collision_pipeline-20260713-053718-1` | 10-step rollout MSE 0.00154; July 2026 code, does not load with current code | none (incompatible with current code) |
| `g1/flat` | 974 | `g1-mixed-jointf-baseline-20260913-140349-1` | 10-step rollout MSE 0.00411 | [g1_flat.mp4](media/checkpoints/g1_flat.mp4) |

Each video (also in the dataset as `<path>/rollout_mjwarp_vs_nerd.mp4`) shows the same teacher policy, seed and camera side by side: left under
native MJWarp, right under the archived NeRD model (one world, headless Newton GL). The
rollout notes, including known failures, are in each `metadata.json` (`video_note`):

- SO101 Keyboard: 10/11 clean successes under both backends (450 control steps).
- Allegro: 14.3 goals/episode under MJWarp, 17.0 under NeRD, no drops (900 control steps).
- Franka Reach: NeRD tracks the targets like MJWarp.
- Franka Lift: MJWarp lifts the cube; under NeRD the cube is not lifted and leaves the table.
- ANYmal-C rough and G1 flat: the camera follows the robot; no falls under NeRD. Velocity
  commands are sampled independently per backend.

Rollout MSE values are only comparable within one task and data recipe; Allegro NeRD is more
optimistic than MJWarp (fewer drops), so validate fidelity before trusting rewards.

Training these models from scratch uses the listed preset on the listed branch; the
checkpoints let you evaluate the archived results without retraining. Download with the repository helper (needs `NGC_API_KEY` and `NVDATASET_TENANTID`):

    python osmo_scripts/pull_nvdataset.py IsaacLab-NeRD-Checkpoints --prefix <path> --output-dir <dir>

Verify `sha256sum model.pt` against `metadata.json`, then evaluate with the `eval_command` in
`metadata.json` (where given) from the listed branch.
