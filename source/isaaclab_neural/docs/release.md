# Release validation for a tagged `rowany/nerd-optimized` version

This page records what a release tag of the unified NeRD branch has to prove, what has already been verified, and which
runs are still required. G1 is out of scope for this release (it lives on `feature/g1-nerd` and
`feature/g1-29dof-nerd`).

A full retrain of every task is **not** required to establish correctness. Code correctness is established by the CPU
test suite and one end-to-end OSMO smoke run per task on the tagged commit; numerical results are re-established only
where the recipe or its code path changed. Retraining recipes whose code path is unchanged would only re-sample
noise (one SO101 seed alone varied between 93.5% and 95.5% closed-loop success across FP32 runs), not test the code.

## Release candidate

| Item | Value |
|---|---|
| Branch | `rowany/nerd-optimized` (GitHub `yufengsjtu/IsaacLab-NeRD`) |
| Candidate commit | `5a55360a88` (2026-10-15) |
| Proposed tags | `nerd-v1.0.0-rc1` now; `nerd-v1.0.0` on the same commit once every required run below passes |
| Image | `docker.io/rowany28070/isaac-lab-nerd:so101-newton160-cu130-sdk095` (Newton 1.6.0) |

If a required run exposes a bug, fix it, cut `-rc2` and repeat only the smoke runs and the formal runs whose code path
the fix touches.

## Verified on the candidate

**CPU tests.** `source/isaaclab_neural/test`: 731 passed, 6 skipped (`CUDA_VISIBLE_DEVICES=""`, with `pytest-mock`).

**OSMO end-to-end smoke runs.** Each runs the teacher gate, multi-GPU env-driven collection, training with closed-loop
evaluation and the throughput benchmarks at small sizes (two epochs; the models are not meaningful).

| Task spec | Smoke preset | Workflow | Native teacher metric (gate) |
|---|---|---|---|
| `cartpole` | `cartpole_control_step_smoke` | `cartpole-control-step-smoke-20261015-102132-1` | failure rate 0, full 300-step episodes |
| `anymal_c_flat` | `anymal_flat_control_step_smoke` | `anymal-flat-control-step-smoke-20261015-094142-1` | teacher: failure rate 2.3%, tracking error 0.12 m/s |
| `anymal_c_rough` | `anymal_rough_control_step_smoke` | `anymal-rough-control-step-smoke-20261015-104703-1` | teacher: failure rate 4.7%, tracking error 0.17 m/s |
| `franka_reach` | `franka_reach_control_step_smoke` | `franka-reach-control-step-smoke-20261015-102238-1` | end-effector error 0.038 m |
| `franka_lift` | `franka_lift_control_step_smoke` | `franka-lift-control-step-smoke-20261015-094041-1` | pose success rate 0.548 |
| `allegro_reorient` | `allegro_reorient_control_step_smoke` | `allegro-reorient-control-step-smoke-20261015-090044-1` | smoke teacher |
| `so101_keyboard` | `so101_keyboard_control_step_smoke` | `so101-keyboard-control-step-smoke-20261015-090145-1` | packaged teacher |

The smoke runs for Cartpole, rough ANYmal-C and Franka Reach used commits between `f9504bd574` and `5a55360a88`; the
later commits only fix the frame cache for world-frame states and the body-force order of weighted mixtures, which those
runs exercised. The ANYmal flat, Lift, Allegro and SO101 smoke runs predate those two fixes; neither fix changes their
code path (body-frame states, no body-force head).

**Archived checkpoint compatibility.** Every model in `IsaacLab-NeRD-Checkpoints` that this release keeps (SO101
control step, Allegro control step) and the legacy Franka and rough ANYmal-C models load with strict `state_dict`
matching under the candidate code, and a forward pass on synthetic inputs is finite. This checks the architecture and
the checkpoint format only, not closed-loop behaviour.

**Unchanged code paths.** Relative to `rowany/allegro-nerd` (where the SO101 and Allegro results were produced), the
candidate does not change the Keyboard environments, collector, replay validation or evaluator, the NeRD physics
manager, the training loss or the data loaders' sampling. Shared changes are additive (task-spec registry,
`policy_rollout` evaluation mode, contact-free and body-force collector options, a mixture property) or guards
(`use_frame_cache` for world-frame states).

## Per-task decision

| Task | Result before the release | Decision | Required run |
|---|---|---|---|
| SO101 Keyboard (control step) | archived epoch 299, 95.3% over 6 seeds | keep; code path unchanged | closed-loop re-evaluation of the archived model (6 seeds) |
| Allegro (control step) | archived epoch 299, 16.9 goals/episode; 482.8k control steps/s | keep; code path unchanged | closed-loop re-evaluation of the archived model |
| Franka Lift | physics-frame model fails closed loop | replace with the control-step recipe | `franka_lift_control_step` (running: `franka-lift-control-step-20261015-061955-1`) |
| ANYmal-C flat | July 2026 model no longer loads | replace with the control-step recipe | `anymal_flat_control_step` (running: `anymal-flat-control-step-20261015-065648-1`) |
| ANYmal-C rough | physics-frame model with the LSTM actuator, trained on `contact_encoder_set` | replace (stateless actuator, control step, body-force head) | `anymal_rough_control_step` |
| Franka Reach | physics-frame model from `rowany/franka-nerd` | replace with the control-step recipe | `franka_reach_control_step` |
| Cartpole | none on this branch | new | `cartpole_control_step` |

The superseded physics-frame entries stay in `IsaacLab-NeRD-Checkpoints` as legacy references; their presets remain in
the repository but are not part of this release's validated set.

## Acceptance criteria

- **Re-evaluations (SO101, Allegro):** closed-loop metrics within the run-to-run spread already recorded (SO101: pooled
  six-seed success compared with the archived 95.3%, single FP32 seeds already spanned 93.5% to 95.5%; Allegro goals per episode and drop rate within one standard error of the archived
  numbers). A larger deviation is a code regression and blocks the tag.
- **New recipes:** the run completes; `best_eval_model.pt` is the checkpoint closest to the native metric that the
  teacher gate reported; that closed-loop metric is within a task-specific tolerance of native and the MJWarp vs NeRD
  side-by-side video shows no qualitative failure (falls, dropped or unlifted objects, divergence). Throughput is
  reported from the 8192-world benchmark, not from smoke runs.
- **Archive:** every accepted model is uploaded to `IsaacLab-NeRD-Checkpoints` with `metadata.json` recording the tag,
  commit, workflow ID, selection rule, metrics and SHA-256, and `checkpoints.md` is updated.

## Running the required jobs

Check free quota first and avoid pools with storage failures (see `AGENTS.md`):

```bash
osmo pool list --mode free
./osmo_scripts/start.sh --preset anymal_rough_control_step --pool isaac-lab-l40-07
./osmo_scripts/start.sh --preset franka_reach_control_step --pool isaac-lab-l40-07
./osmo_scripts/start.sh --preset cartpole_control_step --pool isaac-lab-l40-07
```

Submit from the isolated NV-Datasets SDK environment described in `AGENTS.md`. Each formal preset needs eight GPUs;
in `isaac-lab-l40s-03` pass `--platform ovx-l40s`.

Re-evaluate an archived model with the tagged code (SO101 uses its Keyboard evaluator; the archived `metadata.json`
holds the exact `eval_command`):

```bash
python osmo_scripts/pull_nvdataset.py IsaacLab-NeRD-Checkpoints --prefix allegro_reorient/control_step \
  --output-dir checkpoints
python -m isaaclab_neural.eval.allegro_reorient_policy \
  --checkpoint pretrained/control_policy/allegro_reorient/model_4999.pt \
  --nerd-checkpoint checkpoints/allegro_reorient/control_step/model.pt --num-envs 256 --control-steps 1500 \
  --report allegro_reeval.json
```

## Tagging

```bash
git tag -a nerd-v1.0.0-rc1 5a55360a88 -m "NeRD control-step pipeline release candidate (all tasks except G1)"
git push git@github.com:yufengsjtu/IsaacLab-NeRD.git nerd-v1.0.0-rc1
```

Create `nerd-v1.0.0` on the same commit after the acceptance criteria hold, and record the final workflow IDs and
metrics in the table above and in [checkpoints.md](checkpoints.md).
