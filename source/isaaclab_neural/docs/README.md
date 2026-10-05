# IsaacLab Neural

`isaaclab_neural` integrates NeRD neural dynamics with Isaac Lab's Newton
backend. The public release supports local dataset generation, model training,
evaluation, RSL-RL, and optional OSMO execution.

Choose a path:

1. [Getting Started](getting_started.md) — install the extension and understand
   the common workflow.
2. [Cartpole Tutorial](cartpole.md) — run the smallest contact-free example.
3. [Anymal-C Guide](anymal.md) — work with flat or rough terrain and native
   contacts.
4. [Franka Guide](franka.md) — contact-free Reach and contact-aware cube lift.
5. [Allegro Guide](allegro.md) — in-hand cube reorientation with the hand and
   cube predicted together.
6. [Reinforcement Learning](rl.md) — train or evaluate a policy on NeRD.
7. [Advanced Features](advanced.md) — contact tokens, shared-per-body encoding,
   contact-force prediction, RoPE, and KV-cache inference.
8. [OSMO Reference](osmo.md) — submit one of the maintained remote presets.

The deprecated `fixed_ground` mode is intentionally rejected after emitting a
deprecation warning. New code must use `newton_native`; set
`num_contacts_per_env: 0` only for contact-free systems such as Cartpole and
Franka Reach.

G1 development lives in a separate worktree and is not included in this
release.

## Fixed SO101 keyboard data and replay

The keyboard prototype supports native MJWarp data collection and an experimental
NeRD training/inference path. The learned model is **not task-fidelity-qualified**.
Collection copies `IsaacContrib-Keyboard-SO101`, selects one keyboard variant, and freezes both
keyboard-root reset paths at the authored pose. All 108 keys, robot resets,
reset curriculum, rewards, terminations, and the 100 Hz physics / 25 Hz control
intervals remain enabled. The original task configuration is not modified.

From the repository root, in an environment with Isaac Lab and NeRD dependencies:

```bash
export PYTHONPATH="$PWD/source/isaaclab_neural${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m isaaclab_neural.generate.keyboard_dataset collect \
  --dataset logs/keyboard_poc/press_v2.hdf5 \
  --report logs/keyboard_poc/collect.json \
  --num-envs 8 --control-steps 420 --horizon 64 \
  --collection-mode press --script-action-limit 5 \
  --keyboard-variant 0 --seed 0 --device cuda:0 --visualizer none

uv run python -m isaaclab_neural.generate.keyboard_dataset replay \
  --dataset logs/keyboard_poc/press_v2.hdf5 \
  --report logs/keyboard_poc/replay.json \
  --device cuda:0 --visualizer none
```

Existing dataset/report paths are rejected rather than overwritten. `zero` and
`random` collection modes are also available. The DLS press probe is a data
source, not a trained typing policy; its action limit does not change the task's
action scale or actuator limits. Inspect actual key/contact coverage before
using a small collection as training data.

The recorder observes the original `env.step()` after each physics-frame scene
update and before automatic resets. It stores world-frame generalized states,
backend-order shadow-PD inputs, full native contact CSR, and post-solve robot
normal-force vectors in newtons. Shadow-PD observation restores the original
control buffer so analytic PD is not applied twice. The robot force labels are
not Franka finger-object labels or per-body total/friction force labels.

Training clips never cross resets. A terminal tail may overlap an earlier clip
to retain the terminal transition. Consequently, reported edges inside clips
are not counts of unique events. `key_pressed` is a 100 Hz physical occupancy
label; `typed_after` follows the original 25 Hz command update, with terminal
reset rows excluded from command-state comparisons.

New collections also store a `synchronous_world_timeline_v1` replay payload under
`/replay/frames` and `/replay/contexts`. Replay restores the initial physical/MDP
state and MJWarp warm start, then advances all source worlds on their original
shared timeline. Recorded reset outcomes are restored only for worlds where both
reference and replay have actually terminated. Ongoing episodes receive no state
corrections, and termination mismatches remain failures. This validates recorded
action dynamics, not the stochastic reset sampler itself.

Preserving world slots alone is insufficient: assembling independently clipped
worlds from different source times changes the batched solver inputs and can
produce contact-sensitive replay drift even in deterministic mode. Older files
without the synchronous payload retain the legacy clip-replay path; they cannot
be retroactively certified by the new protocol. Omitting `--num-envs` uses the
recorded world count. Arbitrary relocation/reordering is rejected.

The JSON report checks positions, velocities, force labels, key occupancy,
typing state and termination flags. A failed check returns a non-zero exit code;
collecting a file alone is not a declaration that it is suitable for training.
Any future NeRD FPS comparison must use a matching fixed-keyboard MJWarp
baseline, not the original 32-variant task as an unmatched comparator.
Synchronous replay always checks the complete recording; `stored_horizon` is the
training-clip length and `replay_horizon` is the full physics-frame count per world.
`--trajectory-index` and `--replay-horizon` are accepted only for legacy clip files.
A passing prefix never certifies the remaining frames.

### Visual review of recorded data

Full synchronous replay can record an MP4 without an X/Wayland desktop. Use a fresh
report path and video directory; existing video directories are not overwritten:

```bash
uv run python -m isaaclab_neural.generate.keyboard_dataset replay \
  --dataset logs/keyboard_poc/data.hdf5 \
  --report logs/keyboard_poc/replay_video.json \
  --video-dir logs/keyboard_poc/videos \
  --video-env-index 0 --device cuda:0
```

The headless Newton GL view selects one recorded world and offsets the task camera
by that world's recorded environment origin. All source worlds and the complete
physics timeline still replay and are validated with the original thresholds.
Video frames are captured at control boundaries after matched recorded resets are
restored; ongoing episodes are never corrected. Video FPS follows the environment's
control timestep. Do not pass `--num-envs 1` to isolate a world physically.

For an interactive window, run the replay command in a desktop terminal with
`--visualizer newton_gl --max_visible_envs 8` and omit `--video-dir`. An SSH session
without a display should use headless video instead. Videos help inspect behavior
but are not a substitute for the replay report, and policy-evaluation videos are
not replays of an HDF5 recording.

### Policy collection

`keyboard_dataset collect --collection-mode policy --checkpoint PATH` records the
native typing policy in the same fixed scene (variant 0, authored keyboard pose).
The randomized 32-variant scene and its `so101_keyboard_original_v1` schema were
removed; such files are rejected and must be recollected. The policy uses the
current registered RSL-RL agent configuration and compatibility loader, matching
native task playback; the resolved agent and checkpoint SHA-256 are recorded in the
dataset contract. Replay restores the recorded seed, variant and world count.

`--deterministic-physics` explicitly requests Newton run-to-run determinism for a
new collection and selects eager physics (`use_cuda_graph=False`) to avoid
capture-time worst-case deterministic workspaces. Replay restores both settings
from metadata. Eager execution alone does not eliminate workspace scaling limits;
use bounded shards or the audited `--bounded-deterministic-workspace` profile, not
an arbitrary large world count. This is distinct from the launcher's
rendering-related `--deterministic` flag. The physics request also
disables MuJoCo's internal sensor stage, so confirm native contact-force labels
remain valid. Do not mix deterministic and non-deterministic collections or assume
that enabling the flag guarantees sufficient workspace capacity or a passing replay.

Model inputs are the generalized state embedding, `joint_f`, `gravity_dir` and
contact tokens. Physical-parameter and articulation-pose conditioning were removed
with the randomized scene: in one fixed scene they are constant and carry no
information. Use independently collected validation shards and keep
`--eval-interval 0`; `isaaclab_neural.eval.keyboard_checkpoint` evaluates
synchronized source-time windows and stops scoring each world at its first teacher
or student termination. Checkpoint execution is not a claim of dynamics fidelity.

A passing collection is not a passing replay. Keep formal collection gated on
replay validation, and qualify learned multi-step errors and key events separately
before using NeRD for policy training. Do not certify full recordings using only
shorter passing prefixes.

### Native typing policy evaluation

`python -m isaaclab_neural.eval.keyboard_policy` evaluates a native RSL-RL checkpoint
on the fixed keyboard with `--checkpoint PATH --report PATH --reset-mode uniform`.
Use `--reset-mode training` to retain the reset snapshot curriculum. Neither mode
forces an empty typed buffer; inspect the episode-level initial buffer lengths.
The evaluator uses the current task environment configuration, not the checkpoint's
saved environment configuration. Agent configuration is loaded from the checkpoint's run.

For isolated reset-pose comparisons, `--reset_ik_iters 32` overrides the IK iteration
budget and records it as `reset_ik_iters` in the report. Omit the option to evaluate
the task's original reset distribution. Near-target-reset success is not evidence
of success under the original reset distribution. Reports include completed steps,
finite-state checks and clean episode success counts; `policy_accepted` remains false
until metrics and recorded behavior have been reviewed.

### Small Keyboard NeRD prototype

Importing `isaaclab_neural.envs` registers `IsaacContrib-Keyboard-SO101-NeRD`.
It is a manager-based, fixed-variant-0 task with a hard-wired NeRD backend, not a
multi-backend preset. It reuses the original RSL-RL config but requires a NeRD
checkpoint; PPO training is not validated yet. Like the other optional NeRD
tasks, it is documented here rather than in the default core environment browser.

The small recipe uses a two-layer, width-64 Transformer with four physics frames
of history and a shared-per-body contact encoder supporting 128 other-body slots.
Its 228-dimensional state output is accompanied by separate SO101 robot normal
force and contact-activity heads. The checkpoint records the seven-body order,
world frame, newtons, and post-solve label semantics. The corresponding USD
`*_link` body names are accepted in exactly the same order; arbitrary body
reordering is still rejected. Friction is not predicted;
publication clears friction source buffers so the task's normal-force history
matches the learned normal vectors. Do not interpret the reconstructed aggregate
force channel as a learned total-force estimate including friction.

Collect validation in a separate run with a different seed. Do not randomly
split overlapping clips from the same collection. The trainer checks dataset
contracts, seed separation and the 0.01 s dynamics timestep; normalization uses
only training data. The recipe currently expects `press_v2.hdf5` and
`valid_seed1.hdf5` under `logs/keyboard_poc`; override its dataset paths if needed.

```bash
uv run python -m isaaclab_neural.generate.keyboard_dataset collect \
  --dataset logs/keyboard_poc/valid_seed1.hdf5 \
  --report logs/keyboard_poc/collect_valid_seed1.json \
  --num-envs 8 --control-steps 420 --horizon 64 \
  --script-action-limit 5 --seed 1 --device cuda:0 --visualizer none

uv run python -m isaaclab_neural.train.train \
  --task IsaacContrib-Keyboard-SO101-NeRD \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/SO101/transformer_keyboard_small.yaml \
  --num-envs 8 --device cuda:0 --visualizer none \
  --eval-interval 0 --save-interval 5 --logdir logs/keyboard_poc/train_small
```

The dedicated checkpoint evaluator replaces generic training-time rollout eval;
keep `--eval-interval 0`. Set `CHECKPOINT` to the produced
`nn/best_valid_independent_seed_model.pt` inside the timestamped run directory:

```bash
uv run python -m isaaclab_neural.eval.keyboard_checkpoint \
  --checkpoint "$CHECKPOINT" --dataset logs/keyboard_poc/valid_seed1.hdf5 \
  --report logs/keyboard_poc/checkpoint_rollout.json \
  --horizon 16 --device cuda:0 --visualizer none
```

This evaluates synchronous source-time windows for new recordings, or source-world
clips for legacy files. It restores only the initial physical/MDP state,
cold-starts history and advances recorded actions. Scoring
stops for a world after the first teacher or student terminal boundary. It checks
one learned step per physics frame and sensor/cache equality. `execution_passed`
means finite execution and correct force publication, **not** successful typing
or accepted simulation fidelity. Diagnostic recording overhead makes this
unsuitable for FPS measurements.

The initial fixed-keyboard 20-epoch development run produced a loadable checkpoint,
but 16-frame rollouts drifted beyond key travel and generated false key presses.
Its legacy clip-only native replay also had a strong-contact mismatch; new
synchronous recordings address that separate replay-protocol limitation.
A later randomized-scene pilot (since removed) also had substantial 16-frame state
and key-event errors. Neither pilot was task-fidelity-qualified for PPO training or
simulation speed comparisons.

### Keyboard model, loss and inference

Both SO101 recipes (`transformer_keyboard_throughput.yaml` for OSMO and
`transformer_keyboard_small.yaml` for local runs) use the K model and losses, the
configuration that matched native closed-loop typing, with a streaming trunk and FP16
inference:

- `network.key_head`: the Transformer trunk sees robot coordinates, robot `joint_f`,
  gravity, non-key contact latents and an order-independent sum over per-key
  features. One decoder with shared parameters predicts each key's position and
  velocity change from the trunk context, that key's last `history_steps` states and
  contact latent, and a learned key embedding. Key inputs and outputs use statistics
  pooled over all keys; outputs are scattered back into the original prediction order,
  so the solver interface is unchanged. It requires the `shared_per_body` contact
  encoder, identity low-dim inputs `[states_embedding, joint_f, gravity_dir]`, output
  normalization and full history windows (the feature/KV caches are rejected; the frame
  cache below is supported).
  `key_q_start`, `num_keys` and `key_body_slot_start` are checked against the
  recorded joint and body names.
- `keyboard_state_group_weights: {robot: 0.5, key: 0.5}`: replaces the plain mean over
  228 coordinates, where 216 key coordinates dominated the six-joint arm, with
  `0.5 * mean(robot q/qd) + 0.5 * mean(key q/qd)`. Requires the MSE loss.
- `keyboard_key_position_scale_m: 0.0002`: key-position weights use a 0.2 mm physical
  scale instead of variance weights truncated by the `1e-5` floor.
- `keyboard_key_velocity_weighting: pooled`: all key velocity changes share one weight
  from statistics pooled over keys, matching the shared decoder.
- `network.transformer.position_encoding: relative_lag` with `attention_spans`
  (`[3, 3, 3, 2, 2, 2]` for six layers): layer `l` attends only to lags `0..span-1` with a
  learned per-head lag bias, so the receptive field `sum(span - 1) + 1` equals
  `num_states_history`. Training uses the same banded masks, and inference streams one
  new position per layer and physics frame, matching full-window evaluation up to FP32
  roundoff.
- `neural_solver_cfg.use_frame_cache: true`: only the newest frame is preprocessed and
  encoded each physics frame; earlier encodings, per-layer keys and values are reused,
  and reset rows rebuild their cone from the cached window. Heads decode only the newest
  step.
- `neural_solver_cfg.inference_precision: fp16`: CUDA autocast for online inference
  only (training stays FP32). `keyboard_policy` and `keyboard_benchmark env` accept
  `--inference-precision` and `--frame-cache/--no-frame-cache` to override a
  checkpoint's settings.

Loss overrides are applied after checkpoint training-state restoration and recomputed
on every start, so resuming does not compound them. Training logs
`<group>_weighted_state_MSE` for `robot_q`, `robot_qd`, `key_q` and `key_qd`.

On 8x L40 with the full 10M/3M fixed-v2 data (preset `so101_keyboard_resident_unique`,
2026-09-29), `model_epoch99.pt` reached 94.7% closed-loop typing success under
`eval/keyboard_policy.py --nerd-checkpoint` (3 seeds x 64 worlds), matching native
MJWarp's 94.7%. The same objective without the key head fell to 4.2% through
arm-velocity terminations, and short local runs (9k updates) ranked the two the other
way. The best-validation checkpoint (Epoch 51, 87.4%) was worse than Epoch 99 because
the force head overfit, so select checkpoints by closed-loop evaluation.

The streaming recipe (2026-10-01, same data and platform) matched K epoch for epoch on
validation metrics. Offline closed loop gave 95.5% for its best in-training-eval
checkpoint (Epoch 249; 95.3% with FP16 inference) versus 95.2-95.5% for K Epochs
99/299/699 and 94.7% native. At 8192 worlds on an idle RTX PRO 5000 Blackwell it ran
44.1k control steps/s with FP16 (MJWarp 52.3k; K with FP16 and the frame cache 31.3k;
the original K implementation 7.1k). Checkpoints trained with the earlier
learned-absolute trunk still load and run with the frame cache, but cannot be resumed
with these recipes.

### In-training closed-loop evaluation

The throughput recipe sets `algorithm.eval.mode: keyboard_policy` with `interval: 5`.
Every fifth epoch, rank 0 runs the typing policy (`policy_checkpoint`) for
`control_steps` control steps in the training environment, whose NeRD solver shares
the model being trained (no checkpoint export). The environment has 64 worlds, and
each evaluation reseeds a forked RNG from `seed + epoch`. W&B receives
`eval_closed_loop/success_rate` and the rates of `excessive_contact`, `abnormal_robot`,
`timeout_no_key`, `timeout_wrong_key` and `timeout_incomplete`, the same exclusive
categories `eval/keyboard_policy.py` reports, plus episode counts and lengths. The best
success rate saves `best_eval_model.pt` (the W&B file copy is not refreshed while the
run is active). Loading K Epoch 99 into the
trainer gave 95.3-95.8% over three evaluations (about 1,120 episodes each), consistent
with offline evaluation. Keyboard training rejects every other evaluation mode because
the generic rollout evaluator does not preserve Keyboard worlds, contacts or force
heads. `--eval-interval` defaults to 5 and overrides the recipe.

### Split throughput measurement

`python -m isaaclab_neural.eval.keyboard_benchmark model --nerd-checkpoint CKPT` times
one full-window model call on GPU-resident synthetic inputs over a batch sweep, with
optional `--autocast fp16|bf16`. It excludes the simulator, collision detection, contact
packing and input preparation. `... env --num-envs N [--nerd-checkpoint CKPT]` times
policy-driven environment steps with MJWarp or NeRD. A synchronized phase splits
physics from the rest of the step, and NeRD `step_profile` sections are recorded. Run
both modes on an otherwise idle GPU. With K Epoch 99 at 4096 worlds, the model alone
took about 110 ms per call (four calls per control step), capping NeRD below 10k
control steps/s before any simulator cost.

Training logs separate robot/key position and velocity errors in physical units.
Checkpoint rollout reports include per-frame error curves, true/false pressed
states, missed presses, and requested versus actually scored coverage. A student
that terminates early does not receive credit for the excluded tail; inspect
coverage and termination disagreements as well as state error.

Controlled local experiments found matching offline/online encoded inputs and
predictions, and no benefit from restoring available true prehistory on the
matched subset. Increasing training alone helped, but explicit position scales
reduced false presses substantially at the same update budget. Frozen-checkpoint
evaluation on a newly collected seed still showed position/force outliers and
incorrect task events. The diagnostic configuration is therefore not qualified
for PPO or FPS claims; more training is not a substitute for those checks.
