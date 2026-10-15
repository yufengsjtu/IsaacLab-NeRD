# Control-step tasks and adding new ones

All maintained tasks except G1 share one optimized NeRD pipeline (the SO101/Allegro recipe): env-driven reset-safe
clips recorded per control step, one prediction per control step with joint-target input, unique-transition sliding
windows, GPU-resident data, FP16 training autocast, streaming FP16/compiled inference, closed-loop checkpoint
selection against the native MJWarp metric and post-training throughput benchmarks.

| Spec (`isaaclab_neural.tasks`) | Presets | Contacts | Auxiliary labels | Teacher |
|---|---|---|---|---|
| `cartpole` | `cartpole_control_step[_smoke]` | none | - | `cartpole_teacher` |
| `anymal_c_flat` | `anymal_flat_control_step[_smoke]` | 64 tokens | - | `anymal_flat_teacher` |
| `anymal_c_rough` | `anymal_rough_control_step[_smoke]` | 64 tokens | per-body net contact forces | `anymal_rough_teacher` |
| `franka_reach` | `franka_reach_control_step[_smoke]` | none | - | packaged Menagerie Reach policy |
| `franka_lift` | `franka_lift_control_step[_smoke]` | 64 tokens | finger-object normal forces | official Lift policy |
| `allegro_reorient` | `allegro_reorient_control_step[_smoke]` | 128 tokens | - | `allegro_reorient_teacher` |
| `so101_keyboard` | `so101_keyboard_control_step[_smoke]`, `so101_keyboard_randomized[_smoke]` | 128 tokens | robot normal forces | packaged Keyboard policy |

The physics-frame recipes (`anymal_*newton_native*`, `franka_*_mixed`, `cartpole_newton_native`, ...) remain available.

## Task specs

A `NerdTaskSpec` (`isaaclab_neural/tasks/specs.py`, standard library only) names four things per task:

- `policy_task`: native MJWarp task with CUDA graphs (teacher runner config, native evaluation, baselines);
- `record_task`: a twin of the policy task stepped frame by frame through Isaac Lab (`envs/recording.py`), which the
  collector needs to observe every physics frame;
- `nerd_task`: the same task with NeRD dynamics;
- `env_name`: the dataset directory name.

It also carries the teacher checkpoint, contact capacity (`0` for contact-free tasks), auxiliary labels, closed-loop
metrics and the selection metric. Tools take `--task-spec NAME` and explicit arguments override it:

```bash
python -m isaaclab_neural.generate.allegro_dataset --task-spec anymal_c_rough --record-control-steps \
  --collection-mode policy --checkpoint pretrained/control_policy/anymal_c_rough_control_step/model_1499.pt \
  --dataset data/datasets/demo/Anymal-C-Rough-ControlStep/dataset_policy_train.hdf5 --num-transitions 100000 \
  --horizon 32 --contact-filter solver_active_movable
python -m isaaclab_neural.eval.policy_rollout --task-spec anymal_c_rough \
  --checkpoint pretrained/control_policy/anymal_c_rough_control_step/model_1499.pt --report native.json
python -m isaaclab_neural.eval.policy_rollout --task-spec anymal_c_rough --nerd-checkpoint best_eval_model.pt \
  --checkpoint pretrained/control_policy/anymal_c_rough_control_step/model_1499.pt --report nerd.json \
  --video-dir video --follow-asset robot --camera-eye 2.2 2.2 1.3
python -m isaaclab_neural.eval.task_benchmark env --task-spec anymal_c_rough --num-envs 8192 \
  --checkpoint pretrained/control_policy/anymal_c_rough_control_step/model_1499.pt --report bench.json
```

OSMO presets set `experiment.task_spec`; `osmo_scripts/lib/preset.py` fills the task ids, teacher, contact capacity,
auxiliary recording flags and the evaluation/benchmark task arguments, and any key in the preset wins. Environment
sizes, data budgets and training overrides (`train_cfg_overrides`) stay in the preset, so one task can be resized or
re-targeted without code changes.

## Auxiliary force heads

- `finger_object_normal_forces`: recorded automatically when the task config declares
  `finger_object_normal_force_sensors` (Franka Lift); enable `network.finger_object_normal_force_head`.
- `body_net_contact_forces`: recorded with `--record-body-net-contact-forces` (preset key
  `record_body_net_contact_forces`, implied by the spec) from the scene sensor `contact_forces`; enable
  `network.body_net_contact_force_head` and `algorithm.loss.body_net_contact_force`.
- `robot_normal_forces`: the SO101 Keyboard pipeline's robot-key normal-force head.

## Adding a task

1. Environments: a native policy config, its recording twin (`apply_frame_stepping`) and a NeRD config with the same
   scene, actuators and contacts. Register the three ids in `isaaclab_neural/envs/__init__.py` with an RSL-RL entry.
   Stateful actuators (e.g. LSTMs) cannot be reproduced by a control-step model; use stateless ones.
2. A teacher: a packaged checkpoint or a `teacher_only` preset (see `cartpole_teacher.yaml`).
3. `register_task_spec(NerdTaskSpec(...))` in `tasks/specs.py`, choosing closed-loop metrics from
   `isaaclab_neural.eval.policy_rollout` or adding one there.
4. A training config copied from `train/cfg/*/transformer_*_control_step.yaml` (drop `inputs.contact_set` for
   contact-free tasks) and a preset plus smoke preset with `task_spec`.
5. Extend `CONTROL_STEP_TASK_PRESETS` in `test/test_osmo_preset_configs.py`, then run the smoke preset on OSMO.
