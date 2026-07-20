# IsaacLab-NeRD Local Evaluation

本文档说明如何在本地用 `isaaclab_neural.eval.eval` 做 NeRD checkpoint smoke evaluation，覆盖：

- Cartpole fixed-ground
- Anymal-C fixed-ground
- Anymal-C Newton native contacts

所有命令假设当前目录是仓库根目录 `IsaacLab-NeRD`。

## Setup

```bash
./isaaclab.sh -p -m pip install -e source/isaaclab_neural --no-deps
```

`eval.py` 会启动注册的 NeRD task，然后执行若干步环境 step：

- 不传 `--policy-checkpoint` 时使用 zero action。
- 传 `--policy-checkpoint` 时加载 RSL-RL policy 做闭环 rollout。
- 传 `--checkpoint` 时从 checkpoint 恢复训练时保存的 `env.neural_solver_cfg`，并把 `neural_model_path` 指向该 checkpoint。

它主要用于确认 checkpoint、task 注册、环境 reset/step、policy 推理链路能跑通；训练过程中的 dataset rollout MSE 由 trainer 内部 evaluator 负责。

## Common Arguments

```text
--task                 Gym task id
--checkpoint           NeRD checkpoint
--policy-checkpoint    Optional RSL-RL policy checkpoint
--policy-agent         RSL-RL agent cfg entry point
--num-envs             Number of parallel envs
--num-steps            Number of env steps
--seed                 Environment seed
--contact-mode         fixed_ground or newton_native
--num-contacts-per-env Native fixed contact slots
--video                Record rollout video
--video-dir            Video output directory
--headless             Headless launcher mode
--device               Launcher device, e.g. cuda:0
```

For Newton/MJWarp Anymal tasks, keep `presets=newton_mjwarp`.

## Checkpoint Paths

Training with timestamped logdirs usually writes checkpoints under:

```text
./data/trained_models/<experiment>/<timestamp>/nn/final_model.pt
```

If `--no-time-stamp` was used, the path is usually:

```text
./data/trained_models/<experiment>/nn/final_model.pt
```

OSMO runs upload checkpoints and logs to NV-Datasets under:

```text
IsaacLab-NeRD-Output/<workflow_name>/<env_name>/<timestamp>/nn/final_model.pt
```

Download a run locally before eval:

```bash
./isaaclab.sh -p osmo_scripts/pull_nvdataset.py \
  IsaacLab-NeRD-Output \
  --prefix <workflow_name> \
  --output-dir ./data/osmo_outputs \
  --clean \
  --list
```

Then point `--checkpoint` at the downloaded `final_model.pt`.

## Cartpole Fixed Ground

### Zero-Action Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Cartpole-NeRD-v0 \
  --checkpoint ./data/trained_models/Cartpole/nn/final_model.pt \
  --num-envs 16 \
  --num-steps 32 \
  --seed 0 \
  --contact-mode fixed_ground \
  --headless
```

If the checkpoint is under a timestamped run, replace the checkpoint path, for example:

```text
./data/trained_models/Cartpole/07-08-2026-12-00-00/nn/final_model.pt
```

## Anymal-C Fixed Ground

### Zero-Action Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C/nn/final_model.pt \
  --num-envs 16 \
  --num-steps 32 \
  --seed 0 \
  --contact-mode fixed_ground \
  --headless \
  presets=newton_mjwarp
```

### Policy Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C/nn/final_model.pt \
  --policy-checkpoint /path/to/rsl_rl/Anymal-C-Velocity-Flat/model.pt \
  --policy-agent rsl_rl_cfg_entry_point \
  --num-envs 16 \
  --num-steps 200 \
  --seed 0 \
  --contact-mode fixed_ground \
  --headless \
  presets=newton_mjwarp
```

If `params/agent.yaml` exists next to the policy checkpoint, `eval.py` uses it to match the original RSL-RL runner config.

## Anymal-C Newton Native

Native eval should use a native-trained checkpoint. The checkpoint stores:

```yaml
contact_mode: newton_native
num_contacts_per_env: 64
contact_packing_policy: penetration_priority
```

Passing `--contact-mode newton_native --num-contacts-per-env 64` makes the CLI override explicit. The packing policy is restored from the checkpoint config.

### Zero-Action Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C-Native/nn/final_model.pt \
  --num-envs 16 \
  --num-steps 32 \
  --seed 0 \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --headless \
  presets=newton_mjwarp
```

### Policy Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C-Native/nn/final_model.pt \
  --policy-checkpoint /path/to/rsl_rl/Anymal-C-Velocity-Flat/model.pt \
  --policy-agent rsl_rl_cfg_entry_point \
  --num-envs 16 \
  --num-steps 200 \
  --seed 0 \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --headless \
  presets=newton_mjwarp
```

## Anymal-C Rough Newton Native

Rough eval should use the rough NeRD task, a rough-native checkpoint, and a
matching rough RSL-RL policy checkpoint. Do not evaluate the rough checkpoint
through the flat NeRD task.

### Zero-Action Smoke Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C-Rough-Native/nn/final_model.pt \
  --num-envs 16 \
  --num-steps 32 \
  --seed 0 \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --headless \
  presets=newton_mjwarp
```

### Rough Policy Eval

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C-Rough-Native/nn/final_model.pt \
  --policy-checkpoint /path/to/rsl_rl/Anymal-C-Velocity-Rough/model.pt \
  --policy-agent rsl_rl_cfg_entry_point \
  --num-envs 16 \
  --num-steps 400 \
  --seed 0 \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --headless \
  presets=newton_mjwarp
```

For the negative/control baseline, run the flat-native checkpoint on the same
rough task and compare the rollout video/reward profile against the rough-native
checkpoint.

## Video Recording

Add these flags to any eval command:

```bash
--video \
--video-length 400 \
--video-dir ./videos/eval
```

## Expected Output

Zero-action mode prints lines like:

```text
[eval] task=..., num_envs=..., steps=..., action_shape=..., action_source=zero
[step 0] obs_keys=..., reward_mean=..., terminated=..., truncated=..., info_keys=...
```

Policy mode prints lines like:

```text
[eval] task=..., num_envs=..., steps=..., action_source=rsl_rl
[step 0] reward_mean=..., dones=..., info_keys=...
```

Use these outputs to confirm reset/step succeeds, rewards are finite, and no unexpected termination or contact-mode mismatch occurs.

## Troubleshooting

### `No module named 'omni.physics'`

Anymal-C eval creates the inherited PhysX contact sensor scene entities. If the
current local Python/Isaac Sim runtime cannot import `omni.physics.tensors`, eval
can fail while constructing the environment with an error similar to:

```text
Could not resolve ... isaaclab_physx.sensors.contact_sensor.contact_sensor:ContactSensor
Received the error:
 No module named 'omni.physics'
```

This is a runtime installation issue, not a NeRD checkpoint issue. Use a full
Isaac Sim/PhysX environment, the same Isaac Lab container used by OSMO, or run on
a machine where this import succeeds:

```bash
./isaaclab.sh -p -c "import omni.physics.tensors as physx; print('ok')"
```

### Zero-action vs policy eval

Without `--policy-checkpoint`, `eval.py` uses zero actions. This only smoke-tests
checkpoint loading and environment stepping. To evaluate closed-loop locomotion,
pass the matching RSL-RL policy checkpoint with `--policy-checkpoint`.
