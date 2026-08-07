# Rough Native Contact-Token Smoke Train

本地小规模流程：生成 **Newton native + contact tokens** 的 rough-terrain
数据集，并做短训 smoke。完整规模见 [train.md](train.md)。

假设当前目录是仓库根目录 `IsaacLab-NeRD`。

## Goal

| item | smoke scale |
|---|---:|
| train transitions | ``1000000`` |
| valid / zero-action transitions | ``100000`` each |
| ``num_envs`` (data gen) | ``256`` |
| ``trajectory_length`` | ``100`` |
| contact | ``newton_native`` + ``contact_tokens`` (K=64) |
| train epochs | ``5`` (smoke cfg) |

输出目录：

```text
./data/datasets/Anymal-C-Rough-Native-ContactTokens/
./data/trained_models/Anymal-C-Rough-Native-ContactTokens-Smoke/
```

## Setup

```bash
./isaaclab.sh -p -m pip install -e source/isaaclab_neural --no-deps
```

Rough + tokens 关键 flags（所有 generate 命令共用）：

```text
--task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0
--contact-mode newton_native
--contact-representation contact_tokens
--max-contact-tokens 64
--num-contacts-per-env 64
--env-name Anymal-C-Rough-Native-ContactTokens
```

HDF5 实际路径为 ``{dataset_dir}/{env_name}/{dataset_name}``。

## 1. Generate train dataset (1M)

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 256 \
  --num-transitions 1000000 \
  --write-chunk-transitions 1000000 \
  --trajectory-length 100 \
  --seed 0 \
  --headless \
  --force-overwrite \
  presets=newton_mjwarp
```

## 2. Generate validation datasets (100k each)

### Randomized-PD valid

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 10 \
  --headless \
  --force-overwrite \
  presets=newton_mjwarp
```

### Zero-action valid (same Dataset-Gen task)

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --zero-actions \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 20 \
  --headless \
  --force-overwrite \
  presets=newton_mjwarp
```

Smoke 训练 cfg 只依赖上面三个 HDF5。若要贴近完整 rough cfg（含 LSTM
actuator valids），再跑可选步骤 3。

## 3. Optional LSTM actuator valids

使用 stock rough task（LSTM actuator）+ 同一 contact-token 设置：

```bash
POLICY=./pretrained/control_policy/anymal_c_rough_terrain/model_1499.pt

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --zero-actions \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 30 \
  --headless \
  --force-overwrite \
  presets=newton_mjwarp

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_policy_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode policy \
  --policy-checkpoint "$POLICY" \
  --policy-agent rsl_rl_cfg_entry_point \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 40 \
  --headless \
  --force-overwrite \
  presets=newton_mjwarp
```

完整训练 cfg（非 smoke）需要这些 LSTM files：

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens.yaml
```

## 4. Smoke train

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens_smoke.yaml \
  --logdir ./data/trained_models/Anymal-C-Rough-Native-ContactTokens-Smoke \
  --num-envs 64 \
  --seed 0 \
  --headless \
  --update-dataset-statistics \
  --skip-check-log-override \
  --save-interval 1 \
  --eval-interval 1 \
  presets=newton_mjwarp
```

Smoke cfg 要点：

- ``contact_representation: contact_tokens`` / ``max_contact_tokens: 64``
- ``num_epochs: 5``, ``num_iters_per_epoch: 200``, ``batch_size: 128``
- ``max_capacity: 1000000``
- valid 仅 ``exp_trajectory`` + ``zero_action_trajectory``

## 5. Quick eval (optional)

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C-Rough-Native-ContactTokens-Smoke/<timestamp>/nn/final_model.pt \
  --num-envs 16 \
  --num-steps 32 \
  --seed 0 \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --headless \
  presets=newton_mjwarp
```

``--checkpoint`` restores the embedded ``neural_solver_cfg`` (including
``contact_representation: contact_tokens``).

Token 数据诊断：

```bash
DS=./data/datasets/Anymal-C-Rough-Native-ContactTokens/dataset_train.hdf5
./isaaclab.sh -p -m isaaclab_neural.eval.contact_distribution_stats --dataset "$DS"
./isaaclab.sh -p -m isaaclab_neural.eval.contact_regime_eval --dataset "$DS" --overflow-gate
```

## Even smaller debug

若只想先打通链路，可再降到：

```text
--num-envs 32
--num-transitions 2000
--trajectory-length 20
--write-chunk-transitions 2000
```

并把 smoke cfg 里 ``algorithm.num_epochs`` / ``num_iters_per_epoch`` 再减小，或：

```bash
--cfg-overrides algorithm.num_epochs 1 algorithm.num_iters_per_epoch 20
```

## Notes

- Flat slots 与 contact tokens **不兼容**；不要混用 ``Anymal-C-Rough-Native`` flat
  数据与本 smoke 的 token 目录。
- Rough 需要 terrain context；smoke cfg 打开了 ``require_terrain_context``.
- 正式复现用 OSMO preset
  ``osmo_scripts/presets/anymal_rough_newton_native_contact_tokens.yaml``
  （20M train / 1M valid）和完整
  ``transformer_rough_native_contact_tokens.yaml``。
