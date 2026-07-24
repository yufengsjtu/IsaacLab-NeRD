# IsaacLab-NeRD Local Training

本文档记录本地数据采集和训练流程，覆盖：

- Cartpole fixed-ground
- Anymal-C fixed-ground
- Anymal-C Newton native contacts

所有命令假设当前目录是仓库根目录 `IsaacLab-NeRD`。

## Setup

```bash
./isaaclab.sh -p -m pip install -e source/isaaclab_neural --no-deps
```

数据默认写入 `./data/datasets`，训练输出默认写入 `./data/trained_models`。

本地调试时可以先把 `--num-transitions`、`--num-envs`、`--trajectory-length` 调小；正式复现实验时使用下文的完整规模。

## Recommended Preset Runner

当前推荐优先使用 `osmo_scripts/run_experiment.py` 和
`osmo_scripts/presets/*.yaml` 来保持本地训练与 OSMO 训练配置一致。preset
负责选择任务、数据集规格、contact mode、训练 config 和运行规模；训练超参数仍然只来自
`source/isaaclab_neural/isaaclab_neural/train/cfg/` 下的 YAML。

本地运行 Cartpole preset：

```bash
./isaaclab.sh -p osmo_scripts/run_experiment.py \
  --preset cartpole_fixed_ground \
  --workflow-base-name cartpole-fixed-ground \
  --dataset-subdir cartpole-fixed-ground \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/osmo-local-cartpole \
  --dataset-dir ./data/datasets
```

本地运行 Anymal-C fixed-ground preset：

```bash
./isaaclab.sh -p osmo_scripts/run_experiment.py \
  --preset anymal_fixed_ground \
  --workflow-base-name anymal-c-fixed-ground \
  --dataset-subdir anymal-c-fixed-ground \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/osmo-local-anymal-c \
  --dataset-dir ./data/datasets
```

本地运行 Anymal-C Newton-native preset：

```bash
./isaaclab.sh -p osmo_scripts/run_experiment.py \
  --preset anymal_newton_native \
  --workflow-base-name anymal-c-newton-native \
  --dataset-subdir anymal-c-newton-native \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/osmo-local-anymal-c-native \
  --dataset-dir ./data/datasets
```

如果只想复用已存在的本地 HDF5 并跳过数据生成，可以直接运行下文的
`isaaclab_neural.train.train` 命令。

## OSMO Training

OSMO 训练入口在 `osmo_scripts/start.sh`，配置在
`osmo_scripts/presets/*.yaml`。默认会：

1. 打包当前代码并上传到 `IsaacLab-NeRD-Code`；
2. 从 `IsaacLab-NeRD-Datasets/<dataset_subdir>/<env_name>/` 查找缓存数据；
3. 缓存缺失时在 OSMO 内生成数据并上传回 `IsaacLab-NeRD-Datasets`；
4. 将日志、TensorBoard 和 checkpoints 上传到
   `IsaacLab-NeRD-Output/<workflow_name>/`。

示例：

```bash
export NGC_API_KEY=<your-nvapi-key>
export NVDATASET_TENANTID=<your-tenant-id>

./osmo_scripts/start.sh \
  --preset anymal_fixed_ground \
  --pool <osmo-pool>
```

更多 OSMO/NV-Datasets 使用方式见 `osmo_scripts/README.md`。
如果需要同步到 Weights & Biases，使用 `osmo_scripts/start.sh --enable-wandb`
并设置 `WANDB_API_KEY`；详细参数见 `osmo_scripts/README.md` 的 W&B 部分。

## Memory and Storage Notes

数据生成可以通过 `--write-chunk-transitions` 分块写入 HDF5。每个 chunk
会先校验完整 schema、shape 和 dtype；追加写入失败时会回滚已 resize 的 dataset，
避免留下部分写入的 HDF5 文件。

`SequenceModelTrainer` 的 ``eager`` 模式会在初始化时把
`algorithm.dataset.max_capacity` 对应的数据加载到 CPU 内存。DDP 下 trajectory 会先按
全局 ``max_capacity`` 截断，再由各 rank 加载互不重叠的 trajectory shard，因此所有
rank 合计只保留约一份训练集，而不是每个 rank 各复制一份。Validation datasets 仍只在
rank 0 完整加载。

``lazy`` 模式只保留 metadata 和 HDF5 handle，并按 batch 读取。CUDA 训练时默认启用
pinned memory、non-blocking copy 和 persistent workers。可使用以下配置调优：

```yaml
algorithm:
  dataset:
    load_mode: lazy
    num_data_workers: 4
    pin_memory: auto
    non_blocking: auto
    persistent_workers: auto
    prefetch_factor: 2
    # Optional global cap divided across DDP ranks.
    max_total_workers: 32
```

``eager`` 通常吞吐更高，``lazy`` 内存最低。OSMO 上可先使用 ``eager`` rank sharding；
若 validation 或 contact-token 数据仍超过节点内存，再切换 ``lazy``。降低
``max_capacity`` 或 ``num_data_workers`` 适合 smoke test。

## Cartpole Fixed Ground

配置文件：

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml
```

关键设置：

```yaml
contact_mode: fixed_ground
states_frame: world
```

### Generate Train Dataset

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Cartpole \
  --robot-name Cartpole \
  --sample-mode joint_f \
  --initial-states-source sample \
  --contact-mode fixed_ground \
  --states-frame world \
  --num-envs 256 \
  --num-transitions 1000000 \
  --trajectory-length 100 \
  --seed 0 \
  --headless \
  --force-overwrite
```

### Generate Validation Datasets

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Cartpole \
  --robot-name Cartpole \
  --sample-mode joint_f \
  --initial-states-source sample \
  --contact-mode fixed_ground \
  --states-frame world \
  --num-envs 256 \
  --num-transitions 100000 \
  --trajectory-length 100 \
  --seed 1 \
  --headless \
  --force-overwrite

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_passive_valid.hdf5 \
  --env-name Cartpole \
  --robot-name Cartpole \
  --sample-mode joint_f \
  --initial-states-source sample \
  --contact-mode fixed_ground \
  --states-frame world \
  --zero-actions \
  --num-envs 256 \
  --num-transitions 100000 \
  --trajectory-length 100 \
  --seed 2 \
  --headless \
  --force-overwrite
```

### Train

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Cartpole-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml \
  --logdir ./data/trained_models/Cartpole \
  --num-envs 256 \
  --seed 0 \
  --headless \
  --update-dataset-statistics \
  --skip-check-log-override
```

## Anymal-C Fixed Ground

配置文件：

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer.yaml
```

关键设置：

```yaml
contact_mode: fixed_ground
states_frame: body
anchor_frame_step: every
```

### Generate Train Dataset

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 20000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 0 \
  --headless \
  --force-overwrite
```

### Generate Validation Datasets

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 10 \
  --headless \
  --force-overwrite

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 20 \
  --headless \
  --force-overwrite

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 30 \
  --headless \
  --force-overwrite
```

### Train

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer.yaml \
  --logdir ./data/trained_models/Anymal-C \
  --num-envs 1024 \
  --seed 0 \
  --headless \
  --update-dataset-statistics \
  --skip-check-log-override \
  presets=newton_mjwarp
```

## Anymal-C Newton Native

配置文件：

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native.yaml
```

关键设置：

```yaml
contact_mode: newton_native
num_contacts_per_env: 64
contact_packing_policy: penetration_priority
states_frame: body
anchor_frame_step: every
```

Native 数据集记录 Newton collision pipeline 产生的动态 contacts，并打包成固定
64 个 slots。打包默认使用 `penetration_priority`，即优先保留 signed surface
separation 最小的接触，并用几何信息做确定性 tie-break。

Native contact 输入的约定是：

- contact side 会 canonicalize 为 primary robot articulation first；
- `contact_normals` 从 robot side 指向另一侧；
- `contact_depths` 存 signed surface separation，而不是旧的 raw normal distance；
- `contact_masks` 表示 slot 是否被 native contact 占用；
- inactive slots 会在 preprocessing 和模型输入边界再次清零，避免归一化或噪声把 padding
  变成非零特征；
- contact RMS 统计只使用 `contact_masks=True` 的有效 contacts；样本过少的 slot 会回退到
  pooled field statistics。

当前 native 输入包含 `contact_points_0` 和 `contact_points_1`；两者采集时保存为
world frame，训练 preprocessing 时转换到 body frame。

### Generate Train Dataset

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 20000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 0 \
  --headless \
  --force-overwrite
```

### Generate Validation Datasets

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 10 \
  --headless \
  --force-overwrite

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 20 \
  --headless \
  --force-overwrite

./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 30 \
  --headless \
  --force-overwrite
```

### Train

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native.yaml \
  --logdir ./data/trained_models/Anymal-C-Native \
  --num-envs 1024 \
  --seed 0 \
  --headless \
  --update-dataset-statistics \
  --skip-check-log-override \
  presets=newton_mjwarp
```

## TensorBoard

```bash
python3 -m tensorboard.main \
  --logdir ./data/trained_models \
  --host 0.0.0.0 \
  --port 6006
```

## Local Smoke Testing

For a quick local test, reduce data generation scale first:

```bash
--num-envs 32
--num-transitions 2000
--trajectory-length 20
--write-chunk-transitions 2000
```

For native experiments, always regenerate datasets after changing contact schema, contact packing policy, `num_contacts_per_env`, or contact point frame semantics.

## Contact Token Sets

PhysicsNeMo-style contact token sets store directed contacts as
``[trajectories, steps, max_contact_tokens, 17]`` tensors instead of flat
64-slot fields. Enable them for dataset generation, training, and deployment
with:

```yaml
env:
  neural_solver_cfg:
    contact_representation: contact_tokens
    max_contact_tokens: 64
    # Token mode always uses pair-atomic body round-robin inside ContactSetEncoder.
    # This field is metadata for docs/config clarity; flat packing ignores it.
    contact_packing_policy: body_round_robin_pair_atomic
inputs:
  low_dim: [states_embedding, joint_f, gravity_dir]
  contact_set:
    dim: 17
    encoder_layers: 2
    encoder_heads: 4
    num_latent_queries: 8
    hidden_size: 384
```

Notes:

- Token capacity is controlled by ``max_contact_tokens`` (not ``num_contacts_per_env``).
- Eager and lazy loaders preserve ``contact_tokens`` shape ``[..., K, 17]``.
- Token HDF5 must include ``root_body_q`` and ``gravity_dir`` for body-frame training.

Dataset generation:

```bash
./isaaclab.sh -p -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets/Anymal-C-Native-ContactTokens \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64
```

Training configs:

- Flat smoke: `source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native_contact_tokens.yaml`
- Rough A/B: `source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens.yaml`

Diagnostics:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.contact_distribution_stats --dataset PATH
./isaaclab.sh -p -m isaaclab_neural.eval.contact_regime_eval --dataset PATH --overflow-gate
```

Contact-token checkpoints are not compatible with flat contact checkpoints.
Regenerate HDF5 whenever `max_contact_tokens` or token channel semantics change.
