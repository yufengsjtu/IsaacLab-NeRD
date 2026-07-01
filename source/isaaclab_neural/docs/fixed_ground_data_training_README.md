# IsaacLab-NeRD Fixed Ground Data Collection and Training

本文档记录 **fixed ground** 版本的 IsaacLab-NeRD 数据采集到模型训练流程，覆盖 Cartpole 和 Anymal-C。

## 前置假设

- 当前工作目录为仓库根目录 `IsaacLab-NeRD`。
- Python 环境已经包含标准 Isaac Lab、Isaac Sim、Newton/MuJoCo-Warp 等运行依赖。
- `isaaclab_neural` 已安装，例如：

```bash
python3 -m pip install -e source/isaaclab_neural --no-deps
```

所有数据采集命令都显式使用：

```bash
--contact-mode fixed_ground
```

生成的数据默认写入：

```bash
./data/datasets
```

训练输出默认写入：

```bash
./data/trained_models
```

## Cartpole Fixed Ground

### 1. 采集训练集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --dataset-dir ./data/datasets \
    --dataset-name dataset_train.hdf5 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    --num-envs 512 \
    --num-transitions 100000 \
    --trajectory-length 100 \
    --seed 0 \
    --headless \
    --force-overwrite
```

### 2. 采集验证集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --dataset-dir ./data/datasets \
    --dataset-name dataset_valid.hdf5 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    --num-envs 512 \
    --num-transitions 20000 \
    --trajectory-length 100 \
    --seed 1 \
    --headless \
    --force-overwrite
```

### 3. 采集 passive 验证集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --dataset-dir ./data/datasets \
    --dataset-name dataset_passive_valid.hdf5 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    --zero-actions \
    --num-envs 512 \
    --num-transitions 20000 \
    --trajectory-length 100 \
    --seed 2 \
    --headless \
    --force-overwrite
```

### 4. 训练 Cartpole 模型

```bash
python3 source/isaaclab_neural/isaaclab_neural/train/train.py \
    --task Isaac-Cartpole-NeRD-v0 \
    --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml \
    --logdir ./data/trained_models/Cartpole \
    --num-envs 512 \
    --seed 0 \
    --headless \
    --skip-check-log-override
```

训练配置读取的数据路径为：

```text
./data/datasets/Cartpole/dataset_train.hdf5
./data/datasets/Cartpole/dataset_valid.hdf5
./data/datasets/Cartpole/dataset_passive_valid.hdf5
```

## Anymal-C Fixed Ground

Anymal-C 数据采集包含随机 PD 增益训练/验证数据，以及 deployment-time LSTM actuator 分布下的验证数据。完整 OSMO 脚本位于：

```text
osmo_scripts/main_scripts.sh
```

默认规模较大：

```bash
TRAIN_TRANSITIONS=20000000
VALID_TRANSITIONS=1000000
NUM_ENVS=1024
TRAJECTORY_LENGTH=400
WRITE_CHUNK_TRANSITIONS=10000000
```

本地 smoke test 时建议先覆盖成较小值：

```bash
TRAIN_TRANSITIONS=100000 \
VALID_TRANSITIONS=20000 \
NUM_ENVS=512 \
TRAJECTORY_LENGTH=100 \
WRITE_CHUNK_TRANSITIONS=100000 \
bash osmo_scripts/main_scripts.sh
```

### 1. 采集随机 PD 增益训练集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
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

### 2. 采集随机 PD 增益验证集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
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
```

### 3. 采集 zero-action 验证集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
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
```

### 4. 采集 LSTM actuator zero-action 验证集

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
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

### 5. 可选：采集 policy validation 数据

如果存在 policy checkpoint，可以生成：

```text
./data/datasets/Anymal-C/dataset_lstm_actuator_policy_valid.hdf5
```

命令示例：

```bash
python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --dataset-dir ./data/datasets \
    --dataset-name dataset_lstm_actuator_policy_valid.hdf5 \
    --env-name Anymal-C \
    --robot-name Anymal-C \
    --sample-mode policy \
    --policy-checkpoint /path/to/Anymal-C-Velocity-Flat/model.pt \
    --initial-states-source env \
    --contact-mode fixed_ground \
    --num-envs 1024 \
    --num-transitions 1000000 \
    --write-chunk-transitions 10000000 \
    --trajectory-length 400 \
    --seed 40 \
    --headless \
    --force-overwrite
```

### 6. 训练 Anymal-C 模型

```bash
python3 -m isaaclab_neural.train.train \
    --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
    --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer.yaml \
    --logdir ./data/trained_models/Anymal-C \
    --num-envs 1024 \
    --seed 0 \
    --headless \
    --skip-check-log-override \
    presets=newton_mjwarp
```

训练配置读取的数据路径为：

```text
./data/datasets/Anymal-C/dataset_train.hdf5
./data/datasets/Anymal-C/dataset_valid.hdf5
./data/datasets/Anymal-C/dataset_zero_action_valid.hdf5
./data/datasets/Anymal-C/dataset_lstm_actuator_zero_action_valid.hdf5
```

## OSMO Workflow

OSMO fixed ground workflow 入口为：

```text
osmo_scripts/osmo_workflow.yaml
```

默认执行脚本为：

```text
osmo_scripts/init.sh
osmo_scripts/main_scripts.sh
```

当前 `main_scripts.sh` 是 Anymal-C fixed ground 全流程：先生成数据集，再训练模型。使用 GitHub 拉代码时需要在 workflow 参数中设置：

```yaml
code_git_url: "https://github.com/<org>/<repo>.git"
code_git_ref: "<branch-or-commit>"
```

如果只想做小规模 smoke test，可以通过环境变量覆盖数据规模：

```bash
TRAIN_TRANSITIONS=100000
VALID_TRANSITIONS=20000
NUM_ENVS=512
TRAJECTORY_LENGTH=100
WRITE_CHUNK_TRANSITIONS=100000
```

训练输出目录由 `OUTPUT_LOCAL_PATH` 控制，OSMO 模板默认值为：

```text
/tmp/runs/output
```
