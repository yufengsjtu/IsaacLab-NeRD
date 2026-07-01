# IsaacLab-NeRD Eval Runner README

本文档说明 `source/isaaclab_neural/isaaclab_neural/eval/eval.py` 的用途和常用命令。

`eval.py` 是一个通用 NeRD 在线评估/冒烟测试入口。它会启动注册的 Isaac Lab NeRD task，然后执行若干步环境 step：

- 不传 `--policy-checkpoint` 时，使用 zero action。
- 传 `--policy-checkpoint` 时，加载 RSL-RL policy 做闭环 rollout。
- 传 `--checkpoint` 时，用 NeRD 模型 checkpoint 覆盖注册环境默认的 neural solver 配置。

它和训练过程中的 `training_evaluator.py` 不同：`eval.py` 不计算 HDF5 dataset 上的 rollout MSE，也不写 TensorBoard；它主要用于确认 checkpoint、task 注册、环境 reset/step、policy 推理链路能正常跑通。

## 配置优先级

`eval.py` 中 NeRD solver 配置优先级如下：

1. 注册 task 自带默认 `NerdSolverCfg`。
2. `--checkpoint` 会读取训练 checkpoint 里保存的 `env.neural_solver_cfg`，并把 `neural_model_path` 指向该 checkpoint。
3. CLI solver 参数继续覆盖当前 solver cfg，例如 `--contact-mode fixed_ground`。

常用 solver 覆盖参数：

```bash
--contact-mode fixed_ground
--num-contacts-per-env 0
```

## 常用参数

```text
--task                 注册的 gym task id
--checkpoint           可选 NeRD 模型 checkpoint
--policy-checkpoint    可选 RSL-RL policy checkpoint
--policy-agent         RSL-RL agent cfg entry point，默认 rsl_rl_cfg_entry_point
--num-envs             并行环境数，默认 1
--num-steps            执行 step 数，默认 8
--seed                 环境随机种子，默认 0
--contact-mode         fixed_ground 或 newton_native
--num-contacts-per-env 覆盖每个环境的 contact 数
--headless             来自 Isaac Lab launcher，集群/OSMO 上通常需要
--visualize            选择可视化的前端
--device               来自 Isaac Lab launcher，例如 cuda:0
```

## Cartpole Fixed Ground Eval

### Zero-action smoke eval

不传 `--checkpoint` 时，会使用 `Isaac-Cartpole-NeRD-v0` 注册环境里的默认 solver/model 配置。

```bash
python3 -m isaaclab_neural.eval.eval \
    --task Isaac-Cartpole-NeRD-v0 \
    --num-envs 16 \
    --num-steps 32 \
    --seed 0 \
    --contact-mode fixed_ground \
    --headless
```

### 使用训练出的 NeRD checkpoint

```bash
python3 -m isaaclab_neural.eval.eval \
    --task Isaac-Cartpole-NeRD-v0 \
    --checkpoint ./data/trained_models/Cartpole/nn/final_model.pt \
    --num-envs 16 \
    --num-steps 32 \
    --seed 0 \
    --contact-mode fixed_ground \
    --headless
```

如果训练输出带 timestamp，checkpoint 路径通常类似：

```text
./data/trained_models/Cartpole/<timestamp>/nn/final_model.pt
```

## Anymal-C Fixed Ground Eval

### Zero-action smoke eval

```bash
python3 -m isaaclab_neural.eval.eval \
    --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
    --num-envs 16 \
    --num-steps 32 \
    --seed 0 \
    --contact-mode fixed_ground \
    --headless \
    presets=newton_mjwarp
```

### 使用训练出的 NeRD checkpoint

```bash
python3 -m isaaclab_neural.eval.eval \
    --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
    --checkpoint ./data/trained_models/Anymal-C/nn/final_model.pt \
    --num-envs 16 \
    --num-steps 32 \
    --seed 0 \
    --contact-mode fixed_ground \
    --headless \
    presets=newton_mjwarp
```

如果训练输出带 timestamp，checkpoint 路径通常类似：

```text
./data/trained_models/Anymal-C/<timestamp>/nn/final_model.pt
```

## Policy Eval

传入 `--policy-checkpoint` 后，`eval.py` 会构建 RSL-RL wrapper，并使用 checkpoint 对应的 policy 输出 action。

如果 policy checkpoint 旁边存在：

```text
params/agent.yaml
```

脚本会优先读取这个保存下来的 runner 配置，以匹配训练时的 `clip_actions`、runner class、device 等设置；否则使用注册 task 的 `--policy-agent` entry point。

### Anymal-C policy eval 示例

```bash
python3 -m isaaclab_neural.eval.eval \
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

## 输出内容

zero-action 模式会打印类似：

```text
[eval] task=..., num_envs=..., steps=..., action_shape=..., action_source=zero
[step 0] obs_keys=..., reward_mean=..., terminated=..., truncated=..., info_keys=...
```

policy 模式会打印类似：

```text
[eval] task=..., num_envs=..., steps=..., action_source=rsl_rl
[step 0] reward_mean=..., dones=..., info_keys=...
```

这些输出用于快速判断环境是否能 reset/step、reward 是否为有限值、是否出现异常终止。

## OSMO 使用建议

在 OSMO 中运行 eval 时，建议：

- 使用与训练一致的 Docker image。
- 先运行较小规模 smoke eval，例如 `--num-envs 16 --num-steps 32`。
- 明确传入 `--contact-mode fixed_ground`，保持 fixed ground 版本一致。
- 将 `--checkpoint` 指向训练输出目录中的 `nn/final_model.pt` 或验证最优模型。

示例：

```bash
cd $HOME/code/IsaacLab-NeRD
python3 -m isaaclab_neural.eval.eval \
    --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
    --checkpoint "$OUTPUT_LOCAL_PATH/Anymal-C/nn/final_model.pt" \
    --num-envs 16 \
    --num-steps 32 \
    --contact-mode fixed_ground \
    --headless \
    presets=newton_mjwarp
```
