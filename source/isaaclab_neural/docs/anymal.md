# Anymal-C Guide

Anymal-C is the contact-aware NeRD path. The release includes flat and rough
terrain tasks, flat contact tensors, contact tokens, shared-per-body encoding,
contact-force targets, and RoPE/KV-cache models.

## Generate flat-terrain data

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_anymal_c.sh \
  --dataset-name dataset_train.hdf5 \
  --num-transitions 1000000
```

For rough terrain, override both the registered ground-truth task and dataset
environment name:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_anymal_c.sh \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --env-name Anymal-C-Rough-Native \
  --dataset-name dataset_train.hdf5 \
  --num-transitions 1000000
```

Generate validation splits with distinct names and seeds matching the selected
training YAML.

## Train

Flat native-contact model:

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native.yaml
```

Rough native-contact model:

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native.yaml
```

Every Anymal-C configuration uses `contact_mode: newton_native` and a positive
contact capacity. Do not reuse Cartpole's zero-contact setting.

## Evaluate

Set `NERD_CHECKPOINT` to a checkpoint produced by training, then run:

```bash
test -f "$NERD_CHECKPOINT"
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --checkpoint "$NERD_CHECKPOINT" \
  --num-envs 16 \
  --num-steps 100
```

See [Advanced Features](advanced.md) before selecting a token, force, or RoPE
training configuration. Pre-trained dynamics checkpoints are not distributed
with this release.

## Control-step recipe (SO101/Allegro pipeline)

`anymal_flat_control_step` trains flat ANYmal-C with the optimized control-step pipeline: the env-driven collector
(`isaaclab_neural.generate.allegro_dataset --task Isaac-Velocity-Flat-Anymal-C-Policy-v0 --env-name
Anymal-C-Flat-ControlStep --record-control-steps`) records reset-safe clips with the packaged flat teacher on eight
GPUs; `train/cfg/Anymal/transformer_flat_control_step.yaml` predicts one 20 ms control step per call with joint-target
input, unique-transition sliding windows, GPU-resident data and FP16 autocast. Collection and
`Isaac-Velocity-Flat-Anymal-C-ControlStep-NeRD-v0` both use the stateless ANYdrive actuator: a control-step model
evaluates the actuator once per control step and cannot reproduce the per-frame LSTM state.

The OSMO teacher gate evaluates the policy natively with `isaaclab_neural.eval.policy_rollout` and passes its velocity
tracking error to training, where `best_eval_model` is the checkpoint closest to that native error. Run
`anymal_flat_control_step_smoke` first; both presets end with MJWarp/NeRD throughput benchmarks.
