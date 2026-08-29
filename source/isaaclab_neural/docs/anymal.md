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
