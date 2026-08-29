# Cartpole Tutorial

Cartpole is the minimal NeRD example. It has no collision inputs, so both data
generation and inference use `num_contacts_per_env=0` and skip the Newton
collision pipeline.

## Generate data

Generate the training split:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_cartpole.sh \
  --dataset-name dataset_train.hdf5 \
  --num-transitions 100000
```

Generate validation data with the same command and a different name and seed:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_cartpole.sh \
  --dataset-name dataset_valid.hdf5 \
  --num-transitions 100000 \
  --seed 1
```

Generate the passive validation split used by the config:

```bash
./source/isaaclab_neural/isaaclab_neural/generate/scripts/generate_cartpole.sh \
  --dataset-name dataset_passive_valid.hdf5 \
  --num-transitions 100000 \
  --zero-actions \
  --seed 2
```

The maintained training config expects datasets under
`data/datasets/Cartpole/`. Adjust its dataset paths if you use another
directory.

The local tutorial uses 100,000 transitions per split for a quick first run.
The maintained OSMO preset uses one million training transitions for a
full-scale remote run, so its training cache is intentionally larger.

## Train

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Cartpole-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml
```

## Evaluate

Pass the checkpoint produced by training:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Cartpole-NeRD-v0 \
  --checkpoint /path/to/checkpoint.pt \
  --num-envs 16 \
  --num-steps 100
```

For remote execution, use the `cartpole_newton_native` OSMO preset described in
the [OSMO reference](osmo.md).
