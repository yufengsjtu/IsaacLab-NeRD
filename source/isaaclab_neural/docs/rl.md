# Reinforcement Learning

The registered NeRD environments can be used with Isaac Lab's RSL-RL workflow
after a compatible neural dynamics checkpoint is available.

Supported public environments:

- `Isaac-Velocity-Flat-Anymal-C-NeRD-v0`
- `Isaac-Velocity-Rough-Anymal-C-NeRD-v0`
- `Isaac-Cartpole-NeRD-v0`

No dynamics checkpoint is bundled or configured by default. Supply a trained
NeRD checkpoint before launching RL.

## Train a policy

Set `NERD_CHECKPOINT` to a trained neural dynamics checkpoint, then use the
NeRD-specific RSL-RL entrypoint:

```bash
test -f "$NERD_CHECKPOINT"
./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path "$NERD_CHECKPOINT" \
  --num_envs 4096 \
  --headless \
  presets=newton_mjwarp
```

## Evaluate a policy

The NeRD evaluator accepts both dynamics and policy checkpoints. Set
`NERD_CHECKPOINT` to a dynamics checkpoint produced by training:

```bash
test -f "$NERD_CHECKPOINT"
./isaaclab.sh -p -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint "$NERD_CHECKPOINT" \
  --policy-checkpoint pretrained/control_policy/anymal_c_flat_terrain/model_299.pt \
  --num-envs 16 \
  --num-steps 1000
```

Keep the policy's adjacent `params/agent.yaml` when moving a checkpoint. The
evaluator uses it to reproduce inference settings such as action clipping.

Policy checkpoints under `pretrained/control_policy/anymal_c_flat_terrain/`
and `pretrained/control_policy/anymal_c_rough_terrain/` are managed with Git
LFS. A local G1 policy is not part of this release.
