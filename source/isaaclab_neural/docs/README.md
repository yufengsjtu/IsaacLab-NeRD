# IsaacLab Neural

`isaaclab_neural` provides NeRD-backed neural simulation interfaces for IsaacLab.
It includes registered NeRD tasks, Newton solver integration, dataset generation,
training, evaluation, and RSL-RL policy learning utilities.

For workflow details, see:

- [Training](train.md) (NeRD dynamics datasets / OSMO)
- [Smoke train](smoke_train.md) (local 1M-scale rough native contact tokens)
- [Per-body net contact forces](body_net_contact_force.md) (Stage 1 labels, training, and smoke commands)
- [Evaluation](eval.md) (dynamics smoke / policy-closed-loop eval)
- [RL policy learning](rl.md) (flat Anymal-C locomotion on NeRD)
