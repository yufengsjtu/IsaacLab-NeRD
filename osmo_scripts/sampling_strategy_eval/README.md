# Fixed-Epoch-199 sampling-strategy evaluation

This directory is a standalone, read-only evaluator. It imports the existing
NeRD model, dataset, and rollout APIs but does not modify the trainer, model, or
their normal evaluation paths.

For one encoder, the evaluator compares old-sampling and new-sampling
three-seed checkpoint ensembles at exactly `model_epoch199.pt`. Both checkpoint
groups are evaluated on each frozen old and new validation distribution across
four regimes. The policy regime additionally runs 1,024 deterministic 10-step
rollouts. The primary comparison is always old/new checkpoints on the same
suite; own-distribution scores alone are descriptive.

The jobs receive a minimal immutable code archive as their first OSMO input.
They download the exact fixed-Epoch-199 files from the W&B run IDs frozen in
the checked-in checkpoint manifests, then enforce size, SHA-256, checkpoint,
model, and training-contract validation. Final result trees are written to a
campaign-specific OSMO DATA output prefix before the tasks exit.

Encoder A and D are executable now. Encoder C is intentionally absent: the old
C run has no recoverable fixed Epoch-199 checkpoint, so substituting a
metric-selected best checkpoint would invalidate the causal contract.
