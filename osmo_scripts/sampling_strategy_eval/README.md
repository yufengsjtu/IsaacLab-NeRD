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

The jobs embed a Base64-encoded minimal immutable code archive in each OSMO
workflow. Production workflows mount the exact fixed-Epoch-199 files as a
read-only OSMO DATA input; a W&B download path remains available for runs that
publish periodic files. Both paths enforce size, SHA-256, checkpoint, model,
and training-contract validation. Final Markdown and JSON analyses are printed
into persistent OSMO logs; each task waits for an explicit result ACK before it
exits.

Encoder A and D are executable now. Encoder C is intentionally absent: the old
C run has no recoverable fixed Epoch-199 checkpoint, so substituting a
metric-selected best checkpoint would invalidate the causal contract.
