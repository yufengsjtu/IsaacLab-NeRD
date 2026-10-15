# IsaacLab Neural

`isaaclab_neural` integrates NeRD neural dynamics with Isaac Lab's Newton
backend. The public release supports local dataset generation, model training,
evaluation, RSL-RL, and optional OSMO execution.

Choose a path:

1. [Getting Started](getting_started.md) — install the extension and understand
   the common workflow.
2. [Cartpole Tutorial](cartpole.md) — run the smallest contact-free example.
3. [Anymal-C Guide](anymal.md) — work with flat or rough terrain and native
   contacts.
4. [Reinforcement Learning](rl.md) — train or evaluate a policy on NeRD.
5. [Advanced Features](advanced.md) — contact tokens, shared-per-body encoding,
   contact-force prediction, RoPE, and KV-cache inference.
6. [OSMO Reference](osmo.md) — submit one of the maintained remote presets.

The deprecated `fixed_ground` mode is intentionally rejected after emitting a
deprecation warning. New code must use `newton_native`; set
`num_contacts_per_env: 0` only for contact-free systems such as Cartpole.

G1 development lives in a separate worktree and is not included in this
release.

Archived best NeRD checkpoints for every task are listed in [checkpoints.md](checkpoints.md).
