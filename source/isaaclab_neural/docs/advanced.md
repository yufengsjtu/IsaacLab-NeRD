# Advanced Features

Use the baseline flat-contact configurations first. The following Anymal-C
features are opt-in and require matching generation, training, and inference
settings.

## Contact tokens

Set `contact_representation: contact_tokens` to represent directed native
contacts as a bounded set. Maintained token configs use
`body_round_robin_pair_atomic` packing and `max_contact_tokens: 64`.

Available training configs:

- `transformer_native_contact_tokens.yaml`
- `transformer_rough_native_contact_tokens.yaml`

## Shared-per-body encoding

The shared-per-body encoder groups token features by primary body while sharing
encoder weights. Maintained release configs include:

- `transformer_native_shared_per_body.yaml`
- `transformer_rough_native_shared_per_body_formal.yaml`

The previous small/medium/data-budget ablation matrix is intentionally not part
of the public preset surface.

## Contact-force prediction

The auxiliary body force head predicts resultant world-frame linear contact
forces for each Anymal-C primary body. Generate matching labels with
`--record-body-net-contact-forces`; force configs reject datasets without the
required schema.

Use:

- `transformer_rough_native_shared_per_body_contact_force_formal.yaml`
- `transformer_rough_native_shared_per_body_contact_force_rope.yaml`

## RoPE and KV cache

RoPE configs support incremental transformer inference with a KV cache. Use
`transformer_rough_native_shared_per_body_rope_formal.yaml` for the maintained
RoPE-only path. Checkpoints must be evaluated with their embedded model and
solver configuration; do not mix token ordering or history settings between
training and deployment. Pre-trained dynamics checkpoints are not distributed
with this release.

The NeRD RSL-RL train and play entrypoints accept `--use_kv_cache` for a
compatible RoPE checkpoint.

## Local smoke training

The retained smoke config is local-only and expects matching rough
contact-token datasets:

```bash
./isaaclab.sh -p -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --cfg source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens_smoke.yaml
```

It is intentionally not exposed as an OSMO preset.

## Validation tools

- `isaaclab_neural.eval.reset_smoke` checks reset synchronization.
- `isaaclab_neural.eval.contact_reconstruction_diagnostic` validates contact
  reconstruction.
- Dataset contact regime and distribution tools remain available under
  `isaaclab_neural.eval`.

Internal performance benchmark scripts were removed from the public release.
