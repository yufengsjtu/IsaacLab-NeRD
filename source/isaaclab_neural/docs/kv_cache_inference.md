# KV-cache inference

NeRD Transformer models can opt into rotary position embeddings (RoPE) and use
incremental KV-cache inference. The existing `ModelMixedInput.forward()` API and
the default learned absolute position embeddings remain unchanged.

## Legacy-checkpoint feature cache

Existing learned-absolute checkpoints can use an exact feature-cache
optimization without retraining. Each sequential step normalizes and encodes
only the latest input frame, retains the preceding encoded frames, and then
runs the unchanged Transformer over the complete ordered feature window.

Enable it on a Transformer solver with:

```yaml
env:
  neural_solver_cfg:
    use_feature_cache: true
```

The cache is rebuilt from raw history after a global reset. For partial
environment resets, only affected batch rows are re-encoded. The model output
therefore remains equivalent to `forward(..., single_step=True)` within
floating-point tolerance.

The Anymal rough environment benchmark also exposes `--use-feature-cache`:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_nerd_contact_modes \
  --contact-mode newton_native \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --neural-model-path /path/to/checkpoint.pt \
  --num_envs 512 --warmup 30 --steps 100 --headless --device cuda:0 \
  --use-feature-cache
```

With the evaluated 10-frame shared-per-body Anymal checkpoint and the optimized
native contact adapter, section profiling showed that the cache reduced model
forward time from 27.370 ms to 24.905 ms per step at 512 environments (1.10x)
and from 53.148 ms to 47.248 ms at 1024 environments (1.12x). Profiled
end-to-end time changed from 65.315 ms to 62.073 ms (1.05x) and from
100.895 ms to 93.752 ms (1.08x), respectively. At one environment, the
difference was within run-to-run noise.

## Native contact adapter optimization

The contact-token adapter uses a vectorized segmented scan instead of launching
one Python iteration per directed contact. It also reuses output/scratch
buffers, writes directly into adapter-owned storage, caches Warp-to-Torch
views, and uses a fused Warp kernel for world-space contact geometry. Stable
sorting, pair-atomic capacity handling, overflow counts, and body-slot identity
remain unchanged.

Run the simulator-independent scaling benchmark with:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_contact_packing \
  --device cuda:0 --num_envs 1 64 512 1024 \
  --contacts_per_env 8 32 --capacity 64
```

At 32 contacts per environment, token packing changed from 270.96 ms to
1.01 ms for 512 environments and from 536.27 ms to 1.07 ms for 1024
environments. The optimized pack time was nearly flat over that range.

With the real Anymal rough checkpoint, the synchronized `adapter_update`
section changed from 54.21 ms to 11.22 ms at 64 environments and from
320.04 ms to 12.50 ms at 512 environments. End-to-end environment step time
changed from 87.68 ms to 41.76 ms at 64 environments and from 377.71 ms to
72.19 ms at 512 environments. Results depend on contact density and hardware.

## Training configuration

KV-cache inference requires a model trained with RoPE:

```yaml
network:
  transformer:
    position_encoding: rope
```

Existing checkpoints use learned absolute position embeddings unless their
training configuration explicitly selects `rope`. They continue to load and run
through the existing full-window API, but cannot use `forward_cached()`.

## Incremental API

Pass only the latest timestep to `forward_cached()`. Cache ownership is explicit,
so deployment code can keep separate caches for separate model instances:

```python
model.eval()
cache = None
with torch.inference_mode():
    prediction, cache = model.forward_cached(
        latest_inputs,
        cache,
        max_cache_length=10,
    )
```

The returned prediction has the existing single-step shape `(B, 1, output_dim)`.
To reset all environments:

```python
model.reset_transformer_cache(cache)
```

To reset selected batch rows after asynchronous environment resets:

```python
model.reset_transformer_cache(cache, batch_indices=reset_env_ids)
```

`forward_cached()` requires `eval()` and `torch.inference_mode()` or
`torch.no_grad()`. The cache is runtime state and is not part of a checkpoint.

## Sliding-window behavior

Before the cache reaches its capacity, cached output matches a RoPE full-prefix
forward within floating-point tolerance. Once a fixed-size cache starts evicting
tokens, its semantics differ from recomputing the old sliding window:

- cached hidden states retain information from tokens that have since been
  evicted;
- a full-window recomputation removes that earlier context from every layer.

This is inherent to multi-layer contextual KV caching. A retrained RoPE model
must therefore be evaluated in closed loop before deployment; the speedup is not
an accuracy-equivalence claim for legacy checkpoints.

## Model benchmark

Run the model-only benchmark without Isaac Sim:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_model_forward \
  --device cuda:0 --warmup 20 --iterations 200
```

On the development machine, the representative Anymal-sized configuration
(`B=512`, history `10`, 6 layers, 12 heads, embedding size 384) produced:

- full-window inference: 6.352 ms;
- KV-cache inference: 2.415 ms;
- speedup: 2.63x;
- maximum prefix output error: `3.02e-7`.

Results depend on GPU, PyTorch version, batch size, and model dimensions. The
benchmark uses synthetic inputs because the branch does not contain a trained
NeRD checkpoint.

## Policy deployment benchmark

Use the deployment benchmark to measure a complete policy-closed-loop rollout
without section-level CUDA synchronization:

```bash
./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_deployment \
  --checkpoint /path/to/nerd.pt \
  --policy-checkpoint /path/to/policy.pt \
  --num-envs 1 --warmup 200 --steps 1000 --headless --device cuda:0
```

The benchmark includes policy inference, four NeRD physics substeps, observation
and reward computation, termination handling, and resets. It synchronizes CUDA
only before and after the complete timed rollout.

Two paired runs with the optimized native contact adapter produced:

- 1 environment: 29.203 ms/step baseline and 29.092 ms/step with the cache
  (difference within noise);
- 512 environments: 60.490 ms/step baseline and 58.838 ms/step with the cache
  (1.03x speedup);
- 1024 environments: 98.331 ms/step baseline and 91.999 ms/step with the cache
  (1.07x speedup).

These numbers exclude rendering. Add `--use-feature-cache` to enable the
compatible feature-cache path.
