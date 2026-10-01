Added
^^^^^

* Added ``NerdSolverCfg.use_frame_cache`` for ``TransformerNeuralSolver``. Each step preprocesses and
  encodes only the newest frame, reuses earlier frame encodings (re-encoding reset rows or externally
  loaded windows) and still attends over the full window, so predictions and auxiliary force heads
  matched uncached inference up to FP32 roundoff (SO101 K closed loop: 2.7e-6 max state and 9.5e-6 N
  max force deviation over 600 frames). It requires per-frame anchoring and excludes the feature/KV
  caches. The SO101 throughput recipe enables it; ``keyboard_policy`` and ``keyboard_benchmark env``
  accept ``--frame-cache/--no-frame-cache`` to override older checkpoints.
* Added ``ModelMixedInput.encode_frames()`` / ``forward_encoded()`` and ``GPT.forward(last_only=True)``.

Changed
^^^^^^^

* Changed single-step ``ModelMixedInput`` inference to evaluate the final Transformer block, state
  decoder, keyboard key head and auxiliary heads only at the newest position. Outputs were unchanged
  up to floating-point roundoff; full-window training outputs were unaffected.
* Added ``NerdSolverCfg.inference_precision`` (``fp32`` default, ``tf32``, ``fp16``, ``bf16``) for online
  model inference on CUDA, with ``--inference-precision`` overrides in ``keyboard_policy`` and
  ``keyboard_benchmark env``. Reduced precisions changed predictions beyond FP32 roundoff and must be
  validated closed loop before use.
* Changed native contact packing to gather shape/body/world ids with bounded ``torch.where`` lookups and a
  single ``nonzero`` per frame instead of boolean-mask indexing per field, and cached the fixed-slot filter
  tensor. Outputs were unchanged (verified against the previous helpers on 1478 live SO101 frames); host
  synchronizations per NeRD physics frame dropped from about 49 to 8.
* Added ``NewtonContactAdapter(retain_native_pairs=False)``, used by the online NeRD solver: with a contact
  filter, pairs the filter rejects are dropped before per-pair geometry is built (SO101 frames contain mostly
  inactive key-ground pairs). Packing selected the same tokens, bodies and overflow counts on 299 live
  frames, with token geometry equal up to FP32 roundoff (max 4.8e-7); adapter time per control step at 8192
  worlds fell from about 84 to 34 ms on a shared GPU. ``native_contact_pairs()`` raises in this mode; data
  collection keeps the default ``True``.
* Added ``NerdSolverCfg.inference_compile`` (``--inference-compile`` in the keyboard evaluation and benchmark
  tools). Streaming frame-cached transformers compile their fixed-shape steady-state frame encoding and stream
  step with ``torch.compile``; window builds and reset-row rebuilds stay eager. Mean closed-loop FP16 error
  against an FP32 reference stayed at FP16 level (state 1.3e-5, force 7.2e-5 N); model time per control step at
  8192 SO101 worlds fell by about 35%. The first steps include compilation time. Defaults to False.
* Changed streaming frame-cached steps to stack only the key-feature frames the key head reads and to gather
  reset rows before stacking, instead of materializing the whole history window every frame.
