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
  tools). Frame-cached transformers compile the newest frame's contact-token transform, and streaming models
  also compile their fixed-shape steady-state frame encoding and stream step, with ``torch.compile``; window
  builds, reset-row rebuilds and training preprocessing stay eager. FP32 compiled inference matched eager
  inference up to roundoff (state 1.1e-5, force 2.6e-4 N max) and mean FP16 error against an FP32 reference
  stayed at FP16 level. On an idle GPU at 8192 SO101 worlds, FP16 throughput rose from 59.7k to 69.7k control
  steps/s (MJWarp: 53.0k); 6-seed closed-loop success was 94.9% versus 95.2% before the stage-1 changes. The
  first steps include compilation time. Defaults to False.
* Changed NeRD resets to reuse their contacts: the reset sync already collides the reset state and packs its
  contact inputs, so the next physics frame skips both while body poses and velocities are bitwise unchanged
  (any intermediate state write or other collision pass recomputes them).
* Changed prefiltered contact packing to skip the redundant resolution mask and input re-filtering (two host
  synchronizations per frame); packed body ids and overflow were identical on 367 live frames.
* Changed ``convert_prediction_to_next_states`` to convert runs of consecutive scalar joints as one elementwise
  operation (identical results; SO101 has 114 scalar joints).
* Changed ``inference_compile`` to compile an elementwise contact-token transform for the newest frame
  (``contact_tokens_to_model_frame_elementwise``); the reference rotation's batched matrix products ran as slow
  per-token matrix-vector kernels (4.6 to 0.56 ms per frame at 8192 worlds, equal up to roundoff).
* Changed streaming frame-cached steps to stack only the key-feature frames the key head reads and to gather
  reset rows before stacking, instead of materializing the whole history window every frame.
