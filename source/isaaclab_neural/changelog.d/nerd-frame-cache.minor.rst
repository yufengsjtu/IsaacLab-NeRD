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
