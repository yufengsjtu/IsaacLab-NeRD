Added
^^^^^

* Added ``position_encoding: relative_lag`` with per-layer ``attention_spans`` to the NeRD Transformer.
  Layer ``l`` attends only to lags ``0..span-1`` with a learned per-head lag bias, so the receptive field is
  ``sum(span - 1) + 1`` frames. With ``use_frame_cache`` the solver then streams one position per layer and
  step (``GPT.build_stream_cache`` / ``forward_stream``), matching full-window inference up to FP32
  roundoff; reset rows rebuild their cone from the cached window. The receptive field must fit within
  ``num_states_history``. Checkpoints with other position encodings are unchanged.

Changed
^^^^^^^

* Changed both SO101 Keyboard recipes to the streaming trunk (spans ``[3, 3, 3, 2, 2, 2]`` in
  ``transformer_keyboard_throughput.yaml``, ``[3, 2]`` in ``transformer_keyboard_small.yaml``) with the frame
  cache and ``inference_precision: fp16``. The streaming model matched K closed loop (95.5% best checkpoint,
  95.3% with FP16, versus 95.2-95.5% for K) and ran 44.1k control steps/s at 8192 worlds (MJWarp 52.3k).
  **Migration:** checkpoints from the earlier learned-absolute K recipe still load and evaluate (add
  ``--frame-cache`` for faster inference), but cannot be resumed with the new recipes; train a new model
  or pass the old configuration explicitly.

* Changed streaming ``relative_lag`` steps to evaluate the short-span attention with elementwise products and to
  shift each layer's cached keys/values in place instead of concatenating new tensors and calling a masked SDPA,
  and changed inference-only key-head paths (``encode_frames(..., inference=True)`` and single-step decoding) to
  split the first summary/decoder layer by input block instead of materializing per-key concatenations. Results
  matched the previous implementation up to FP32 roundoff (closed-loop SO101 parity 7e-6 state, 3.4e-5 N force);
  the streaming trunk step at 8192 worlds dropped from about 13.5 to 3.2 ms (FP16). Training paths are unchanged.
