Added
^^^^^

* Added ``position_encoding: relative_lag`` with per-layer ``attention_spans`` to the NeRD Transformer.
  Layer ``l`` attends only to lags ``0..span-1`` with a learned per-head lag bias, so the receptive field is
  ``sum(span - 1) + 1`` frames. With ``use_frame_cache`` the solver then streams one position per layer and
  step (``GPT.build_stream_cache`` / ``forward_stream``), matching full-window inference up to FP32
  roundoff; reset rows rebuild their cone from the cached window. The receptive field must fit within
  ``num_states_history``. Checkpoints with other position encodings are unchanged.
* Added the ``transformer_keyboard_stream.yaml`` SO101 recipe (K recipe with spans ``[3, 3, 3, 2, 2, 2]``)
  and the ``so101_keyboard_stream`` OSMO preset. It trains a new model; K checkpoints cannot be resumed with it.
