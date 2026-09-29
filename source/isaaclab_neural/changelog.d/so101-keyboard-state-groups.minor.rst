Added
^^^^^

* Added optional ``algorithm.loss.keyboard_state_group_weights`` for SO101 Keyboard
  training, balancing robot and key state-loss groups independently of their
  coordinate counts. The default ``null`` retained the existing variance-weighted
  mean; per-coordinate weights and force losses were unchanged.
* Added per-group variance-weighted state errors (``robot_q``, ``robot_qd``,
  ``key_q``, ``key_qd``) to Keyboard training and validation logs.
* Added an optional factorized ``network.key_head`` for SO101 Keyboard models: a
  Transformer trunk over robot state and a key summary, plus one shared per-key decoder
  using pooled key statistics. Outputs retained the original prediction order and solver
  interface; models without the key head and existing checkpoints were unchanged.
* Added optional ``keyboard_key_velocity_weighting: pooled`` and
  ``keyboard_key_event_weighting`` Keyboard state losses. Defaults kept the previous objective.
* Added the ``SO101/transformer_keyboard_formal_key_head.yaml`` recipe and the
  ``so101_keyboard_formal_key_head`` OSMO preset (key head, pooled key-velocity weights,
  0.2 mm key-position scale, balanced groups, eager full-data loading).
