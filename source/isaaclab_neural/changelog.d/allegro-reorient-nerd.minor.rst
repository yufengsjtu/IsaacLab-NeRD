Added
^^^^^

* Added Allegro in-hand cube reorientation NeRD tasks
  (``Isaac-Reorient-Cube-Allegro-Policy-v0``, ``-Dataset-Gen-v0`` and ``-NeRD-v0``)
  that predict the hand and the free cube together with Newton-native contacts,
  a mixed-data training recipe, a dataset generation script and the
  :mod:`isaaclab_neural.eval.allegro_reorient_policy` closed-loop evaluator.
* Added the ``allegro_policy`` training evaluation mode, which runs the teacher
  closed loop in the training environment and selects ``best_eval_model`` by goals
  per episode.
* Added :mod:`isaaclab_neural.train.train_control_policy` to train RSL-RL teachers on
  tasks registered by :mod:`isaaclab_neural.envs`, including distributed runs.
* Added :mod:`isaaclab_neural.eval.allegro_benchmark` for model-only and end-to-end
  MJWarp/NeRD throughput comparisons, and made the
  :mod:`isaaclab_neural.eval.keyboard_benchmark` model and environment timers reusable
  for other model widths and decimations.
* Added Allegro concurrent shard collection, a post-training throughput benchmark stage
  with a MuJoCo-contact MJWarp baseline, and teacher/pipeline smoke presets. The recipe
  now filters native pairs to penetrating contacts and uses fused Adam.
* Added :mod:`isaaclab_neural.generate.allegro_dataset`, an env-driven Allegro collector of
  reset-safe physics-frame or control-step clips with a ``recording_contract``, and the
  SO101-style control-step recipe and presets. Unique-transition indexing and movable
  contact filtering read ``recording_contract`` when ``keyboard_contract`` is absent, and
  training rejects such recordings when their step mode or body mobility differs from the
  solver.
