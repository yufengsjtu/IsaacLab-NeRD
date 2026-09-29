Added
^^^^^

* Added ``--camera-eye`` / ``--camera-lookat`` video options to ``eval/keyboard_policy.py``. Recording
  now showed the next-key marker for world 0 and wrote per-frame target/typed state to
  ``typing_overlay.json``; headless captures previously never dispatched marker callbacks.
* Added ``--nerd-checkpoint`` to ``isaaclab_neural.eval.keyboard_policy`` for closed-loop
  policy evaluation under learned dynamics with unchanged task, resets and terminations,
  plus ``--robot-usd`` for offline asset copies. Native evaluation remained the default.
* Added closed-loop typing evaluation during SO101 Keyboard training
  (``algorithm.eval.mode: keyboard_policy``). Rank 0 runs the typing policy in the training
  environment with the current weights, logs success and exclusive failure rates to W&B and
  keeps ``best_eval_model.pt`` by success rate. The throughput recipe and resident preset
  evaluated every five epochs with 64 worlds; other evaluation modes remained rejected for
  Keyboard training.
* Added ``isaaclab_neural.eval.keyboard_benchmark`` with ``model`` (model-only inference
  sweep) and ``env`` (policy-driven environment steps with physics and NeRD step-profile
  breakdowns) throughput modes.
