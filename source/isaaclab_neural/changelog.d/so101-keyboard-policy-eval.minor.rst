Added
^^^^^

* Added ``--camera-eye`` / ``--camera-lookat`` video options to ``eval/keyboard_policy.py``. Recording
  now showed the next-key marker for world 0 and wrote per-frame target/typed state to
  ``typing_overlay.json``; headless captures previously never dispatched marker callbacks.
* Added ``--nerd-checkpoint`` to ``isaaclab_neural.eval.keyboard_policy`` for closed-loop
  policy evaluation under learned dynamics with unchanged task, resets and terminations,
  plus ``--robot-usd`` for offline asset copies. Native evaluation remained the default.
