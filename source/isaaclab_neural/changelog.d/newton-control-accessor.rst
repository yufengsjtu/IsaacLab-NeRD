Fixed
^^^^^

* Fixed training-time ``joint_f`` rollout evaluation and the joint-force diagnostics
  after Isaac Lab moved the Newton control buffer behind
  ``NewtonManager.get_control()``. Evaluation no longer fails with
  ``AttributeError: ... has no attribute '_control'``.
