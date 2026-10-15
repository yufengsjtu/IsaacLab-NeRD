Added
^^^^^

* Added control-step NeRD recipes for flat ANYmal-C and Franka cube lift that follow the
  SO101/Allegro pipeline: env-driven reset-safe clips, one prediction per control step with
  joint-target input, unique-transition sliding windows, GPU-resident data, FP16 training
  autocast, FP16/compiled streaming inference and post-training throughput benchmarks
  (``anymal_flat_control_step`` and ``franka_lift_control_step`` presets, each with a smoke
  preset).
* Added ``Isaac-Velocity-Flat-Anymal-C-Policy-v0`` and
  ``Isaac-Velocity-Flat-Anymal-C-ControlStep-NeRD-v0``, which share the stateless ANYdrive
  actuator because a control-step model evaluates the actuator once per control step.
* Added :mod:`isaaclab_neural.eval.policy_rollout` for closed-loop policy evaluation of any
  manager-based task under MJWarp or NeRD (failure rate, velocity tracking error, pose success
  rate, optional follow-camera video) and the ``policy_rollout`` training evaluation mode. It
  selects ``best_eval_model`` closest to the native metric that the OSMO teacher gate passes in
  as ``algorithm.eval.selection.reference``.
* Added :mod:`isaaclab_neural.eval.task_benchmark`, a task-agnostic version of the Allegro
  throughput benchmark.

Changed
^^^^^^^

* The env-driven collector (:mod:`isaaclab_neural.generate.allegro_dataset`) accepts
  ``--env-name`` and ``--policy-task`` and records filtered finger-object normal forces for
  tasks that declare ``finger_object_normal_force_sensors``. Allegro defaults are unchanged.
