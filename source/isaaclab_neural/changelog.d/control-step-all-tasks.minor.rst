Added
^^^^^

* Added :mod:`isaaclab_neural.tasks`, a simulator-free registry of NeRD task specs (policy, recording and NeRD task
  ids, teacher, contact capacity, auxiliary force labels, closed-loop metrics) used by the env-driven collector, the
  closed-loop evaluator, the task benchmark (``--task-spec``) and OSMO presets (``experiment.task_spec``).
* Added control-step recipes, smoke presets and native/recording/NeRD environments for Cartpole, rough-terrain ANYmal-C
  (with the per-body net contact-force head) and Franka Reach, plus ``cartpole_teacher`` and ``anymal_rough_teacher``.
* The env-driven collector records contact-free tasks (``--max-contact-tokens 0``) and per-body net contact-force
  labels (``--record-body-net-contact-forces``); ``policy_rollout`` gained the ``command_error`` metric.

Fixed
^^^^^

* Fixed Franka Reach configs for per-backend articulation schema fragments and pinned the Reach recipes to their
  60 Hz physics, 30 Hz control and single-substep contract after the core task moved to 120 Hz.
