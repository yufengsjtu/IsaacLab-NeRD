Fixed
^^^^^

* Fixed policy collection to refresh commands and observations each control
  interval, and invalidated Newton kinematics after direct state writes.
  Regenerate policy datasets collected with stale observations.
* Fixed Transformer partial resets to refill history from the current state.
* Fixed implicit-actuator recording and inference to supply applied effort as
  neural input without overwriting recorded forces during dataset evaluation.
  Regenerate implicit-actuator datasets with zero recorded effort, recompute
  statistics, and retrain their models.

Added
^^^^^

* Added configurable action holding for physics-frame dataset collection.
