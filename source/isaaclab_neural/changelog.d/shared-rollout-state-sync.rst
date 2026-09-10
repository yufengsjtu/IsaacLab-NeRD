Fixed
^^^^^

* Fixed dataset policy collection to refresh commands and observations each
  control interval. Regenerate policy datasets recorded with stale observations.
* Fixed direct state writes and neural resets to invalidate Newton kinematics
  before forwarding, and refilled partial Transformer histories from reset state.

Added
^^^^^

* Added ``--action-hold-frames`` for controlling random-action collection intervals.
  The default retained the environment control decimation.
