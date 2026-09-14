Fixed
^^^^^

* Fixed the shared NeRD RSL-RL train/play helper to apply ANYmal contact and
  reward overrides only to ANYmal tasks. Preserved G1 reward weights and
  robot-specific contact-body selectors when applying solver overrides.
* Preserved custom G1 NeRD reward weights, thresholds, contact-body selectors,
  and explicitly disabled terms during environment construction. Kept the
  geometry-based G1 NeRD defaults unchanged and rejected unsupported runtime
  contact modes before simulation initialization. Reapply intended MDP
  overrides when rerunning experiments that previously used reset defaults.
