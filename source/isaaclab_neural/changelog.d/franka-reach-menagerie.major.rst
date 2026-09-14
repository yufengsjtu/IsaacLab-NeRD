Changed
^^^^^^^

* **Breaking:** Unified Reach Dataset-Gen and NeRD defaults with the historical
  Legacy Panda control policy: Isaac 6.0 Legacy USD, explicit Legacy PD, 1/60 s
  physics, decimation 2, and one solver substep. Preserved contact-free scenes,
  wide random resets, and 200-frame random clips; those clips now span about
  3.33 s. Left Lift unchanged. Removed temporary Menagerie experiment profiles
  from ordinary Reach. Moved local datasets to ``data/datasets/legacy-60hz/Franka-Reach``
  and Reach OSMO caches to ``franka-reach-legacy-60hz-*``. Regenerate all splits,
  recompute statistics, and retrain NeRD rather than reusing Menagerie or
  different-time-step data/models. Existing artifacts were not deleted; checkpoint
  loading does not automatically reject incompatible dynamics models.
* Derived Franka Reach random-action increments from resolved joint effort limits and
  stiffness at initialization via ``IncrementalJointPositionActionCfg.scale=None``.
  Preserved explicit numeric/dict scales and the legacy ``FRANKA_REACH_DATASET_INCREMENT_SCALE``
  constant for callers reproducing the old recipe.
