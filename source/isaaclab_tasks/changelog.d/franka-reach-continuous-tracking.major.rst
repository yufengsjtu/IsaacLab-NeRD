Changed
^^^^^^^

* **Breaking:** Changed Franka Reach to continuous tracking by disabling success
  termination and its terminal reward, and restoring the fine-grained position
  tanh reward with weight 0.1 and scale 0.1 m. Disabled the action-magnitude
  reward term while retaining action-rate and joint-velocity penalties. Expanded
  joint-velocity regularization to all robot joints, including fingers, without
  changing its weight or curriculum. Preserved pose-success thresholds for
  metrics and the time limit. These active rewards and timeout-only termination
  matched the archived September 5 Legacy policy, not the success-terminating
  upstream recipe. Left UR10 Reach unchanged. Train policies under a new
  experiment name rather than comparing returns across the two episode rules.
  To restore the upstream MDP, restore ``RewardsCfg().success``,
  ``RewardsCfg().action_magnitude``, and ``TerminationsCfg().success`` from ``isaaclab_tasks.core.reach.reach_env_cfg``
  and disable ``end_effector_position_tracking_fine_grained``. To restore the
  arm-only velocity penalty, set its ``asset_cfg.joint_names`` to ``["panda_joint.*"]``.
* **Breaking:** Restored the historical Legacy asset and drives as the ordinary
  Franka Reach default, with the Isaac 6.0 asset revision, 1/60 s simulation step,
  decimation 2, and one MJWarp substep. Removed the temporary asset/drive
  experiment selectors; no asset preset is needed for the Legacy recipe.
  Menagerie checkpoints require their original configuration for replay and
  must not be used with the simplified Legacy task. Datasets from another asset
  or time step must not be reused. Kept the separate OSC task on its existing
  Menagerie configuration.
