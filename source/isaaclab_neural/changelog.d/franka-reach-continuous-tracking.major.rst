Changed
^^^^^^^

* **Breaking:** Inherited the continuous-tracking Franka Reach MDP in NeRD Reach
  and Dataset-Gen: disabled success termination and its reward, restored the
  position tanh reward, disabled the action-magnitude reward term, and retained
  success metrics, the timeout, action-rate and joint-velocity penalties. Expanded
  the joint-velocity term to all robot joints, including fingers, while retaining
  its weight and curriculum. Kept
  contact-free scenes and dynamics unchanged. Use a distinct RL experiment
  name and re-evaluate policies rather than comparing returns to runs with
  success-triggered resets. To restore the previous MDP, restore the base
  Reach ``RewardsCfg().success``, ``RewardsCfg().action_magnitude``, and
  ``TerminationsCfg().success`` and disable
  ``end_effector_position_tracking_fine_grained``. To restore the arm-only
  velocity penalty, set its ``asset_cfg.joint_names`` to ``["panda_joint.*"]``.
