Fixed
^^^^^

* **Breaking:** Aligned the NeRD Franka Lift static tabletop with the standard
  SeattleLabTable collision surface at z=-0.003 m. Corrected the table's identity
  quaternion to XYZW and stopped resting cubes from continuously earning the
  standard lifting reward. Regenerate Lift datasets in fresh directories,
  recompute normalization statistics, retrain dynamics models, and revalidate
  or retrain policies because the support height changed. Old checkpoints are
  not automatically rejected.
* Preserved the standard Franka Lift parent robot configuration and its overrides
  instead of replacing it with a separate inlined copy. Kept native contacts,
  the dynamic cube, standard MDP terms, and the non-articulated support surface.
