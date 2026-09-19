Changed
^^^^^^^

* **Breaking:** Switched Franka Reach Dataset-Gen, policy, and NeRD environments
  from the Legacy Panda to the core Lift Menagerie Panda and calibrated drives.
  Added a policy environment matching data collection and moved maintained Reach
  datasets and OSMO caches to ``menagerie-60hz`` names. Regenerate data,
  normalization statistics, control policies, and NeRD checkpoints.
* **Breaking:** Rebased Franka Lift Dataset-Gen and NeRD on core
  ``Isaac-Lift-Franka`` with its Menagerie Panda, 120 Hz physics, 30 Hz relative
  joint-position action, observations, rewards, and resets. Pinned the object to
  one cube, replaced the kinematic table with a static collider retaining its top
  surface but extending downward to prevent reset-drop tunneling, excluded the
  fixed-base link-1 collider from reset ground-clearance
  checks, and removed ADR/physics randomization so one NeRD model represents
  the nominal dynamics. Previous
  contrib-Lift datasets and policies are incompatible.
* Switched Lift dynamics to body-frame joint state plus shared per-body native
  contact tokens, predicting the Panda and free cube together while keeping the
  table static. Added policy-data training and OSMO recipes and a control-policy
  player that does not require exporting multi-observation-group RSL-RL actors.

Added
^^^^^

* Added post-solve right/left finger-to-cube normal-force labels, independent
  normalization and an auxiliary prediction head sharing the dynamics trunk.
  Supervised per-finger contact logits to gate no-contact predictions to zero
  without substituting geometric contact flags for force vectors.
  Published predictions to the existing filtered sensor buffers for unchanged
  Lift observations, clipping, history and contact rewards, with partial reset.
  Kept friction and unfiltered resultant-force prediction separate. Required
  matching force metadata and checkpoints, and moved Lift data/caches to
  ``lift-core-policy-normal-force-v1``. Regenerate old unlabeled datasets and
  retrain; validate physical force errors and no-contact false positives before
  closed-loop deployment. Rejected cached-inference modes that omit the new head.
* Added a 60/20/20 clean-policy, noisy-policy and independent zero-action Lift
  training recipe. Matched policy/mixed learning rates, training budgets and
  common validation seeds and sizes for the formal Franka comparisons.

Fixed
^^^^^

* Avoided forked DataLoader workers inheriting live Warp CUDA allocations by
  selecting ``spawn`` for multi-process loaders. Changed eager Lift recipes to
  main-process loading to avoid duplicating large arrays across spawned workers.
  Recheck loading throughput and worker memory before enabling workers.
* Added original-coordinate Lift rollout evaluation with explicit source-world
  routing, separate batches for repeated source worlds and exclusion of padding
  from metrics. Changed formal Lift simulation grids to 1024 replicas to match
  collection, without changing the training batch size or strict tolerances.
  Use matching source grids when evaluating existing datasets.
* Moved normalization statistics with model device transfers, cleared auxiliary
  force caches on recurrent resets, and rejected state-only checkpoint resumes
  that would silently disable the requested finger-force head.
* Converted every free-object pose and velocity when transforming neural state
  frames, recorded replica origins for scenes without terrain, and rebased static
  replicated-scene rollout evaluation onto destination origins.
* Separated neural contact inputs from Newton native contact buffers, filtered
  degenerate native pairs with invalid contact normals, and updated history,
  recurrent, Transformer-cache, and evaluation paths for multiple free objects
  and per-body contact tokens.
* Restricted ANYmal contact-MDP overrides to ANYmal tasks so shared NeRD entry
  points did not install locomotion terms in Franka Lift.

Removed
^^^^^^^

* **Breaking:** Removed the ``franka_reach_newton_native_smoke``,
  ``franka_reach_policy_train_smoke``, ``franka_reach_mixed_smoke``, and
  ``franka_reach_replay_smoke`` OSMO preset entry points. Use a local copy of a
  maintained Reach preset with ``--preset-file`` for reduced training budgets,
  matching GPU resource and process counts and a separate dataset cache name.
  Use ``isaaclab_neural.eval.replay_franka_dataset`` for dataset playback.
