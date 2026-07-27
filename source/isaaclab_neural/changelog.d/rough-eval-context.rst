Changed
^^^^^^^

* Changed rough-terrain rollout evaluation to reproduce and validate the
  dataset terrain context and warm-start transformer state history. Regenerate
  rough-terrain evaluation datasets to include the required context metadata.

* Changed Newton-native rollout contact mismatches to diagnostic warnings
  because repeated reconstruction may produce different contact manifolds.
  Terrain, reset-state, and transformer-history context remain strictly
  validated.

Added
^^^^^

* Added random-window terrain/contact reconstruction diagnostics with
  trajectory, timestep, terrain coordinates, unordered Hausdorff distance, and
  nearest-neighbor feature errors for contact mismatches.

Fixed
^^^^^

* Fixed distributed training waiting for the process-group timeout when
  main-process validation or rollout evaluation fails.

* Fixed strict rollout evaluation rejecting physically equivalent native
  contact sets solely because their fixed-slot ordering or duplicate manifold
  points differed.

* Fixed contact-token context diagnostics to report point, normal, lever-arm,
  gap, and relative-velocity errors separately.

* Fixed contact-token context diagnostics to record and validate root-link pose
  and root center-of-mass velocity replay.
