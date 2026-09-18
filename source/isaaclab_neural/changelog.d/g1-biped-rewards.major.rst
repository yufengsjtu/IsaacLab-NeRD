Changed
^^^^^^^

* **Breaking:** Aligned G1 NeRD feet-air-time rewards with stock biped single-support semantics and
  inherited terrain-specific weights. Contact durations were tracked at physics-step frequency.
  Existing policies must be revalidated or retrained; the previous ``GeometryFeetAirTime`` term
  remained available for existing tasks and explicit legacy configurations. Raw returns across
  reward definitions were not directly comparable.
* Enabled the flat G1 NeRD geometry-contact-history sliding penalty with the stock weight of -0.1,
  retaining configuration C as the default; rough terrain kept zero weight. Existing policies must
  be revalidated. To reproduce the previous single-support-only objective, set
  ``env.rewards.feet_slide.weight=0.0`` explicitly during training or evaluation.

Added
^^^^^

* Added a geometry-contact-history foot sliding reward term using horizontal foot COM speed.
* Added flat G1 NeRD failure termination for extreme root height and velocities, and rejected
  non-finite states before reward computation. New and legacy reward experiments must use
  identical outlier guards for meaningful comparisons.
