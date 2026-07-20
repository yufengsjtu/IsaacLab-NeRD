Added
^^^^^

* Added a contact reconstruction diagnostic for comparing dataset and runtime
  contact inputs and one-step predictions.

Changed
^^^^^^^

* **Breaking:** Changed Newton-native contact inputs to place the primary robot
  articulation first, orient normals from the robot toward the other shape,
  and store signed surface separation in ``contact_depths``. Regenerate
  existing Newton-native datasets and retrain their checkpoints before use.

Fixed
^^^^^

* Fixed NeRD extension metadata, default checkpoint paths, and dataset
  generation handling for invalid rollout batches.
* Fixed Newton-native dataset generation to use the same Newton collision
  pipeline as online NeRD evaluation.
* Fixed Newton-native inactive contact slots becoming nonzero after input
  normalization or noise injection by reapplying contact masks at the encoder boundary.
  Newton-native datasets now require explicit masks because padding cannot be
  reconstructed from signed surface separation.
* Fixed Newton-native contact normalization statistics to use only valid
  contacts while preserving independent statistics for each sorted slot.
  Sparse slots use pooled field statistics, and per-slot sample counts are
  saved in checkpoints for diagnostics.
* Fixed chunked HDF5 appends to validate complete schemas before writing and
  roll back dataset sizes if a write fails.
* Fixed Newton-native contact canonicalization when Newton exposes shape
  indices as 32-bit integers.
