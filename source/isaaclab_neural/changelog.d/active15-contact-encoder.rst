Added
^^^^^

* Added owner-body-frame Raw15 and Active15 contact representations with the
  original shared-phi, per-body sum-pooling, shared-rho encoder. Raw15 keeps all
  canonical raw Newton candidates, while Active15 retains Newton's solver-active
  contact gate. Both use raw-point midpoint relative velocity, full contact
  margins, independent HDF5 metadata, and representation-specific categorical
  normalization. OSMO presets support generation to standalone Swift followed
  by Lustre-first multi-seed training.
