Active15 body-routed contact encoding
-------------------------------------

Added an owner-body-frame Active15 contact representation and the original
shared-phi, per-body sum-pooling, shared-rho encoder. The representation keeps
Newton's solver-active contact gate, raw-point midpoint relative velocity, full
contact margins, independent HDF5 metadata, and an A-only categorical
normalization path. OSMO presets support generation to standalone Swift while
retaining the same dataset in pool-wide Lustre, followed by Lustre-first
multi-seed training.
