Changed
^^^^^^^

* **Breaking:** Changed contact-enabled HDF5 datasets to store only reusable
  ragged native contact pairs. Regenerate existing datasets; flat and contact
  token model inputs are now projected while loading.
* Changed OSMO dataset cache paths so flat, contact-token, and shared-per-body
  models reuse one canonical native dataset for each terrain.

Fixed
^^^^^

* Fixed rollout evaluation to project native contact history before
  warm-starting sequence models.
