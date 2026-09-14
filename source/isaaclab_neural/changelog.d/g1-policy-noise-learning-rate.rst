Changed
^^^^^^^

* Reduced the G1 policy-noise dynamics training learning-rate schedule from
  ``1e-3 -> 1e-4`` to ``2e-4 -> 2e-5`` while leaving the mixed-data recipe
  unchanged. Start a new run with the updated configuration; existing running
  jobs and checkpoints were not modified.
