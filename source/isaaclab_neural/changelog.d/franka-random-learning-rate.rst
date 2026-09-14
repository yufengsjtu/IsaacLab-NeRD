Changed
^^^^^^^

* Reduced the Franka Reach random-action dynamics training learning-rate
  schedule from ``1e-3 -> 1e-4`` to ``2e-4 -> 2e-5`` while leaving the policy-data
  recipe unchanged. Start a new run with the updated configuration and datasets
  matching its robot asset and dynamics; existing jobs and checkpoints were not
  modified.
