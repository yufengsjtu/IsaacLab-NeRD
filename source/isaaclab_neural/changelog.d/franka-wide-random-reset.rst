Changed
^^^^^^^

* Expanded random-action Franka Reach resets to sample all seven arm joint
  positions uniformly within their soft limits, with zero initial velocities
  and fingers at their clamped default positions. Increased local-wrapper and
  random-data OSMO trajectory lengths from 100 to 200 physics frames while
  retaining 20-frame target holds. Policy and zero-action splits retained their
  original Reach reset. Regenerate random datasets, recompute training
  statistics, and use the new ``franka-reach-wide-reset-200`` cache (or its
  ``-smoke`` variant) instead of reusing the previous reset distribution.
