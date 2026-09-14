Changed
^^^^^^^

* Reduced the Franka Reach Dataset-Gen ``soft_joint_pos_limit_factor`` from
  ``0.8`` to ``0.6`` so random incremental targets stay farther from the USD
  joint limits. With the Legacy drives this lowered joint-limit crossings in
  random splits from about 17% to about 6% of frames without changing the
  robot, drives, or physics timing. Random datasets recorded with the previous
  factor cover a wider joint range; regenerate them under a new cache name
  rather than mixing the two reset envelopes.
