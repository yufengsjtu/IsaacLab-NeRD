Changed
^^^^^^^

* Changed G1 action-mode dataset generation to hold stock numbered hand joints
  (``*_zero/one/two/three/four/five/six_joint``) at the trajectory reset pose
  instead of sampling random targets. Policy-mode walking is unchanged.
  Pass ``--no-hold-g1-hand-actions`` to restore full 37-DoF random actions.
  **Migration:** regenerate under the ``*-stock-holdhand`` OSMO cache keys;
  do not reuse a prior stock mix or perturb cache that randomized the hands.
