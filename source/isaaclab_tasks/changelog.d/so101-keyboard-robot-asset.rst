Changed
^^^^^^^

* Changed the SO101 Keyboard task to use
  ``omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Robots/SO101/so101.usda``
  without changing the shared ``SO101_CFG`` used by other tasks. Selected only
  the target asset's supported ``Physics`` variant set. This asset requires
  access to the isaac-dev Nucleus server. Existing policies must be revalidated
  against its dynamics; restoring the previous SysID asset also requires its
  ``Robot`` and ``Sensor`` variant selections.
* Restored the Keyboard task's six default joint positions to zero as the seed
  for its existing noisy reset IK and snapshot curriculum. Other SO101 tasks
  retained their shared asset defaults.
* Restored the Keyboard task's default robot joint armature to 0.028 while
  retaining USD-authored drive gains. The equivalent command-line armature
  override was no longer required for this task.
* Restored the Keyboard robot's requested joint velocity limit to 5 rad/s using
  ``joint_velocity_limit``. The task's existing velocity-limit termination used
  the resolved soft limit; solver-side enforcement remained backend-dependent.

Fixed
^^^^^

* Restored missing Franka Reach configuration imports that prevented optional NeRD
  task registration and the shared neural training entry point from loading.
