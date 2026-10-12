Added
^^^^^

* Added :data:`~isaaclab_neural.robots.g1_29dof.G1_29DOF_BFM_CFG`, a Unitree G1 29-DoF articulation without hands
  whose actuators follow the BFM-Zero deployment configuration. Its spawner localizes the Isaac Lab
  ``g1_29dof.usd`` root layer, which references payloads through absolute HTTPS URLs that OpenUSD cannot resolve
  without Kit; ``ISAACLAB_NEURAL_G1_29DOF_USD`` overrides the asset path.
* Added :class:`~isaaclab_neural.controllers.bfm_zero.BfmZeroLatentAction`, an action term that runs a frozen
  BFM-Zero actor from a 256-dimensional latent at 50 Hz with the humanoidverse observation convention. The
  CC BY-NC 4.0 weights are not distributed; export them to a file and set ``actor_path`` or ``BFM_ZERO_ACTOR_PATH``.
* Added root force pulses through Newton's floating-base generalized force (``apply_root_wrench_pulses``) and
  lying-start resets for G1 29-DoF data collection, plus the ``Isaac-G1-29DoF-BFM-Dataset-Gen-v0`` and
  ``Isaac-G1-29DoF-BFM-Perturbed-Dataset-Gen-v0`` environments.
* The 29-DoF walking teacher task used the flat-terrain gait rewards and terminations of isaac-sim/IsaacLab PR #8055
  (swing/stance duration variance, double-flight and pelvis-height penalties), its standing pose, a 0.9 soft
  joint-limit factor and the ``0.25 * effort_limit / stiffness`` action scale, while keeping the BFM-Zero actuators.
* Added geometry-contact versions of the PR #8055 gait terms (``geometry_feet_flight`` and
  ``GeometryFeetAirTimeVariance``). ``Isaac-Velocity-Flat-G1-29DoF-NeRD-v0`` trained the same walking task as the
  teacher on NeRD physics, counting feet within 2.5 mm of the ground as in contact, with the teacher's
  ``[512, 256, 128]`` PPO runner.
