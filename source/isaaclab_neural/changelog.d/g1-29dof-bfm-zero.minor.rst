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
  ``Isaac-G1-29DoF-BFM-Push-Dataset-Gen-v0`` environments.
