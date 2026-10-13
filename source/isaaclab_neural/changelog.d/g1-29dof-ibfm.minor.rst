Added
^^^^^

* Added :mod:`isaaclab_neural.controllers.ibfm`, a PyTorch port of the I-BFM humanoid-object interaction policy
  (actor, backward-map log-map prompt and observation normalizers) and of its observation convention: the
  34-dimensional box context, the heading-local privileged state and hand-capsule box gaps. It reproduced the
  released I-BFM MuJoCo runtime to within 1.3e-6 on 400 recorded Carry Box steps. The CC BY-NC 4.0 weights are not
  distributed; :func:`~isaaclab_neural.controllers.ibfm.load_ibfm_policy` documents the export format.
* Added :data:`~isaaclab_neural.robots.g1_29dof.G1_29DOF_IBFM_CFG` with the actuator gains, torque limits and joint
  friction of the I-BFM robot configuration, which differ from BFM-Zero at the waist and wrists.
* Added :class:`~isaaclab_neural.controllers.ibfm_carry.IbfmCarryController`, the phased latent selection of the
  I-BFM Carry Box runtime, with optional re-approach and backoff after failed grasps, and the
  ``isaaclab_neural.eval.ibfm_carry_demo`` script that picks up, carries and places a box on Newton MJWarp.
