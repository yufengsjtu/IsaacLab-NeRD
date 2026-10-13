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
  I-BFM Carry Box runtime, with optional re-approach and backoff after failed grasps.
* Added the ``isaaclab_neural.eval.ibfm_box_demo`` script for Carry, Push and Kick Box on Newton MJWarp. It replaces
  the asset colliders with the I-BFM training MJCF colliders through a referencing USD layer, starts the box in the
  expert-data orientation and runs the released Push and Kick Box controllers in a subprocess of the I-BFM
  environment. Carry succeeded in all 192 randomized starts, push in 97% and kick in 75% of 32 starts.
* Added the ``isaaclab_neural.eval.bfm_zero_demo`` script for BFM-Zero tracking, reward-prompt and push-recovery
  demos, and :class:`~isaaclab_neural.eval.newton_video.NewtonFollowVideo` for headless Newton GL recording with a
  following camera.
