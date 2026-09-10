Added
^^^^^

* Added contact-free Franka Reach and contact-aware Franka cube-lift NeRD
  tasks, including dataset-generation environments
  ``Isaac-Reach-Franka-Dataset-Gen-v0`` and
  ``Isaac-Lift-Cube-Franka-Dataset-Gen-v0``.
* Added the ``franka_reach_newton_native`` OSMO preset for contact-free
  Franka Reach dataset generation and training, using the Anymal-C rough
  OSMO resource request (8 L40S GPUs, 80 CPUs) and data scale (ten million
  training transitions, trajectory length 100). The preset records a
  policy validation split from
  ``pretrained/control_policy/franka_reach/model_999.pt``.
* Added the ``franka_reach_newton_native_smoke`` OSMO preset (1 GPU, 16
  environments, 3200 transitions) to verify Franka Reach dataset generation.
* Added the ``franka_reach_policy_train`` OSMO preset so Franka Reach NeRD
  can train on Reach-policy rollouts instead of incremental random
  joint-position walks, with a matching ``franka_reach_policy_train_smoke``
  preset.
* Added :class:`~isaaclab_neural.envs.incremental_joint_position.IncrementalJointPositionAction`
  so dataset generation can hold ``q_target = clip(q + scale a, soft limits)``
  instead of chasing a receding relative offset.
* Added ``isaaclab_neural.eval.probe_franka_shadow_joint_f`` to check whether
  implicit Panda ``applied_effort`` can stand in for NeRD ``joint_f`` without
  swapping actuators to IdealPD.
* Added ``isaaclab_neural.eval.verify_franka_joint_f_wiring`` to check that
  Dataset-Gen records implicit Panda ``applied_effort`` as ``joint_f`` only
  after the analytic Newton step.
* Added worktree ``PYTHONPATH`` prefixes to the Franka dataset-generation
  wrapper scripts so a conda editable install of the main checkout still
  loads ``Isaac-Reach-Franka-Dataset-Gen-v0``.

Changed
^^^^^^^

* Changed Franka NeRD scene construction to spawn the Panda from an inlined
  Legacy Nucleus USD path instead of importing
  :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_CFG`, so OSMO jobs that
  install only ``isaaclab_neural`` do not inherit the image asset path.
* Changed Franka Reach OSMO dataset generation from open-loop ``joint_f``
  sampling to the same joint-position action path as Anymal-C (``sample_mode:
  action``, 1024 parallel environments). Random torques of ``±100 N·m`` on
  the Isaac Lab Panda exploded MJWarp states; position-target sampling stays
  within PD effort limits. ``joint_f`` sampling still disables CUDA graph
  capture because it never replays the graph.
* Changed Franka Reach dataset-generation random splits to incremental
  joint-position targets ``q_target = clip(q + scale a, soft limits)``,
  with a per-joint ``scale`` of ``effort_limit / kp`` so ``|a| = 1``
  saturates implicit PD at rest. Splits start from the Reach env reset
  (``initial_states_source: env``) and hold each absolute target for 20
  physics frames (``action_hold_frames=20``). The dataset-generation Panda
  uses ``soft_joint_pos_limit_factor=0.8`` as the walk envelope. Mapping
  random actions onto joint limits in one shot saturates implicit PD under
  MJWarp. Policy and zero-action splits keep the Reach RL map
  ``q_default ± 0.5 rad``.
* Changed :class:`~isaaclab_neural.solvers.neural_solver.NeuralSolver` to size
  generalized state per Newton world so a replica may contain more than one
  articulation, as required by Franka cube-lift.
* Changed native contact primary-body selection to include every articulated
  body in the world so object contacts are retained.
* Changed Franka Reach and cube-lift NeRD training configs to keep the
  upstream ``newton_mjwarp`` physics preset handle so OSMO
  ``presets=newton_mjwarp`` still resolves after ``sim.physics`` is replaced
  with :class:`~isaaclab_neural.physics.nerd_newton_cfg.NerdNewtonCfg`.
* Changed Franka Reach OSMO presets to use the public G1 develop image
  ``docker.io/rowany28070/isaac-lab-nerd:g1-develop`` and to overlay the
  uploaded Isaac Lab tree on that image. The previous ``3.0.0-beta2`` image
  still exposes ``isaaclab_tasks.manager_based`` and fails immediately on
  ``import isaaclab_tasks.core``.
* Changed Franka Reach Dataset-Gen EE pose resampling to 1.5 s, and
  ``franka_reach_policy_train`` clips to 240 physics frames (~2.0 s),
  so each policy trajectory records one Reach start-to-stop and the
  first 0.5 s of the next target. Regenerate policy HDF5 splits recorded
  under 0.5 s, 1 s, or 2 s resampling, or with length-100 or length-180
  clips. ``Isaac-Reach-Franka-NeRD-v0`` still uses the 4 s Reach command.
* Changed Franka Reach Transformer width to ``n_embd=384`` so it matches
  the Anymal native training configs. Checkpoints trained at ``n_embd=192``
  are not compatible.
* Changed implicit-actuator Dataset-Gen recording to copy Isaac Lab
  ``applied_effort`` (clipped shadow PD) into the HDF5 ``joint_f`` channel
  after the analytic Newton step. Implicit Panda/MJWarp leaves Newton
  ``control.joint_f`` at feed-forward zero, so policy ``q_target`` was
  previously dropped. Closed-loop NeRD copies the same telemetry into
  ``control.joint_f`` after actuator telemetry and before the neural
  forward; that write is model input only. Do not write shadow PD into
  ``control.joint_f`` before analytic MJWarp (double PD). Finger efforts
  can reach the 200 N·m clip and are kept as features. Regenerate Franka
  HDF5 splits recorded with zero ``joint_f`` and retrain; recompute
  dataset statistics. Old checkpoints trained on zero ``joint_f`` are not
  valid for closed-loop eval. Keep ``sample_mode: action``; ``joint_f``
  sampling still draws random torques and is unchanged.

Fixed
^^^^^

* Fixed OSMO generate, train, and diagnostic wrappers to omit the removed
  Isaac Lab 3 ``--headless`` flag. Apps are headless by default; passing
  ``--headless`` made ``generate_dataset`` exit with
  ``unrecognized arguments``.
* Fixed policy-mode dataset generation so command and observation managers
  tick each control interval without calling ``env.step``. Previously
  ``RslRlVecEnvWrapper.get_observations`` returned the reset ``obs_buf``,
  so the recorded policy action was constant for the whole clip.
* Fixed kinematic dataset replay so writing recorded ``joint_q`` invalidates
  Newton FK and skips CUDA graph capture. Without that, the visualizer kept
  the reset pose even when the HDF5 clip was moving.
* Fixed :class:`~isaaclab_neural.solvers.transformer_neural_solver.TransformerNeuralSolver`
  partial resets to copy the current post-reset state into history instead of
  zeros. Closed-loop Reach/Lift eval and RL no longer see a window of invalid
  zero poses after a subset of environments terminate.
* Fixed closed-loop ``isaaclab_neural.eval.eval`` policy rollouts so
  ``torch.inference_mode`` wraps only the policy forward. Wrapping
  ``env.step`` cloned Transformer history as inference tensors, then Reach
  success resets failed on ``index_copy_`` with ``Inplace update to inference
  tensor``.
* Fixed training 10-step dataset eval (``action_mode: joint_f``) so recorded
  HDF5 ``joint_f`` is not overwritten by implicit-actuator ``applied_effort``
  overlay inside :meth:`~isaaclab_neural.physics.nerd_newton_manager.NewtonNerdManager._simulate_physics_only`.
  Closed-loop ``env.step`` still overlays after actuators.
* Fixed ``isaaclab_neural.eval.eval --video`` to use Isaac Lab 3
  ``env_cfg.video_recorders`` instead of Gymnasium ``RecordVideo`` with
  ``render_mode=rgb_array``, which returned no frames. NeRD tasks now
  record from Newton GL rather than auto-injecting Kit/RTX, which crashed
  at Kit startup on a hybrid NVIDIA+Intel Vulkan host. ``--video`` alone
  stays headless; ``--viz newton`` (alias ``--vis newton``) opens a live
  window while recording. Pass ``--viz kit`` only when Kit recording is
  required.
* Fixed :meth:`~isaaclab_neural.envs.neural_env_wrapper.NeuralEnvAdapter.reset`
  and :meth:`~isaaclab_neural.physics.nerd_newton_manager.NewtonNerdManager.reset_neural_solver`
  to call :meth:`~isaaclab_newton.physics.newton_manager.NewtonManager.invalidate_fk`
  before :meth:`~isaaclab_newton.physics.newton_manager.NewtonManager.forward`
  after direct ``joint_q`` writes. Dataset eval restore and closed-loop resets
  otherwise keep stale ``body_q`` / MJWarp warmstart.
