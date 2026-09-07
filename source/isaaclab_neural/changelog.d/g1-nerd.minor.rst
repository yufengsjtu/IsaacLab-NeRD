Added
^^^^^

* Added Unitree G1 NeRD and dataset-generation gym tasks
  (``Isaac-Velocity-Flat-G1-NeRD-v0``, ``Isaac-Velocity-Rough-G1-NeRD-v0``,
  ``Isaac-Velocity-Flat-G1-Dataset-Gen-v0``, and
  ``Isaac-Velocity-Rough-G1-Dataset-Gen-v0``) with matching NeRD train configs
  under ``train/cfg/G1/``. Teacher rollouts use the stock
  ``Isaac-Velocity-Flat-G1`` policy.
* Added ``--policy-action-noise-std`` / ``policy_action_noise_std`` and
  ``--no-clip-policy-action-noise`` so policy dataset generation can inject
  control-rate Gaussian action noise (held between env decimation ticks).
  G1 mix policy-noise leaves the noisy actions unclipped.
* Added ``--resample-actions-every-frame`` so action-mode dataset generation
  can sample a new action on every recorded physics frame. G1 pool-perturb
  uses this instead of holding the action across env decimation.
* Added G1 ``dataset_mix`` OSMO recipes that expand six weighted sources
  (policy, random, zero, policy-noise, rl-checkpoint mix, and pool-perturb
  rollouts started from a teacher-policy state pool). Default training uses
  20M transitions with ratios ``2/0/0/2/2/4``. Validation always keeps the
  original policy, random, zero, and deployment splits.
* Added a 20M 100% pool-perturb OSMO preset
  (``g1_newton_native_shared_per_body_perturb``) that matches the Lab-policy
  ``HumanoidG1JPC_policy_perturbed`` training split while keeping contact-token
  shared-per-body training. The teacher state pool has the same transition
  count as the perturb train split. Validation keeps walking, env-reset
  random/zero, pool-perturb, and deployment splits.
  Training uses ``num_iters_per_epoch: 2500`` so one eight-GPU epoch covers
  about one pass over the 20M windows. It uses a separate cache from the mix
  preset.

Changed
^^^^^^^

* Changed G1 ``dataset_valid.hdf5`` and 10-step rollout eval to stock G1
  teacher walking (``passive: False``), instead of zero-action Dataset-Gen
  trajectories.
* Changed formal G1 Newton-native OSMO collection to a single mix preset
  (``g1_newton_native_shared_per_body``) plus a scaled smoke mix. Removed the
  policy-only, random-only, zero-only, 4:3:3, and checkpoint-only recipes.
  **Migration:** set ``dataset_mix.ratios`` on the remaining mix preset instead
  of submitting a dedicated ablation workflow.
* Changed G1 NeRD collection to the stock ``Isaac-Velocity-Flat-G1``
  teacher (1500 PPO iterations). Mix collection samples checkpoints 500, 1000,
  and 1499, and deployment / policy-noise generation uses ``model_1499.pt``.
  **Migration:** copy ``model_1499.pt`` and the three mix snapshots into
  ``pretrained/control_policy/g1_flat_terrain/``.
* Changed formal G1 mix training to Lab-policy G1 optimizer and head size:
  linear ``1e-3`` to ``1e-4`` and MLP ``[128]``. Eight-GPU batch ``1024`` /
  ``1250`` iters are unchanged.
* Changed G1 mix generation to Lab-policy wall-clock lengths while still
  recording every physics frame: policy / noise / checkpoint / deployment
  trajectories use 1000 frames, pool-perturb uses 100 physics frames with a
  new random action every frame (no zero-action split and no PD
  randomization), and env-reset random / zero use 500 frames. Policy-action
  noise is unclipped.
  **Migration:** generate under
  ``g1-newton-native-contact-tokens-v3-active-mixed-20m-ckpt3-1499-stock``
  with ``--dataset-cache-mode off``. Do not reuse a prior 23-DoF mix cache.
* Changed the default G1 mix ratios to ``2/0/0/2/2/4`` (4M clean policy, 4M
  noise, 4M 3-checkpoint mix, 8M pool-perturb). Pool-perturb always uses a
  dedicated ``dataset_policy_pool.hdf5`` whose size equals the perturb train
  count (8M in the default mix).
  **Migration:** submit under the new stock ``dataset_subdir`` rather than
  overwriting a Hands-Fixed cache.
* Changed G1 Dataset-Gen velocity commands to independent
  ``vx in [-1, 1]``, ``vy in [-0.5, 0.5]``, ``yaw in [-1, 1]`` resampled every
  1 s, matching the Lab-policy G1 generator. NeRD RL tasks and the stock
  teacher keep the G1Flat command ranges.
* Changed G1 NeRD and dataset-generation tasks to use the stock
  ``G1_MINIMAL_CFG`` robot. Formal presets require the matching stock teacher
  under ``pretrained/control_policy/g1_flat_terrain/``.
* Changed OSMO ``package_code`` to force-include the gitignored stock G1
  teacher-policy checkpoints in the uploaded code tarball.

Fixed
^^^^^

* Fixed Newton-native geometry contact terminations when using contact-token
  models, so selected-body contacts use packed Newton body ids rather than
  encoder ``body_slot`` indices.
* Fixed :class:`~isaaclab_neural.envs.g1_nerd_env.NerdG1FlatEnvCfg` to default
  to ``newton_native`` contacts. Instantiating ``Isaac-Velocity-Flat-G1-NeRD-v0``
  previously used deprecated ``fixed_ground`` and failed before training started.
* Changed OSMO generate and train jobs to omit ``--headless``; headless is
  the default unless ``--viz`` is set.
