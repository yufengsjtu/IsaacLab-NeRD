Added
^^^^^

* Added weighted multi-source training through ``algorithm.dataset.train_datasets``
  (``WeightedMixedTrajectoryDataset`` / ``create_weighted_trajectory_dataset``) for
  :class:`~isaaclab_neural.train.trainers.SequenceModelTrainer`, ported from the G1
  recipe. ``train_dataset_path`` keeps working; configuring both is an error.
* Added dataset-generation options for coverage-oriented Franka data:
  ``--policy-action-noise-std`` (with ``--no-clip-policy-action-noise``) perturbs
  policy actions at each control tick, ``--action-hold-frames`` accepts several
  values and samples one hold length per action segment, and
  ``--rl-action-contract`` samples random actions through the task's RL action
  term when the config exposes ``restore_reach_rl_arm_action()``.
* Added the ``franka_reach_mixed`` and ``franka_reach_mixed_smoke`` OSMO presets and
  ``Franka/transformer_reach_mixed.yaml``: multi-timescale random walk, RL-contract
  random actions, noisy policy rollouts and clean policy rollouts, trained with
  weights 0.35 / 0.2 / 0.25 / 0.2 and learning rate ``2e-4 -> 2e-5``. Dataset specs
  may now set ``num_transitions``, ``policy_checkpoint``,
  ``policy_action_noise_std``, ``action_hold_frames`` and ``rl_action_contract``
  per file. Use the new ``franka-reach-legacy-60hz-mixed`` cache rather than reusing
  single-source caches.
