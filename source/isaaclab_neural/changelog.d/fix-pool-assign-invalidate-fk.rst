Fixed
^^^^^

* Fixed explicit Newton state writes that skipped Isaac Lab 3's dirty-world
  masks. :meth:`~isaaclab_neural.generate.adapter.NewtonDataGenerationBackend.assign_solver_states`
  and :meth:`~isaaclab_neural.envs.neural_env_wrapper.NeuralEnvAdapter.reset`
  now call :meth:`~isaaclab_newton.physics.newton_manager.NewtonManager.invalidate_fk`
  before :meth:`~isaaclab_newton.physics.newton_manager.NewtonManager.forward`.
  Without this, G1 action-mode perturb from an ``input_states_pool`` could NaN
  worlds on the first physics step, and NeRD eval restore could sync stale
  ``body_q`` / contact features. Poisoned MJWarp worlds stayed dead and later
  batches aborted with
  ``Dataset generation produced no valid transitions for a full rollout batch.``
