Fixed
^^^^^

* Fixed NeRD environment reset against Isaac Lab 3 Newton managers.
  :class:`~isaaclab_neural.physics.nerd_newton_manager.NewtonNerdManager`
  no longer forwards ``SolverBase.reset(state, world_mask=..., flags=0)``
  to :class:`~isaaclab_neural.solvers.transformer_neural_solver.TransformerNeuralSolver`,
  which uses ``reset(env_ids)`` for history. Neural solvers now also ignore
  the Newton-style keyword arguments.
