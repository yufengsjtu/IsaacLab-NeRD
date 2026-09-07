Fixed
^^^^^

* Fixed :meth:`~isaaclab_neural.solvers.transformer_neural_solver.TransformerNeuralSolver.reset`
  so episode resets drop or refill transformer history from the current
  state instead of writing zeros, which caused immediate re-falls after
  the first termination.
