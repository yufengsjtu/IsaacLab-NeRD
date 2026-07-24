Added
^^^^^

* Added lazy HDF5 dataset loaders selectable through
  ``algorithm.dataset.load_mode`` to reduce training memory usage while
  preserving the existing batch schema.
* Added distributed multi-GPU training with synchronized dataset statistics,
  rank-local data sampling, and rank-zero logging, validation, and checkpoint
  output.
* Added evaluation video recording and benchmark tools for HDF5 loading and
  upstream Newton CUDA graph stepping.

Changed
^^^^^^^

* Changed eager sequence datasets to load one trajectory shard per distributed
  rank and merge dataset statistics globally, avoiding full dataset replication
  across ranks without changing checkpoint statistics.
* Changed lazy loaders to support rank-4 contact-token batches and use pinned
  memory, non-blocking transfers, persistent workers, and configurable
  prefetching. Validation loaders now restart their sampling order each epoch.
* Changed ``use_cuda_graph`` to follow
  :class:`~isaaclab_newton.physics.newton_manager_cfg.NewtonCfg` and live only
  on :class:`~isaaclab_neural.physics.nerd_newton_cfg.NerdNewtonCfg`. Remove
  ``use_cuda_graph`` from :class:`~isaaclab_neural.physics.nerd_solver_cfg.NerdSolverCfg`
  configurations; legacy checkpoints remain supported. NeRD neural solvers
  always run eagerly because their physics steps include PyTorch.

Fixed
^^^^^

* Fixed native contact inputs to use world-frame contact points consistently
  before body-frame conversion.
* Fixed generated trajectories so contact inputs align with the current state
  instead of the post-step state, and report an error when a rollout batch
  produces no valid transitions.
* Fixed identical state embedding with a caller-provided output tensor to
  return that tensor instead of ``None``.
