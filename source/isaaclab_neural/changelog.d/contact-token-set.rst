Added
^^^^^

* Added PhysicsNeMo-style directed :attr:`~isaaclab_neural.contacts.contact_set_schema.CONTACT_TOKEN_DIM`-channel contact token sets with pair-atomic packing, masked set encoding, regime/offline diagnostics, and dedicated Anymal native training configs.

Fixed
^^^^^

* Fixed HDF5 rollout validation to accept 2D per-step ``contact_token_overflow`` arrays with shape ``[trajectories, steps]``.
* Fixed dataset RMS computation for ``contact_tokens`` mode so frame conversion no longer injects
  ``None`` flat ``contact_*`` keys that incorrectly required ``contact_masks``.
* Fixed :meth:`~isaaclab_neural.solvers.transformer_neural_solver.TransformerNeuralSolver.get_neural_model_inputs`
  history batching to support 2-D overflow and 4-D ``contact_tokens`` tensors.
* Fixed contact-token regime evaluation so invalid padding tokens with gap ``0`` are not treated as touchdown.
