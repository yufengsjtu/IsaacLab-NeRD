Added
^^^^^

* Added PhysicsNeMo-style directed :attr:`~isaaclab_neural.contacts.contact_set_schema.CONTACT_TOKEN_DIM`-channel contact token sets with pair-atomic packing, masked set encoding, regime/offline diagnostics, and dedicated Anymal native training configs.

Fixed
^^^^^

* Fixed HDF5 rollout validation to accept 2D per-step ``contact_token_overflow`` arrays with shape ``[trajectories, steps]``.
