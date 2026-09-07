Fixed
^^^^^

* Fixed :class:`~isaaclab_neural.physics.nerd_newton_manager.NewtonNerdManager`
  startup after Isaac Lab 3 renamed ``NewtonManager._views`` to
  ``PhysicsManager.views``. Environment creation no longer fails while
  mirroring subclass Newton state onto the upstream manager.
