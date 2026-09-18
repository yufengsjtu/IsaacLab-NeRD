Fixed
^^^^^

* Restored Newton post-step callback dispatch in both NeRD stepping paths so physics-step
  contact timers and registered state-publication hooks ran after the solver and before sensors.
