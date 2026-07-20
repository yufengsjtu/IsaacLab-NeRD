Changed
^^^^^^^

* **Breaking:** Aligned the Anymal-C flat NeRD policy observations with the
  upstream 48-dimensional observation space and replaced the bundled
  ``model_499.pt`` policy with the compatible ``model_299.pt`` policy. Users
  with 49-dimensional policy checkpoints must migrate to an upstream-compatible
  checkpoint or restore the root-height observation.

Added
^^^^^

* Added saved RSL-RL agent configurations for the bundled Anymal-C flat- and
  rough-terrain control policies.
