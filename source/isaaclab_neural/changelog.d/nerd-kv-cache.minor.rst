Added
^^^^^

* Added opt-in RoPE Transformer training and incremental KV-cache inference
  through :meth:`~isaaclab_neural.models.models.ModelMixedInput.forward_cached`.
* Added per-environment RoPE KV-cache integration to the Transformer NeRD
  solver, including selective cache rebuilds after asynchronous resets.
* Added exact sliding feature-cache inference for existing Transformer
  checkpoints through
  :meth:`~isaaclab_neural.models.models.ModelMixedInput.forward_feature_cached`.

Changed
^^^^^^^

* Changed native contact-token packing to use vectorized segmented scans,
  reusable buffers, and fused Warp geometry transforms for scalable deployment
  inference.
