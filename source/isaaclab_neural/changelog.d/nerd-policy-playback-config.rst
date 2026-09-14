Fixed
^^^^^

* Restored the policy checkpoint's adjacent ``params/agent.yaml`` in NeRD
  RSL-RL playback before constructing the runner, preserving trained network
  structure, action clipping, and seed. Honored explicit playback clipping
  and seed overrides and loaded weights onto the playback device. Retained
  registered task defaults when no saved configuration existed and rejected
  malformed saved configuration before simulation initialization. Keep
  ``params/agent.yaml`` alongside copied policy checkpoints; omit explicit
  clipping and seed overrides to reuse their saved values.
