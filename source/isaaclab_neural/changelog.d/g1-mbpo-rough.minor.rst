Added
^^^^^

* Added portable manager-based G1 rough-terrain MBPO workers initialized from a
  flat contact-token dynamics checkpoint, with independent rough-policy initialization,
  persistent online model fitting, real-transition PPO mixing, and immutable input provenance.
* Added terrain-aware branch buffers, fixed-atlas and stock-curriculum profiles,
  per-world material/inertial context, event-clock restoration and push-crossing label checks,
  disjoint model holdout worlds, and a coupled smoke before formal training.
* Added an aligned rough preset with 4096 model worlds, fixed 24-control-step branches,
  bounded model warmup, the default rough PPO runner, stock physical events, geometry-based
  feet-slide rewards and stock force-contact MJ evaluation.
* Added an OSMO rough-MBPO preset with bounded single-GPU resources, explicit submission
  confirmation, W&B credential injection, grouped metrics, and held-out-terrain MJ evaluation.
  Inputs and outputs used NV-Datasets through ``nvdataset_cred``, with new immutable input
  datasets, digest verification and an isolated SDK environment.
* Added opt-in rough-terrain selection to G1 workers while preserving the flat defaults.
  Rough workers required a shared scene configuration and did not use the flat-height guard.
