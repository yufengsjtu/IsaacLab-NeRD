Added
^^^^^

* Added fixed SO101 keyboard dataset collection through the native manager-based
  environment, preserving per-physics-frame states, native contact pairs,
  post-solve robot normal-force labels, and episode boundaries.
* Added reset-safe trajectory clipping and strict open-loop action replay with
  source-world layout, physical state, MDP state, and MJWarp warm-start restoration.
* Added an experimental fixed-keyboard NeRD environment, small Transformer recipe,
  ordered robot normal-force heads, checkpoint contracts, and sensor publication.
* Added world-preserving checkpoint rollouts with physics-step counting and
  separate execution and task-fidelity status. The prototype was not qualified
  for PPO training or simulation speed comparisons.
* Added explicit dynamics-frame timing overrides to training adapters while
  preserving the control-frame fallback for existing environments.
* Added optional physical-unit keyboard and robot position loss scales without
  changing legacy defaults or velocity weights, plus grouped training errors and
  per-frame rollout error curves, key-state confusion counts, and scoring coverage.
* Added native keyboard policy evaluation with explicit reset-IK comparisons,
  completed-step and finite-state checks, episode success reports, and optional video.
* Added fixed-keyboard RSL-RL policy collection with checkpoint provenance and
  scene-preserving replay checks.
* Added an explicit, metadata-preserved deterministic-physics collection option,
  without changing the native task defaults or weakening replay thresholds.

* Added multi-source keyboard contract validation and synchronous-window checkpoint
  evaluation using the shared sequence trainer and weighted trajectory datasets.
* Added hash-bound OSMO input bundles with explicit runtime checks and checkpoint
  diagnostics, counting unique source transitions after overlap merging and
  effective loader truncation.

* Added portable virtual trajectory datasets with rebased native CSR offsets,
  allowing bounded replay-qualified recordings to feed the existing lazy loader
  without duplicating dense tensors or crossing recording time axes.
* Added a policy-collection pipeline with per-chunk acceptance/resume, full
  provenance checks, and source-chunk checkpoint evaluation. All chunk seeds
  were included when checking training/validation independence.
* Added optional headless Newton GL video for full synchronous keyboard replay,
  with a camera positioned relative to the selected recorded world. All source
  worlds remained active, and frames were captured after matched reset restoration.

* Added optional GPU-isolated collection workers with dynamically scheduled seeded
  chunks, retaining eight-world eager simulation and full per-chunk replay checks.
* Added hash-bound partial collection recovery, preserving accepted legacy chunks
  without treating a recovery checkpoint as a complete training cache.

* Added an explicit basic-validation Keyboard collection mode for 1024-world
  collectors without per-chunk physics replay. Its separate bundle
  schema retained finite-value, reset, capacity, provenance, and data-volume checks
  and was rejected by replay-required presets.
* Added indexed CPU contact packing for reset-safe Keyboard clips, preserving
  trajectory/frame/pair ordering without repeatedly scanning every world's pairs.

* Added per-GPU resident Keyboard collectors that reused the environment and teacher
  policy across seeded chunks, with process and device provenance and fail-closed
  basic validation before cache publication or training.
* Added an opt-in, source-checked eager deterministic workspace profile for large
  Keyboard batches without reducing native contact capacities. Added configurable,
  metadata-preserved recording token capacity.
* Recorded and restored derived MJWarp model coefficients at initial and matched
  reset boundaries so warm-worker recordings could replay without correcting
  ongoing episodes. Existing recordings lacking these coefficients retained their
  legacy replay behavior; recollect warm-worker data with the new profile.

* Added explicit ``unique_transition`` capacity with global selection before DDP
  partitioning, a shared source-identity index for eager/lazy readers and collection
  gates, and exact masks for capacities not divisible by clip or window lengths.
  Legacy readers retained ``stored_frame`` capacity by default.
* Added canonical disjoint supervision blocks, retaining reusable sequence context
  while counting each selected target once per indexed traversal. Applied masks to
  state/force/contact-classification losses, metrics and normalization, corrected
  DDP weighting for unequal target counts, and reported actual unique targets seen.
* Added the isolated ``so101_keyboard_resident_unique`` recipe with ten million
  unique training targets and three million unique validation targets. Recorded
  capacity contracts and selection fingerprints in resolved configs/checkpoints.
  Existing accepted recordings remained readable without recollection; enabling
  the new mode required source identities and recomputed normalization rather than
  silently reusing old stored-frame sampling semantics.

Changed
^^^^^^^

* Recorded pipeline stage durations, per-rank dataset initialization/eager preload/RMS
  timings, and separate training/validation epoch timers without changing optimization
  or sampling. Declared loader defaults so eager, pin-memory and preload overrides could
  be resolved through the strict submission configuration parser.

* Streamed deterministic unique-dataset eager preloading into preallocated rank-local
  arrays using ``algorithm.dataset.eager_preload_frames`` (default 4096 frames, with a
  minimum of one trajectory). Reduced temporary native-contact decoding memory without
  changing source files, selected windows, masks or normalization semantics. Random
  contact packing retained its original whole-pool RNG behavior. Eager loading remained
  opt-in; existing lazy recipes required no migration.

* Coalesced overlapping dense training windows while retaining direct reads for scattered
  samples, and skipped geometric contact reads for ranges rejected by the existing filter.
  Preserved original sample order, masks, data contracts and overflow checks.
* Encoded only valid contacts during sparse eager execution, retaining static dense paths
  for compilation, CUDA graph capture and fully occupied inputs. Kept parameter layouts
  compatible with existing checkpoints and retained gradients on empty-contact batches.

* Reduced unique Keyboard training reads to model-required fields and batched deterministic
  contact projection across windows. Reused worker-local HDF5 handles, retaining sample
  order, capacity masks, window-weighted normalization and existing checkpoint semantics.
  Existing fixed-v2 data and checkpoints required no migration or recollection.
* Reused complete bundle verification within one Keyboard pipeline while manifest hashes,
  input file identities, sizes, modification/change times and validation settings remained
  unchanged. Standalone verification and subsequent pipeline runs still performed full checks.

* Added opt-in ``solver_active_movable`` token filtering using recorded/live body
  mobility, retaining movable key-ground contacts while excluding immovable pairs
  and nonnegative-separation candidates. Raw native pairs and physics were unchanged.
* Qualified the aligned resident recipes at 128 filtered tokens with a 72-token
  observed high-water mark. Added explicit collection/replay/filter contracts and
  kept overflow rejection. Old unfiltered files without mobility metadata required
  a qualified recording rather than silent relabeling; retrain/recompute statistics
  when changing the input representation.

* Decoupled unique pool capacity from window sampling. Aligned the resident SO101
  recipes with Franka-style stride-one, multi-position supervision and window-based
  normalization, retaining masks only at capacity boundaries. Kept canonical
  single-owner sampling as an explicit compatibility mode.
* **Breaking:** Restricted SO101 Keyboard NeRD to the fixed keyboard scene (variant 0,
  authored pose). Removed randomized-scene collection, the ``so101_keyboard_original_v1``
  schema, ``IsaacContrib-Keyboard-SO101-NeRD-Original``, the optional
  ``model_context_fields``/``include_articulation_context`` solver inputs and
  ``WorldContextDataset``, and the randomized-scene OSMO presets and training
  configurations. The resident recipes now use owner-frame contact tokens and the
  ``so101-keyboard-fixed-v1`` cache prefix. Recollect data with the fixed collector
  and train a new model; randomized-scene files and checkpoints are rejected.
* Assigned complete source timelines to DDP ranks for sliding sampling, avoiding
  duplicate eager trajectory loads. Used compact window ranges and source-target
  bitsets to distinguish unique coverage from repeated training exposures.

Fixed
^^^^^

* Validated requested checkpoint input/network semantics before merging checkpoint
  configuration, and checked training-pool fingerprints before reusing normalization.
* **Breaking:** Upgraded fixed keyboard recordings to ``so101_keyboard_fixed_v2``
  with authored root poses, zero-reset checks, recorded-pose validation and live
  physical-parameter matching. Used isolated fixed-v2 caches; recollect old fixed-v1
  data and train a fresh model instead of relabeling files or checkpoints.
* Corrected global RMSE and contact precision/recall aggregation, including active
  sample denominators and distributed reductions. Corrected unequal sliding DDP
  shard contributions with window-population importance weights.

* Recorded the configured ``window_sampling`` in resolved unique-capacity training
  configs instead of always recording the canonical index schema.
* Rejected checkpoint resumption when CLI overrides or omitted legacy keys changed
  the model's input/target semantics, and rejected reuse of checkpoint statistics
  under a different sampling contract. Start a fresh model, or pass
  ``--update-dataset-statistics`` when only the sampling contract changed.
* Moved training CLI and NCCL initialization into the parent entry point so spawned
  DataLoader workers did not initialize duplicate distributed ranks.
* Preserved explicitly recomputed loss weights and contact sample counts when
  restoring optimizer/epoch state, and reduced maximum error metrics with MAX
  across batches and ranks instead of averaging them.
* Supported hash-bound basic collection recovery without requiring replay receipts,
  rejected cross-mode recovery, and passed W&B keys through managed credentials
  instead of command arguments or literal workflow environment values.

* Updated dataset and NeRD runtime adapters to use the current Newton backend-owned
  model, state, and control buffers, and retained post-step state publication hooks.
* Accepted the corresponding SO101 USD ``*_link`` force-body names while preserving
  the canonical body order and normal-force label semantics.
* Fixed keyboard replay's source-time mixing by recording and validating the full
  synchronous world timeline, retaining reset-safe training clips separately.
  Recorded reset outcomes were applied only after matching terminal boundaries;
  ongoing episodes were never corrected and termination mismatches remained failures.
  Regenerate old clip-only files to use this protocol; prefix/subset replay remained
  available only for legacy files.
* Selected eager execution for deterministic keyboard recording and preserved it
  in replay metadata, without changing physical capacities.
