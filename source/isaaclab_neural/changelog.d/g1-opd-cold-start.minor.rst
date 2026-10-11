Added
^^^^^

* Added bounded G1 OPD workers for random actor/critic initialization, physics-step
  context collection, isolated MJWarp oracle queries, and matched-input dynamics
  evaluation. These workers remained usable independently of the bounded PPO/OPD pilot.
* Added immutable native-contact context shards with exact short histories,
  anchor-only supervision, length-bucketed batches, and split-family validation.
* Added explicit ``--checkpoint-mode finetune`` and ``--data-dt`` options to
  dynamics training. Fine-tuning preserved checkpoint normalization and loss
  weights while starting fresh optimization progress; the default ``resume``
  behavior was retained. Users must provide a checkpoint with complete statistics
  and must not request statistics recomputation in fine-tune mode.
* Added versioned OPD policy snapshots with explicit zero-update initialization,
  optimizer/adaptive-learning-rate state, and optional RNG restoration.
* Added opt-in fixed-anchor-budget context sampling with explicit source quotas
  and history-length probabilities. Exact-length microbatches accumulated a
  sample-weighted gradient before one clip/optimizer step, preventing short
  batches from implicitly receiving a full-size update budget. Configurations
  without ``context_sampling`` retained their existing batching behavior.
* Added deterministic sampling-exposure audits and per-training-run sampler
  fingerprints for equal-budget learning-rate comparisons.
* Added bounded, non-deploying G1 learnability probes for fixed-batch fitting,
  independent validation, input scaling and PD saturation audits, and paired
  same-state/different-target MJWarp queries with an A/A replay noise control.
* Added diagnostic-only, equal-capacity masked/enabled target-error ablations
  with zero-initialized projection columns, frozen training-only target statistics,
  paired-intervention scoring, and isolated target-history binding for NeRD evaluation.
* Added a bounded synchronous G1 PPO/OPD pilot with persistent environment/history,
  actual-rollout queries, environment-disjoint model validation, cross-stage replay,
  small candidate updates, explicit promotion gates, and independent MJWarp policy evaluation.
* Preserved unconsumed RSL-RL configuration in new pilot snapshots and added explicit,
  provenance-recorded recovery of constructor-consumed metadata for earlier snapshots.
* Added an explicit research-only reference-monitor mode for online OPD. Default
  reference enforcement, current-distribution gates and non-finite rejection were
  retained; reports distinguished accepted candidates from updates used by later PPO blocks.
* Added bounded PPO consistency and minibatch diagnostics for 8/128-environment
  fixed-NeRD runs, preserving native optimizer behavior and RNG while recording
  KL, clipping, gradient norms and intra-update learning rates.
* Added a bounded MJWarp-only G1 baseline sharing the NeRD task's MDP, with
  environment-scoped geometry contacts, matched pre-step timing, configuration
  fingerprints, initial-state audits and checkpoint evaluation. Existing NeRD
  geometry-contact lookup remained the default when no scoped adapter was supplied.
* Added frozen-teacher provenance/training-curve audits and common MJWarp policy
  benchmarks across stock and geometry-contact tasks. First-episode failure,
  timeout and censoring were reported separately, with reward-aligned yaw/world
  tracking frames and actor immutability checks.
* Added immutable action-sequence query/label shards, selected-world rollout
  recording, terminal prefixes, and a reusable MJWarp replay worker that consumed
  recorded targets instead of holding one target throughout the horizon.
* Added complete-current-data epoch sampling with balanced logical tail batches
  and explicit reference/replay quotas, without changing legacy context sampling.
* Extended read-only recording checks to Newton's current position/velocity target
  buffers and recorded native initial state/solver-memory provenance for replay audits.
* Added a fixed-data G1 student probe with state/PD/history gradients, explicitly
  detached contact geometry, isolated scratch state and forward-parity gates.
  One- and four-physics-step fitting probes and sixteen-step forward parity were
  validated, without claiming complete online training or policy improvement.
* Added complete-epoch OPD optimization with weighted microbatch accumulation,
  validation patience, hard budgets, frozen-statistics checks and paired best-epoch
  model/optimizer restoration, without promoting candidates into live PPO.
* Added per-epoch local/sequence loss and physical-error diagnostics, optional
  disjoint reference-validation monitoring, and complete model/Adam fingerprints
  for selected-checkpoint restoration audits without changing the OPD objective.
* Added a bounded MJWarp active-collection probe with a frozen NeRD scorer,
  task-regularized capped prediction-error rewards, post-reset history masks,
  pre-reset real transition labels, and offline/online score consistency audits.
  Collector updates were kept separate from dynamics training and policy promotion.
* Extended active collection to explicit base/error/mixed reward profiles and bounded
  all-world GPU scoring with batched, rotating selected-world recording. Added frozen-policy
  validation collection with real action sequences, source/layout provenance and throughput diagnostics.
* Added equal-data single-step model fitting with per-body-region error attribution,
  fixed physical-input caches, shared protected validation and paired checkpoint audits.
  Finite quality rejection could optionally preserve a fixed optimizer budget without
  selecting rejected epochs; non-finite and frozen-statistics checks remained mandatory.
* Added optional root/leg-only acquisition error while preserving the original full-state
  denominator, action contract and NeRD training objective. Full-state and scoped errors
  remained separately monitored and audited; the default full-state behavior stayed unchanged.
* Added read-only old/new-distribution evaluation with whole-world held-out splits,
  preserved-epoch cross-cohort matrices and retention-constrained adaptation scores.
  Missing original data remained unverified; rehearsal proxies could not certify old-data preservation.
  The historical fixed-policy selector was labelled as a legacy diagnostic rather than a coverage test.
* Added independently reseeded pretraining-recipe retention evaluation with nonoverlapping
  trajectory windows, archived mixture weights and full teacher-forced loss alongside anchor error.
  Resampled recipes remained distinct from original files and rehearsal proxies; fixed-prior-candidate
  audits could not change the locked model selection.
* Added matched-size recipe replay preparation with protected-file/seed checks,
  explicit source quotas and unchanged last-anchor supervision. Enabled fresh new-cohort
  evaluation for newly fitted candidates and predeclared fixed-budget audit endpoints.
* Added a bounded stagewise active-learning coordinator with continued collector PPO state,
  independently refreshed analytic reference data, unique per-stage anchor consumption,
  historical-new replay and retention-gated NeRD scorer/model-Adam continuation.
  Existing active collection retained cold-start behavior unless explicitly resumed.
* Added immutable full-history training caches, step-zero fitting fallback and per-position recipe diagnostics.
  Added recorded-action sequence retention audits with endpoint, leg-velocity and ankle-frame metrics.
* Added observed per-position MJ labels with frame/episode continuity checks and frozen-policy training-data collection.
  Unified coordinator and direct fitting workers on full-history pretrained MSE for both data sources and validation.
  Deprecated supervision switches now accept only the final full-history values or may be omitted;
  anchor-only fitting configurations and incomplete caches are rejected. Generic counterfactual OPD readers
  retained their separate anchor-only interface.
* Added optional reference-only full-history output preservation against frozen initial dynamics,
  retaining both real-label objectives and zero-coefficient behavior with separately reported loss components.
* Added an isolated recorded-joint-force retention worker using the pretrained10-step rollout evaluator
  with fixed held-out windows, explicit scratch-world advancement and checkpoint integrity checks.
* Added evaluation-only stock-MJ RL-stage collection with explicit saved-versus-untrained policy provenance.
  Added reset-safe train/validation sequence exports with optional observed future inputs and recorded-force trajectories.
* Added hashed RL coverage evaluation and training-only4/16-step forward/backward profiling with temporal-gradient,
  native-forward and frozen-model checks, without optimizer updates.
* Added bounded direct-study state-feature denoising, explicit old-truth weighting and separately scaled real-action
  rollout objectives, retaining clean reference distillation and unchanged defaults.
* Added training-only objective calibration, immutable reference-cache reuse and deferred audit handling with
  explicit teacher-forced prefilter scope and locked-model audit evaluation.
* Added opt-in paired epoch replay and independently budgeted preservation-input mixtures with frozen-teacher,
  physical-protocol and input-only provenance checks. Fresh unique-row sampling remained the default;
  teacher-generated histories were rejected as truth supervision. Added explicit same-recipe-only controls
  and per-step optimizer-progress reporting without changing the original loss operator or evaluation weights.
* Added separately budgeted matched-geometry MJWarp/NeRD policy studies with cold-start snapshot checks,
  frozen-dynamics validation, reward-frame tracking and first-episode deployment metrics. Existing small-pilot
  limits remained unchanged; extended runs required explicit configurations and smoke/profile validation.
  Added opt-in per-frame feature-cache audits and explicit evaluation transition counts without changing inference defaults.
* Added explicitly bounded fixed-data training-strength studies with fresh-Adam checks, sparse predeclared checkpoints,
  parameter-displacement diagnostics and non-destructive storage guards. Existing pilot limits and losses were retained.
* Added non-mutating oracle-query screening masks for root height, quaternion norm, joint limits and signed
  native-contact penetration. Required explicit joint-limit tolerance calibration for soft backend constraints
  and preserved rejected-state diagnostics without silently clipping query inputs.
* Added yaw/translation-invariant coverage features with explicit training-reference normalization semantics.
* Added an opt-in runtime-matched FP32 PD mode for sequence students, preserving recorded anchor effort and
  retaining the legacy Torch mode by default. Paired replay required the recording's numerical batch layout.
* Added bounded independent NeRD collector workers with frozen dynamics, persistent MJ sequence queries,
  typed future-only labels and delayed positive-optimism/coverage rewards before PPO return computation.
  Task-policy evaluation data remained excluded from training, and sparse snapshots preserved storage budgets.
* Added opt-in independent collector sequence objectives with provenance guards, real future-only labels,
  prefix-aware gradient accumulation and separately scaled root-rollout correction. The default full-history
  objective was retained, and auxiliary coefficients could be calibrated using training-only gradients.

Fixed
^^^^^

* Moved collector advantage diagnostics after return computation, preserving the policy update while
  preventing stale pre-return advantage statistics from appearing in the consistency report.
* Compacted recorded sequence tensors instead of deep-copying views with full multi-world backing storage.
  Values and replay contracts were preserved while unrelated-world storage was excluded from each record.
  Added optional memory-mapped reading for immutable legacy sequence files without changing default loading.
* Separated atomic mutable progress updates from immutable artifact writes, preserving overwrite protection
  for checkpoints and results while allowing fitting and policy-study progress to advance.

* Replaced unordered CUDA atomic contact pooling with batched matrix reduction,
  preserving per-body masked-mean/count semantics and checkpoint parameter layouts.
  Numerical baselines collected with the old reduction should be remeasured.
* Preserved CUDA default-stream identity when NeRD bridged Warp stream zero to
  Torch, avoiding an unintended new stream and unordered native-reference execution.
* Replayed exported active-collection scores in their recorded online batch layouts,
  avoiding false mismatches from comparing different FP32 execution layouts without relaxing audit tolerances.
