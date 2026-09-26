# IsaacLab Development Guidelines
- Make the smallest change that solves the requested problem.
- Preserve existing behavior unless the task explicitly changes it.
- Read surrounding code, callers, tests, and documentation before changing an interface.
- Do not add dependencies unless necessary. Prefer existing project dependencies and the standard library.
- Do not commit generated plans, scratch files, or agent artifacts.
- Follow a more-specific `AGENTS.md` in the directory being changed.
- Use the repository's current SPDX header template for new source files; do not change existing file headers.
- Follow the existing style and abstractions in the affected package.
- Use modern Python type hints, including `X | None` instead of `Optional[X]`.
- Use `snake_case` for methods, functions, and CLI arguments.
- Keep related public symbols discoverable through consistent prefixes.
- Use concrete types for public interfaces where practical.
- Use Google-style docstrings for public APIs.
- Document SI units for public physical quantities in docstrings using inline `[unit]` notation (e.g. `Particle positions [m], shape [N, 3]`); use `[m or rad, depending on joint type]` where applicable, and skip non-physical fields (indices, counts, flags).
- Keep comments brief and explain intent, constraints, or edge cases rather than restating code.
- Do not remove or rename a public API without a prior deprecation and migration path.
- Update public documentation when adding or changing public APIs.
- Verify documented technical claims against the current code and primary sources before relying on them.
- Use the uv-managed environment for routine commands.
- Use `uv run python` for Python scripts and tests.
- Use `uv run isaaclab` for Isaac Lab CLI commands.
- Use `./isaaclab.sh` only for installer workflows that require it.
- Do not define Warp kernels in `python -c`; write a temporary Python file instead so Warp can inspect the source.

## Testing and validation

- Run the narrowest relevant test first.
- Before reporting that a test cannot run because an optional dependency is missing, identify its project extra and retry with `uv run --extra <extra> ...`.
- Do not treat an optional dependency missing from the base environment as a blocker when its project extra is available.
- Run `uv run isaaclab -f` before committing.
- For a regression test, verify that it fails without the fix and passes with it.
- Find and extend the closest existing test before creating a new test file or test case.
- Add a test only when it covers a distinct behavior, regression, boundary, or failure mode that existing tests do not cover clearly.
- Test observable behavior and public contracts, not implementation details.
- Use hard-coded values only when they are the intended contract or a small, independently verified example; otherwise derive the expected result from a separate, simple reference calculation.
- Keep tests focused and remove or consolidate redundant coverage instead of growing overlapping test suites.
- Do not add debug output to production Warp kernels. Use temporary standalone reproductions and remove debug output before committing.
- `isaaclab_neural` is not installed by the root uv project. Prefix NeRD tests with `PYTHONPATH="$PWD/source/isaaclab_neural:$PWD/source/isaaclab${PYTHONPATH:+:$PYTHONPATH}"` so tests load the current worktree, not another checkout.
- When reusing an existing uv environment across worktrees, set `UV_PROJECT_ENVIRONMENT` to its absolute path and use `uv run --no-sync`; install the `test` extra first. Use `--frozen` when syncing extras without changing the lockfile.
- After merging a PyTorch CUDA 13 upgrade, verify that `cuda-toolkit` extras in `uv.lock` include Linux x86_64, not only aarch64 carried over from an older lockfile. A successful frozen sync can still omit CUDA runtime/NVRTC when those markers are stale. Run `uv lock --check`, then validate `import torch` and a CUDA tensor operation before task imports or training.
- For OSMO GPU numerical failures, record the physical node, GPU UUID and PCI bus ID. Compare the same captured inputs and weights across devices, including UUID-based remapping to logical `cuda:0`; matching GPU model names alone does not establish equivalent hardware health.
- For SO101 Keyboard initialization OOMs, inspect resolved MJWarp capacities rather than only task `njmax`/`nconmax`. The local `mujoco-warp==3.12.0` initialization was observed to allocate `Data.cJ` before replacing it with an empty array when compact solving was disabled. With the Nucleus SO101 asset, 4096 worlds requested `(4096, 39584, 128)` float32 entries (83,013,664,768 bytes) despite sparse solving; verify this allocation path before reducing physical contact capacity or sizing hardware.
- Historical SO101 capacity behavior is version-specific: the original task (`135cf98904`) already configured 4096 worlds and capacities of 600 on Newton 1.5.x / MJWarp 3.11.0. Newton 1.6.0rc1 force-increased explicit capacities; upstream Newton #4172 (`f4afe879` on the release branch, included in v1.6.0) restored honoring explicit values. MJWarp's separate `7361b8c` field-specification change introduced full-size generic initialization of compact workspaces. Check both mechanisms rather than treating the rc1 capacity-floor behavior as universal.

## Keyboard replay validation

- SO101 Keyboard NeRD supports only the fixed scene (variant 0, authored keyboard pose). Collection/replay lives in `isaaclab_neural.generate.keyboard_dataset`; use `--collection-mode policy --checkpoint PATH` for policy data. The randomized 32-variant scene, `so101_keyboard_original_v1` schema, `IsaacContrib-Keyboard-SO101-NeRD-Original` and solver model/articulation conditioning were removed; recollect instead of relabeling old files or checkpoints.
- `--deterministic-physics` is the collector's explicit Newton physics request, distinct from the launcher's rendering-related `--deterministic`. With Newton 1.6.0, mujoco-warp 3.12.0 and warp-lang 1.17.0, graph-captured determinism failed during Warp counter-scan/sort-reduce workspace allocation even at one world (5,086,123,519 uint8 elements). Eager execution avoids capture-time worst-case workspaces, but 32-world eager execution still exceeds a deterministic scatter-record int32 capacity. Eight-world deterministic eager shards were verified; the collector now selects eager execution for this option and records it in metadata. Do not lower physical contact capacities or relax replay thresholds to hide failures.
- Strict keyboard replay must preserve the complete synchronous world timeline, not just world slots. New datasets retain `/replay/frames` and `/replay/contexts` separately from reset-safe training clips. Restore recorded reset outcomes only after matching actual/reference terminal boundaries; never correct ongoing episodes. Legacy clip-only data must be recollected for this protocol.

- The opt-in `keyboard_determinism` eager profile supports the audited MJWarp 3.12 / Warp 1.17 sources at 1024 worlds by bounding deterministic scratch records per kernel and active sparse row; it does not lower native capacities. It rejects unrecognized source fingerprints. Keep legacy eight-world replay presets separate.
- Resident Keyboard replay also requires `actuator_acc0`, `body_invweight0`, `dof_invweight0`, `jnt_solref`, and `stat.meaninertia`: cold replay can otherwise diverge from a warm worker despite identical state, control and contact inputs. Record the coefficients after pending model updates have reached the first solve of each control interval; restore them only initially or for matched reset worlds, after flushing pending model changes.
- The resident preset aligns collection and training at 512 contact tokens. A 1024-world warm-worker sample exceeded 256 tokens outside the retained training clips; do not ignore cumulative packing failures. `overflow_frames` is a last-frame gauge, not a cumulative counter, so it must not be differenced across chunks.
- Submission tools can use an isolated SDK environment without modifying the simulator environment: `uv run --no-project --python 3.12 --with-requirements osmo_scripts/nvdataset_requirements.in --index https://artifactory.pdx.nvidia.com/artifactory/api/pypi/sw-ngc-data-platform-pypi-local/simple ...`. The default Miniconda Python may still have unsupported NV-Datasets 0.93.

## Unique-transition capacity

- `so101_keyboard_resident_unique` uses an isolated cache/workflow and `algorithm.dataset.capacity_unit: unique_transition`. Its global `max_capacity` counts selected recording/world/frame targets, not raw clip slots or optimizer visits. Legacy presets retain `stored_frame` behavior; overlapping raw clips need a larger raw cap than their unique minimum.
- `data/transition_index.py` shares unique pool selection with OSMO data-volume gates. `window_sampling: sliding` is independent of capacity: enumerate stride-one windows, deduplicate identical source windows, and keep repeated multi-position supervision. Only capacity-boundary context is masked. `canonical` retains the old single-owner objective for compatibility.
- The SO101 recipes use owner-frame tokens, 128-token `solver_active_movable` input filtering, no model/articulation context, and lazy loading. The filter retains movable key-ground contacts, derives fixed slots from joint ancestry/kinematic flags, and requires matching recorded mobility metadata. Native physics and raw pairs remain unchanged; the qualified three-chunk 1024-world probe peaked at 72 tokens, not a guarantee for all seeds. Sliding DDP owns complete recording/world timelines per rank, avoiding duplicate eager row loads. RMS counts follow window exposures; `unique_training_targets_seen` uses source-ID bitsets. Do not confuse either count with unique pool capacity.
- Unique-mode state/force/BCE losses, normalization and metrics respect the capacity mask. DDP scales local means by supervised-target counts and keeps partial batches. Do not silently reuse owner/canonical checkpoint statistics after switching the sampling or coordinate contract.
- Verify the actual submit-time `lib.preset.load_preset` parser, not only PyYAML. The lightweight parser does not support folded `>-` override strings; keep preset `train_cfg_overrides` on one line.

## Changelog and release metadata

- Do not edit `CHANGELOG.rst` or `config/extension.toml` directly.
- Add one changelog fragment for each changed source package when the change is user-visible.
- Use `.skip` fragments for changes that do not require a release note.
- Write changelog entries in past tense and include migration guidance for deprecated, changed, or removed behavior.
- Mark breaking changes clearly and provide migration guidance.

## Commits and branches

- Work on a feature branch; do not commit directly to `main`.
- Keep commits focused and atomic.
- Use an imperative, capitalized commit subject with no trailing period.
- Inspect staged changes before committing.
- Do not add AI co-author or attribution lines.
- Prefer follow-up commits over amending commits while addressing review feedback.

## Repository skills

- Keep repository-owned skills in `skills/`; do not duplicate their contents in tool-specific discovery directories.
- Validate skill changes with `uv run --no-project python tools/skills/cli.py check`.
- Keep skills concise and point to maintained documentation and source examples.

## Schema fragment consumers

- When migrating a spawner schema slot to fragments, migrate its readers and overrides too.
  Use a bare fragment for a single instance and a list only for multiple fragments. For lists,
  select the owning fragment before accessing fields or calling `replace()`; `fix_root_link`
  belongs on the spawner.
- For file-spawned fixtures that must only tune existing physics bodies, use explicit fragment
  target mappings. A bare fragment or list may create a missing body and change the fixture's validity.

## OSMO operational checks

- Keep YAML environment values such as `'off'` quoted: unquoted `off` can become boolean `False` and fail the runtime CLI. Parse rendered specs before submission. In `isaac-lab-l40s-03`, a one-GPU probe with 128Gi RAM exceeded proportional node limits; the validated probe requests 64Gi instead.
- The explicit basic-validation Keyboard preset collected 1024 non-deterministic worlds on an L40S without per-chunk physics replay. It uses a separate basic bundle schema, not replay acceptance. CPU contact packing must index world pairs once per recorded frame; repeatedly masking all worlds for each reset-safe clip was the measured large-batch bottleneck.
- `osmo workflow exec` requires PTY-backed standard streams, including when launched programmatically; plain subprocess pipes can fail with `Inappropriate ioctl for device`. Its `--entry` is transported in a request URI, so large inline payloads can fail with HTTP 414. Keep commands small and verify remotely staged file hashes before executing them; a successful rsync client exit alone is not proof the expected files arrived.
- Parallel Keyboard collection schedules independent collect/replay pairs with per-child GPU UUID masks. Keep eight worlds per process and unchanged deterministic eager settings; do not mutate the parent environment to assign GPUs across threads.
- A `keyboard_collection_resume.json` checkpoint preserves only accepted source chunks and is not a complete training cache. Verify the remote readback and any final delta before cancelling the producer; recovery must still pass the full data-volume and replay gates before training.
- NV-Datasets 0.95 supports the per-process `DISABLE_RUST_CLIENT=1` transport flag. On 2026-09-25 a local Rust code-upload PUT timed out; after confirming the dataset had no registered files and no workflow had been submitted, a Python-transport upload succeeded. This is not an authentication workaround: retain TLS and integrity checks, and verify state before retrying ambiguous writes.
