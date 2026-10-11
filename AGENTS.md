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
- RSL-RL runner construction consumes nested config fields with `pop` (algorithm/model/distribution class names). Pass a deep copy to `OnPolicyRunner` and retain an untouched config for reproducible snapshots and evaluation.
- RSL-RL adaptive PPO changes LR per minibatch. A per-update LR at the floor is only the terminal value, not proof that the whole update used that LR; inspect minibatch KL/clip/LR traces before diagnosing stalled learning.
- Verify G1 HDF5 state/action dimensions and provenance against the checkpoint before treating matching filenames as original training data. Legacy 23-joint/59-state files cannot certify retention for a 37-joint/87-state model; missing originals must remain explicitly unverified, not padded or replaced by a rehearsal proxy.
- RSL-RL `model_0.pt` may already contain one PPO iteration (inspect Adam steps); label a newly constructed untrained actor separately rather than claiming it reproduces the historical initialization.
- Stock MJWarp benchmarks must retain contact sensors/terminations. Do not disable sensors merely to enable run-to-run determinism; preserve the stock sensor mode and use frozen recorded data for paired model evaluations.
- G1 MBPO OSMO storage uses NV-Datasets (`nvdataset_cred`), matching the repository launchers. Do not reintroduce S3/Backblaze URL inputs; a B2 run on 2026-10-06 failed with HEAD-object 403 while bucket preflight passed. Submit from the isolated SDK environment pinned by `osmo_scripts/nvdataset_requirements.in`; the task installs the SDK into `/tmp/nvdataset-venv`, not the simulator venv.
- Validate OSMO per-GPU resource limits before large uploads. The `isaac-lab-l40s-03` pool rejected 16 CPUs for one GPU on its 127-CPU/eight-GPU nodes; the G1 rough MBPO preset requests 15 CPUs. Adjust job resources, not cluster policies.
- G1 `active-stage-fit` workers accept only 4 or 16 scratch worlds. MBPO smoke fitting uses 4 worlds and formal fitting uses 16; keep parent-preset validation consistent with the worker argument contract.
- G1's raw 87-dimensional NeRD state is 44 position coordinates followed by 43 velocity coordinates. Root linear velocity starts at index 44, unlike the 13-dimensional articulation root-state layout where it starts at 7.
- Packed NeRD contact tokens keep only solver-active pairs (separation < 0). A positive geometry `maximum_separation` must read native contact candidates (`body_min_separation`), otherwise it is silently ignored. G1 MBPO gait rewards (`feet_air_time`, `feet_slide`) default to `feet_contact_separation=0.0025` m for flat (`logs/g1_mbpo/launch_mbpo.py`) and rough (`mbpo_train.py`, preset); the torso termination stays strict. A 0 m criterion caused millimetre contact chatter in NeRD (24% false double flight, air-time correlation with MJ 0.17).
- An unattended NVIDIA userspace upgrade without reboot breaks headless Newton GL rendering (`No standard config is available`) and `nvidia-smi` while CUDA compute still works; reboot before recording videos.
- Newton's deterministic contact packing has a 20-bit contact-ID limit (1,048,575 buffered contacts). Rough MBPO fitting inherits stock collision options and keeps the 2,500,000 triangle-pair capacity; do not silently shrink capacity to force deterministic packing. Legacy flat sequence studies retain their deterministic default.

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
