# Per-Body Net Contact Forces

Stage 1 adds an opt-in auxiliary prediction to the rough-terrain Anymal-C
`shared_per_body` model. At each predicted step, the solver exposes
`predicted_body_net_contact_forces_b` with shape `[num_envs, 17, 3]`;
one `[17, 3]` value is produced per environment. The 17 rows follow the runtime
primary-body order. Each row is the three-dimensional resultant linear contact
force [N] on one primary body, including both normal and tangential/friction
components. Contact moments and torques are not included.

Dataset labels are stored as `body_net_contact_forces_w` in the world
frame. They are sampled after the Newton solve, so label index `t` describes
the contact force for the solved transition from state `t` to state `t+1`.
The `body_net_contact_force_body_names` HDF5 attribute stores the primary body
names in the exact order of the label tensor's body axis.
Training rotates these labels into the current root-body anchor frame. The
model output and the runtime solver cache therefore use body-frame axes and the
`_b` suffix.

Force collection is opt-in through
`--record-body-net-contact-forces`. The Stage 1 OSMO presets request this
flag; existing state-only presets do not. Existing HDF5 datasets do not contain
the required force tensor and metadata, so they cannot train a force-enabled
config. The force presets use separate dataset directories and OSMO cache
subdirectories to prevent accidental reuse.

The auxiliary head does not feed forces back into the simulation. Its loss
weight and active-contact metric threshold are configured independently:

```yaml
algorithm:
  loss:
    body_net_contact_force:
      weight: 0.1
      active_contact_threshold_n: 1.0
network:
  body_net_contact_force_head:
    enabled: True
```

The force head is a global MLP over the temporal trunk feature. It jointly
predicts all 17 three-dimensional body forces as a 51-dimensional vector,
which is reshaped to `[17, 3]`. The force target is normalized with its own
running statistics and trained with normalized MSE.

## Negative ablations

Two alternative objectives and decoders were evaluated on the same 10M
rough-terrain dataset and were not retained:

- Active/inactive balanced force loss did not improve policy-trajectory force
  metrics over normalized MSE.
- A shared per-body force decoder conditioned on each body's contact latent and
  a learned body embedding did not outperform the global decoder. At epoch 359,
  its policy-trajectory force RMSE was `13.59 N` (`32.91 N` on active bodies),
  compared with `12.63 N` (`31.31 N` active) for the global decoder. Its
  zero-action RMSE was `11.25 N` (`20.45 N` active), compared with `9.80 N`
  (`17.55 N` active) for the global decoder. The experimental OSMO workflow was
  `anymal-c-rough-shared-per-body-contact-force-formal-20260824-051930-1`.

The retained configuration is therefore the global force decoder with
normalized MSE.

## Smoke commands

Run the complete rough Anymal smoke flow locally:

```bash
./isaaclab.sh -p osmo_scripts/run_experiment.py \
  --preset anymal_rough_newton_native_shared_per_body_contact_force_smoke \
  --workflow-base-name anymal-c-rough-shared-per-body-contact-force-smoke \
  --dataset-subdir anymal-c-rough-contact-tokens-contact-force-smoke-v1 \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/anymal-c-rough-contact-force-smoke \
  --dataset-dir ./data/datasets
```

Submit the matching smoke flow to OSMO:

```bash
./osmo_scripts/start.sh \
  --preset anymal_rough_newton_native_shared_per_body_contact_force_smoke \
  --dataset-cache-mode off \
  --pool <osmo-pool>
```

Use the
`anymal_rough_newton_native_shared_per_body_contact_force_formal` preset for the
10M-transition formal recipe.

## Policy rollout validation

Validate a force-enabled checkpoint while a trained RSL-RL policy drives Anymal
over ground-truth Newton rough terrain:

```bash
PYTHONPATH=./source/isaaclab_neural ./isaaclab.sh -p \
  -m isaaclab_neural.eval.play_anymal_rough_contact_forces \
  --nerd-checkpoint ./data/trained_models/<run>/nn/final_model.pt \
  --policy-checkpoint ./logs/rsl_rl/anymal_c_rough/<run>/model_1499.pt \
  --num-steps 500 \
  --video \
  --visualizer newton
```

The script intentionally runs NeRD as a shadow observer instead of using it as
the environment physics solver. This preserves post-solve Newton
`ContactSensor` forces as ground truth. With the Newton visualizer enabled, red
arrows show sensor forces and blue arrows show neural estimates. At the end of
the rollout, the script plots the ContactSensor truth and neural estimate force
magnitudes together for each of the 17 primary bodies and writes the figure to
`videos/contact_force_validation/contact_sensor_forces.png` by default. Use
`--force-plot-path`, `--marker-force-scale`, `--marker-threshold`, and
`--plot-env-id` to adjust the output. Four additional files with suffixes
`_04_LF_FOOT`, `_08_LH_FOOT`, `_12_RF_FOOT`, and `_16_RH_FOOT` provide
larger truth-versus-estimate plots for the individual feet. With `--video`,
the Newton-viewer rollout is recorded to
`videos/contact_force_validation/contact_force_validation.mp4`; use
`--video-path` and `--video-fps` to override the output.

The final console summary reports `body_force_RMSE_N`,
`body_force_active_RMSE_N`, and `body_force_active_fraction` using the same
vector-error definitions as training. By default, samples from the first
`num_states_history - 1` physics steps are excluded while transformer history
fills. Use `--warmup-steps` to override this interval and
`--active-force-threshold` to change the active-contact threshold.

To validate under uniformly sampled actions without loading an RSL-RL policy:

```bash
PYTHONPATH=./source/isaaclab_neural ./isaaclab.sh -p \
  -m isaaclab_neural.eval.play_anymal_rough_contact_forces \
  --action-mode random \
  --nerd-checkpoint ./data/trained_models/<run>/nn/final_model.pt \
  --num-steps 500 \
  --visualizer newton
```

Full contact-wrench prediction, including contact moments and torques, is
deferred to Stage 2.
