Added
^^^^^

* Added ``NerdSolverCfg.step_mode`` (``"physics_frame"`` default or ``"control_step"``). In control-step mode
  ``NewtonNerdManager.handles_decimation()`` reports true, so the environment applies actions, steps physics and
  updates the scene once per control step, and contacts, actuator telemetry and the model run once per step.
* Added ``NerdSolverCfg.joint_target_input``: the solver reads Newton ``control.joint_target_q`` and the model
  receives ``joint_target_offset`` (target minus current joint coordinates).
* Added control-step Keyboard recording (``keyboard_dataset collect --record-control-steps``). Each record holds the
  state, body poses and native contacts at the control boundary, the applied effort and joint targets of the first
  physics frame, and the next state, normal forces and key state after the last frame. Records carry
  ``step_mode="control_step"`` and no physics-frame replay timeline; training rejects mixing step modes.
* Added the ``compact_v1`` native contact-pair precision (``--native-contact-precision``): identifiers are stored as
  int16/int32 and bounded normals, offsets, lever arms and margins as FP16, while positions, separations and
  relative velocities stay FP32. Readers widen values to int64/float32.
* Added ``algorithm.dataset.load_mode: gpu`` (``GpuResidentTrajectoryDataset`` and ``GpuBatchLoader``). Rank-local
  recordings stay on the training GPU, optionally with FP16 fields (``resident_float16_keys``). Native pairs stay raw:
  filter-rejected pairs are dropped when loading and the rest are converted to contact tokens per batch with the
  shared projector, matching the eager loader exactly.
* Added ``NerdSolverCfg.reset_history``. The default ``"replicate"`` keeps the previous behavior: streaming
  models attend over copies of the post-reset inputs. ``"episode_start"`` restarts a reset row's stream at its
  first post-reset frame, matching training windows at episode starts. The control-step recipe uses it; evaluate
  earlier control-step checkpoints with ``keyboard_policy --reset-history episode_start``.
* Added ``inputs.contact_set.mask_all`` to train and run a model with every contact token masked, for
  contact-input ablations, plus the ``SO101/transformer_keyboard_control_step_no_contact.yaml`` recipe and the
  ``so101_keyboard_control_step_no_contact`` preset. Added ``--reset-history`` to ``keyboard_benchmark``.
* Added ``algorithm.precision`` (``fp32``, ``bf16`` or ``fp16`` with loss scaling) for the training forward pass and
  ``algorithm.optimizer.fused`` for Adam.
* Added the ``SO101/transformer_keyboard_control_step.yaml`` recipe and the ``so101_keyboard_control_step`` OSMO
  presets.

Changed
^^^^^^^

* Changed training metrics and gradient norms to stay on the device and transfer once per batch instead of
  synchronizing for every metric.
