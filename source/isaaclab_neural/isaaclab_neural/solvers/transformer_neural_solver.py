# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast

import torch
from newton import Contacts, State

from .neural_solver import NeuralSolver

if TYPE_CHECKING:
    from isaaclab_neural.models.model_transformer import GPT

    class _FeatureCacheModel(Protocol):
        def forward_feature_cached(
            self,
            input_dict: dict[str, torch.Tensor | None],
            feature_cache: torch.Tensor | None,
            rebuild_batch_indices: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

    class _KVCacheModel(Protocol):
        def forward_cached(
            self,
            input_dict: dict[str, torch.Tensor | None],
            cache: GPT.KVCache | None,
            *,
            max_cache_length: int,
        ) -> tuple[torch.Tensor, GPT.KVCache]: ...

        def reset_transformer_cache(self, cache: GPT.KVCache, batch_indices: torch.Tensor) -> None: ...


class TransformerNeuralSolver(NeuralSolver):
    """Neural solver variant that feeds a fixed history window to transformer models."""

    def __init__(
        self,
        num_states_history: int = 1,
        use_feature_cache: bool = False,
        use_kv_cache: bool = False,
        use_frame_cache: bool = False,
        **kwargs,
    ):
        if sum((use_feature_cache, use_kv_cache, use_frame_cache)) > 1:
            raise ValueError("use_feature_cache, use_kv_cache and use_frame_cache are mutually exclusive.")
        self.num_states_history = num_states_history
        self.use_feature_cache = use_feature_cache
        self.use_kv_cache = use_kv_cache
        self.use_frame_cache = use_frame_cache
        self.feature_cache: torch.Tensor | None = None
        self.feature_cache_rebuild_env_ids: torch.Tensor | None = None
        self.kv_cache: GPT.KVCache | None = None
        self.kv_cache_rebuild_env_ids: torch.Tensor | None = None
        self.kv_cache_rebuild_inputs: dict[str, torch.Tensor | None] | None = None
        self.frame_cache_rebuild_env_ids: torch.Tensor | None = None
        super().__init__(**kwargs)
        if use_frame_cache and not (self.anchor_frame_step == "every" or self.states_frame == "world"):
            raise ValueError("use_frame_cache requires per-frame anchoring (anchor_frame_step='every').")
        self.reset_states_history()

    def reset_states_history(self):
        self.states_history = deque(maxlen=self.num_states_history)
        self.feature_cache = None
        self.feature_cache_rebuild_env_ids = None
        self.kv_cache = None
        self.kv_cache_rebuild_env_ids = None
        self.kv_cache_rebuild_inputs = None
        # Per-frame (trunk tokens, per-key features or None), aligned with ``states_history``.
        self.frame_cache: deque[tuple[torch.Tensor, torch.Tensor | None]] = deque(maxlen=self.num_states_history)
        self.frame_cache_rebuild_env_ids = None
        self.stream_cache = None

    def preload_states_history(self, history: Mapping[str, torch.Tensor]) -> None:
        """Replace solver history with validated raw dataset inputs.

        Args:
            history: Raw model inputs with shape ``[num_envs, history, features]``.
        """
        required = {"root_body_q", "states", "joint_f", "gravity_dir", *self.contact_inputs.keys()}
        if getattr(self, "contact_token_frame", "root") == "owner":
            required.add("primary_body_q")
        missing = sorted(required - set(history))
        if missing:
            raise ValueError(f"History is missing required model inputs: {missing}.")

        states = history["states"]
        if states.ndim != 3 or states.shape[0] != self.num_envs:
            raise ValueError(
                f"History states must have shape [num_envs, history, state_dim], got {tuple(states.shape)}."
            )
        history_length = states.shape[1]
        if history_length == 0 or history_length > self.num_states_history:
            raise ValueError(f"History length must be between 1 and {self.num_states_history}, got {history_length}.")

        live_inputs = {
            "root_body_q": self.root_body_q,
            "states": self.states,
            "joint_f": self.joint_f,
            "gravity_dir": self.gravity_dir,
            **self.contact_inputs,
        }
        if getattr(self, "contact_token_frame", "root") == "owner":
            if self.primary_body_q is None:
                raise ValueError("contact_token_frame='owner' requires primary_body_q on the solver.")
            live_inputs["primary_body_q"] = self.primary_body_q
        normalized_history: dict[str, torch.Tensor] = {}
        for key in required:
            value = history[key]
            expected = live_inputs[key]
            if expected.ndim == 1 and value.ndim == 3 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            expected_shape = (self.num_envs, history_length, *expected.shape[1:])
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"History {key} must have shape {expected_shape}, got {tuple(value.shape)}.")
            normalized_history[key] = value

        self.reset_states_history()
        for step in range(history_length):
            entry = {
                key: value[:, step].to(device=self.torch_device).clone() for key, value in normalized_history.items()
            }
            entry["states_embedding"] = self.embed_states(entry["states"])
            self.states_history.append(entry)
        if self.use_kv_cache:
            self.kv_cache_rebuild_env_ids = torch.arange(self.num_envs, device=self.torch_device)

    def reset(self, env_ids=None, world_mask=None, flags=None):
        """Reset transformer history globally or for selected environments.

        A full reset (``env_ids is None`` or every environment) drops the
        history window so later steps refill it the same way as episode start.
        A partial reset copies the current post-reset solver inputs into the
        selected rows. Zero-filling those rows previously fed the model a
        window of invalid zero poses and caused immediate re-falls.

        Isaac Lab 3's Newton manager also calls this as
        ``reset(state, world_mask=..., flags=0)``. That path has no history to
        clear; history is reset with ``env_ids``.

        Args:
            env_ids: Environment indices to reset. ``None`` resets every
                environment. Newton-style calls pass the Newton ``State`` here.
            world_mask: Unused Newton SolverBase reset mask.
            flags: Unused Newton SolverBase reset flags.
        """
        if world_mask is not None or flags is not None:
            super().reset(env_ids, world_mask=world_mask, flags=flags)
            return
        super().reset(env_ids)
        if env_ids is None:
            self.reset_states_history()
            return

        env_ids = torch.unique(self._normalize_env_ids(env_ids))
        if env_ids.numel() == 0:
            return
        all_env_ids = torch.arange(self.num_envs, device=env_ids.device, dtype=env_ids.dtype)
        if torch.equal(env_ids, all_env_ids):
            self.reset_states_history()
            return
        if len(self.states_history) == 0:
            return

        self._fill_history_rows_from_current_state(env_ids)
        self._mark_history_rows_for_rebuild(env_ids)

    def _fill_history_rows_from_current_state(self, env_ids: torch.Tensor) -> None:
        """Replace selected history rows with the current solver snapshot."""
        snapshot = self._history_snapshot()
        for history_entry in self.states_history:
            for key, value in history_entry.items():
                source = snapshot.get(key)
                if source is None:
                    continue
                value.index_copy_(0, env_ids, source.index_select(0, env_ids).to(dtype=value.dtype))

    def _mark_history_rows_for_rebuild(self, env_ids: torch.Tensor) -> None:
        """Rebuild cached transformer features for the selected environments."""
        if self.feature_cache_rebuild_env_ids is None:
            self.feature_cache_rebuild_env_ids = env_ids
        else:
            self.feature_cache_rebuild_env_ids = torch.unique(torch.cat((self.feature_cache_rebuild_env_ids, env_ids)))
        if self.kv_cache is not None:
            self._require_kv_cache_model().reset_transformer_cache(self.kv_cache, env_ids)
        if self.kv_cache_rebuild_env_ids is None:
            self.kv_cache_rebuild_env_ids = env_ids
        else:
            self.kv_cache_rebuild_env_ids = torch.unique(torch.cat((self.kv_cache_rebuild_env_ids, env_ids)))
        frame_rebuild = getattr(self, "frame_cache_rebuild_env_ids", None)
        self.frame_cache_rebuild_env_ids = (
            env_ids if frame_rebuild is None else torch.unique(torch.cat((frame_rebuild, env_ids)))
        )

    def _normalize_env_ids(self, env_ids) -> torch.Tensor:
        """Return env ids as a 1-D tensor on the solver torch device."""
        if isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.torch_device, dtype=torch.long)
        else:
            normalized = torch.as_tensor(env_ids, device=self.torch_device, dtype=torch.long)
        return normalized.reshape(-1)

    def sync_from_newton(
        self, newton_states: State, contacts: Contacts | None, joint_f, *, update_history: bool = True
    ) -> None:
        """Synchronize cached inputs, optionally without appending to history."""
        if update_history:
            self._update_states(newton_states, contacts, joint_f)
        else:
            NeuralSolver._update_states(self, newton_states, contacts, joint_f)

    def _update_states(self, newton_states: State, contacts: Contacts | None, joint_f):
        super()._update_states(newton_states, contacts, joint_f)
        self.states_history.append(self._history_snapshot())

    def _history_snapshot(self) -> dict[str, torch.Tensor]:
        """Clone mutable solver inputs before storing one online history frame."""
        snapshot = {
            "root_body_q": self.root_body_q.clone(),
            "states": self.states.clone(),
            "states_embedding": self.states_embedding.clone(),
            "joint_f": self.joint_f.clone(),
            "gravity_dir": self.gravity_dir.clone(),
            **{key: value.clone() for key, value in self.contact_inputs.items()},
        }
        if getattr(self, "contact_token_frame", "root") == "owner":
            if self.primary_body_q is None:
                raise ValueError("contact_token_frame='owner' requires primary_body_q on the solver.")
            snapshot["primary_body_q"] = self.primary_body_q.clone()
        return snapshot

    def get_neural_model_inputs(self):
        if len(self.states_history) == 0:  # for dummy call
            model_inputs = {
                "root_body_q": torch.zeros_like(self.root_body_q),
                "states": torch.zeros_like(self.states),
                "states_embedding": torch.zeros_like(self.states_embedding),
                "joint_f": torch.zeros_like(self.joint_f),
                "gravity_dir": torch.zeros_like(self.gravity_dir),
                **{key: torch.zeros_like(value) for key, value in self.contact_inputs.items()},
            }
            return {key: value.unsqueeze(1) for key, value in model_inputs.items()}

        history = tuple(self.states_history)
        if getattr(self, "use_frame_cache", False):
            return self.process_neural_model_inputs({key: value.unsqueeze(1) for key, value in history[-1].items()})
        if self.use_kv_cache:
            latest_inputs = {key: value.unsqueeze(1) for key, value in history[-1].items()}
            rebuild_env_ids = self.kv_cache_rebuild_env_ids
            if rebuild_env_ids is not None and rebuild_env_ids.numel() > 0:
                rebuild_inputs = {}
                for key in history[0]:
                    rebuild_inputs[key] = torch.stack(
                        tuple(entry[key].index_select(0, rebuild_env_ids) for entry in history),
                        dim=1,
                    )
                self.kv_cache_rebuild_inputs = self.process_neural_model_inputs(rebuild_inputs)
            else:
                self.kv_cache_rebuild_inputs = None
            return self.process_neural_model_inputs(latest_inputs)

        # Stack directly into the model's [B, T, ...] layout. The previous
        # default_collate + transpose + contiguous path materialized the full
        # history twice on every simulation step.
        model_inputs = {}
        for key in history[0]:
            value = torch.stack(tuple(entry[key] for entry in history), dim=1)
            if value.ndim < 2:
                raise ValueError(
                    f"History field {key!r} must have at least two dimensions after collate "
                    f"(got shape {tuple(value.shape)})."
                )
            model_inputs[key] = value

        return self.process_neural_model_inputs(model_inputs)

    def _require_feature_cache_model(self) -> "_FeatureCacheModel":
        neural_model = self._require_neural_model()
        if not callable(getattr(neural_model, "forward_feature_cached", None)):
            raise RuntimeError("use_feature_cache requires a model that implements forward_feature_cached().")
        return cast("_FeatureCacheModel", neural_model)

    def _require_kv_cache_model(self) -> "_KVCacheModel":
        neural_model = self._require_neural_model()
        if not all(
            callable(getattr(neural_model, name, None)) for name in ("forward_cached", "reset_transformer_cache")
        ):
            raise RuntimeError("use_kv_cache requires callable forward_cached() and reset_transformer_cache() methods.")
        return cast("_KVCacheModel", neural_model)

    def _processed_history_rows(self, env_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Stack and preprocess the raw history window, optionally for selected rows only."""
        history = tuple(self.states_history)
        return self.process_neural_model_inputs(
            {
                key: torch.stack(
                    tuple(entry[key] if env_ids is None else entry[key].index_select(0, env_ids) for entry in history),
                    dim=1,
                )
                for key in history[0]
            }
        )

    def _model_forward_frame_cached(self, model_inputs):
        """Encode only the newest frame, reuse earlier frame encodings, and predict the newest step."""
        neural_model = self._require_neural_model()
        tokens, key_features = neural_model.encode_frames(dict(model_inputs))
        self.frame_cache.append((tokens[:, 0], None if key_features is None else key_features[:, 0]))
        rebuild_env_ids = self.frame_cache_rebuild_env_ids
        self.frame_cache_rebuild_env_ids = None
        streaming = bool(getattr(neural_model, "supports_streaming", False))
        if streaming:
            spans = neural_model.transformer_model.config.attention_spans
            if sum(span - 1 for span in spans) + 1 > self.num_states_history:
                raise ValueError("Streaming inference requires a receptive field within num_states_history.")
        stream_cache = getattr(self, "stream_cache", None)
        if len(self.frame_cache) != len(self.states_history):
            stream_cache = None
            # Externally loaded history (e.g. preload_states_history): re-encode the complete window.
            all_tokens, all_keys = neural_model.encode_frames(self._processed_history_rows(None))
            self.frame_cache.clear()
            for step in range(all_tokens.shape[1]):
                self.frame_cache.append((all_tokens[:, step], None if all_keys is None else all_keys[:, step]))
        elif rebuild_env_ids is not None and rebuild_env_ids.numel() > 0:
            # Reset rows had their raw history replaced; re-encode those rows across the window.
            row_tokens, row_keys = neural_model.encode_frames(self._processed_history_rows(rebuild_env_ids))
            for step, (frame_tokens, frame_keys) in enumerate(self.frame_cache):
                frame_tokens.index_copy_(0, rebuild_env_ids, row_tokens[:, step])
                if frame_keys is not None and row_keys is not None:
                    frame_keys.index_copy_(0, rebuild_env_ids, row_keys[:, step])
        window_tokens = torch.stack([frame[0] for frame in self.frame_cache], dim=1)
        window_keys = None if key_features is None else torch.stack([frame[1] for frame in self.frame_cache], dim=1)
        if not streaming:
            output = neural_model.forward_encoded(window_tokens, window_keys)
        elif stream_cache is None:
            output, self.stream_cache = neural_model.build_stream(window_tokens, window_keys)
        else:
            # Each layer attends only to its cached cone; reset rows rebuild their cone from the window.
            output = neural_model.forward_stream(tokens, window_keys, stream_cache)
            if rebuild_env_ids is not None and rebuild_env_ids.numel() > 0:
                row_output, row_cache = neural_model.build_stream(
                    window_tokens.index_select(0, rebuild_env_ids),
                    None if window_keys is None else window_keys.index_select(0, rebuild_env_ids),
                )
                stream_cache.copy_rows_from(row_cache, rebuild_env_ids)
                output.state_prediction.index_copy_(0, rebuild_env_ids, row_output.state_prediction)
                for name, value in output.auxiliary_predictions.items():
                    value.index_copy_(0, rebuild_env_ids, row_output.auxiliary_predictions[name])
        self._publish_auxiliary_predictions(output)
        return output.state_prediction

    def _model_forward(self, model_inputs):
        if getattr(self, "use_frame_cache", False):
            return self._model_forward_frame_cached(model_inputs)
        if self.use_kv_cache:
            return self._model_forward_kv_cached(model_inputs)
        if not self.use_feature_cache:
            return super()._model_forward(model_inputs)
        prediction, self.feature_cache = self._require_feature_cache_model().forward_feature_cached(
            model_inputs,
            self.feature_cache,
            self.feature_cache_rebuild_env_ids,
        )
        self.feature_cache_rebuild_env_ids = None
        return prediction

    def _model_forward_kv_cached(self, model_inputs):
        """Run one incremental step and selectively rebuild reset cache rows."""
        neural_model = self._require_kv_cache_model()
        prediction, self.kv_cache = neural_model.forward_cached(
            model_inputs,
            self.kv_cache,
            max_cache_length=self.num_states_history,
        )

        rebuild_env_ids = self.kv_cache_rebuild_env_ids
        self.kv_cache_rebuild_env_ids = None
        if rebuild_env_ids is None or rebuild_env_ids.numel() == 0:
            return prediction

        rebuild_inputs = self.kv_cache_rebuild_inputs
        self.kv_cache_rebuild_inputs = None
        if rebuild_inputs is None:
            raise RuntimeError("KV-cache reset rows require prepared history inputs.")
        rebuilt_cache = None
        rebuilt_prediction = None
        history_length = next(value.shape[1] for value in rebuild_inputs.values() if isinstance(value, torch.Tensor))
        for step in range(history_length):
            step_inputs = {
                key: value[:, step : step + 1] if isinstance(value, torch.Tensor) else value
                for key, value in rebuild_inputs.items()
            }
            rebuilt_prediction, rebuilt_cache = neural_model.forward_cached(
                step_inputs,
                rebuilt_cache,
                max_cache_length=self.num_states_history,
            )

        if rebuilt_cache is None or rebuilt_prediction is None:
            raise RuntimeError("KV-cache reset rows require non-empty history inputs.")
        self.kv_cache.copy_rows_from(rebuilt_cache, rebuild_env_ids)
        prediction.index_copy_(0, rebuild_env_ids, rebuilt_prediction)
        return prediction
