# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RoPE Transformer incremental inference."""

from collections import deque

import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.models.model_transformer import GPT, GPTConfig
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.solvers.transformer_neural_solver import TransformerNeuralSolver


class _TransformerTestHarness(TransformerNeuralSolver):
    """Expose solver extension hooks without changing the production API."""

    predict = TransformerNeuralSolver._model_forward
    rebuild_history_rows = TransformerNeuralSolver._mark_history_rows_for_rebuild

    def process_neural_model_inputs(self, model_inputs):
        return model_inputs


def _gpt(position_encoding: str = "rope", attention_spans=None) -> GPT:
    torch.manual_seed(7)
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=6,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            position_encoding=position_encoding,
            attention_spans=attention_spans,
        )
    ).eval()
    for name, parameter in model.named_parameters():
        if name.endswith("lag_bias"):
            torch.nn.init.normal_(parameter)
    return model


def test_solver_history_stacks_directly_into_batch_time_layout():
    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.use_kv_cache = False
    solver.states_history = deque(
        (
            {"states": torch.tensor([[1.0], [2.0]]), "contact_token_overflow": torch.tensor([3, 4])},
            {"states": torch.tensor([[5.0], [6.0]]), "contact_token_overflow": torch.tensor([7, 8])},
        )
    )

    inputs = solver.get_neural_model_inputs()

    torch.testing.assert_close(inputs["states"], torch.tensor([[[1.0], [5.0]], [[2.0], [6.0]]]))
    torch.testing.assert_close(inputs["contact_token_overflow"], torch.tensor([[3, 7], [4, 8]]))


def test_solver_kv_input_preparation_uses_only_latest_frame():
    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.use_kv_cache = True
    solver.kv_cache_rebuild_env_ids = None
    solver.kv_cache_rebuild_inputs = None
    solver.states_history = deque(
        ({"observation": torch.full((2, 6), float(step))} for step in range(5)),
        maxlen=5,
    )

    inputs = solver.get_neural_model_inputs()

    assert inputs["observation"].shape == (2, 1, 6)
    torch.testing.assert_close(inputs["observation"][:, 0], torch.full((2, 6), 4.0))
    assert solver.kv_cache_rebuild_inputs is None


def _mixed_input_model(position_encoding: str = "rope", attention_spans=None) -> ModelMixedInput:
    input_sample = {"observation": torch.zeros(2, 1, 6)}
    input_cfg = {"low_dim": ["observation"]}
    network_cfg = {
        "normalize_input": False,
        "normalize_output": False,
        "encoder": {
            "low_dim": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            }
        },
        "transformer": {
            "block_size": 8,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
            "position_encoding": position_encoding,
            "attention_spans": attention_spans,
        },
        "model": {
            "mlp": {
                "layer_sizes": [8],
                "activation": "relu",
                "layernorm": False,
            }
        },
    }
    model = ModelMixedInput(input_sample, 4, input_cfg, network_cfg, device="cpu").eval()
    for name, parameter in model.named_parameters():
        if name.endswith("lag_bias"):
            torch.nn.init.normal_(parameter)
    return model


def _mixed_contact_model() -> ModelMixedInput:
    input_sample = {
        "observation": torch.zeros(2, 1, 6),
        "contact_tokens": torch.zeros(2, 1, 4, CONTACT_TOKEN_DIM),
    }
    input_cfg = {
        "low_dim": ["observation"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "encoder_type": "shared_per_body",
            "body_latent_dim": 4,
            "hidden_dim": 8,
            "max_other_bodies": 4,
        },
    }
    network_cfg = {
        "normalize_input": False,
        "normalize_output": False,
        "encoder": {
            "low_dim": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            }
        },
        "transformer": {
            "block_size": 8,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        "model": {
            "mlp": {
                "layer_sizes": [8],
                "activation": "relu",
                "layernorm": False,
            }
        },
    }
    return ModelMixedInput(
        input_sample,
        4,
        input_cfg,
        network_cfg,
        device="cpu",
        contact_mode="newton_native",
        num_bodies=4,
    ).eval()


def test_default_position_encoding_preserves_legacy_state_dict():
    legacy_model = _gpt("learned_absolute")
    restored_model = _gpt("learned_absolute")

    assert "transformer.wpe.weight" in legacy_model.state_dict()
    restored_model.load_state_dict(legacy_model.state_dict(), strict=True)


def test_rope_cached_forward_matches_full_prefix():
    model = _gpt()
    inputs = torch.randn(3, 6, 6)

    with torch.inference_mode():
        expected = model(inputs)
        cache = None
        predictions = []
        for step in range(inputs.shape[1]):
            prediction, cache = model.forward_cached(
                inputs[:, step : step + 1],
                cache,
                max_cache_length=8,
            )
            predictions.append(prediction)

    torch.testing.assert_close(torch.cat(predictions, dim=1), expected, atol=1e-6, rtol=1e-5)
    assert cache is not None
    assert cache.lengths.tolist() == [6, 6, 6]
    assert cache.next_positions.tolist() == [6, 6, 6]


def test_cache_eviction_and_partial_reset():
    model = _gpt()
    cache = model.init_kv_cache(batch_size=3, max_cache_length=3, device="cpu")

    with torch.inference_mode():
        for _ in range(5):
            _, cache = model.forward_cached(torch.randn(3, 1, 6), cache)

    assert cache.lengths.tolist() == [3, 3, 3]
    assert cache.positions.tolist() == [[3, 4, 2], [3, 4, 2], [3, 4, 2]]

    cache.reset([0, 2])

    assert cache.lengths.tolist() == [0, 3, 0]
    assert cache.next_positions.tolist() == [0, 5, 0]
    assert not cache.valid_mask[0].any()
    assert cache.valid_mask[1].all()
    assert not cache.valid_mask[2].any()


def test_rope_cached_forward_matches_prefix_and_bounds_evicted_cache():
    model = _mixed_input_model()
    inputs = torch.randn(2, 12, 6)
    cache = None

    with torch.inference_mode():
        for step in range(inputs.shape[1]):
            window = {"observation": inputs[:, max(0, step - 4) : step + 1]}
            expected = model(window, single_step=True)
            actual, cache = model.forward_cached(
                {"observation": inputs[:, step : step + 1]},
                cache,
                max_cache_length=5,
            )
            if step < 5:
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            else:
                # Once eviction starts, retained deep-layer K/V entries still
                # encode evicted causal context and are not exactly equivalent
                # to recomputing a fresh sliding window.
                assert torch.isfinite(actual).all()

    assert cache is not None
    assert cache.lengths.tolist() == [5, 5]
    assert cache.next_positions.tolist() == [12, 12]


def test_solver_kv_cache_rebuilds_only_reset_rows():
    model = _mixed_input_model()
    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.num_envs = 2
    solver.num_states_history = 5
    solver.use_feature_cache = False
    solver.use_kv_cache = True
    solver.feature_cache = None
    solver.feature_cache_rebuild_env_ids = None
    solver.kv_cache = None
    solver.kv_cache_rebuild_env_ids = None
    solver.kv_cache_rebuild_inputs = None
    solver.neural_model = model
    solver.torch_device = "cpu"
    solver.predicted_body_net_contact_forces_b = None
    solver.root_body_q = torch.zeros(2, 7)
    solver.states = torch.zeros(2, 4)
    solver.states_embedding = torch.zeros(2, 4)
    solver.joint_f = torch.zeros(2, 2)
    solver.gravity_dir = torch.zeros(2, 3)
    solver.contact_inputs = {}

    initial_values = torch.randn(2, 5, 6)
    solver.states_history = deque(
        ({"observation": initial_values[:, step].clone()} for step in range(5)),
        maxlen=5,
    )
    with torch.inference_mode():
        for step in range(5):
            solver.predict({"observation": initial_values[:, step : step + 1]})

        solver.reset([0])
        next_values = torch.randn(2, 1, 6)
        solver.states_history.append({"observation": next_values[:, 0].clone()})
        full_window = {
            "observation": torch.stack(tuple(entry["observation"] for entry in solver.states_history), dim=1)
        }
        model_inputs = solver.get_neural_model_inputs()
        assert solver.kv_cache_rebuild_inputs is not None
        rebuild_observation = solver.kv_cache_rebuild_inputs["observation"]
        assert isinstance(rebuild_observation, torch.Tensor)
        assert rebuild_observation.shape == (1, 5, 6)
        expected = model(full_window, single_step=True)
        actual = solver.predict(model_inputs)

    # The reset row is rebuilt from its complete solver window and remains
    # exactly equivalent. The uninterrupted row keeps incremental cache
    # semantics after eviction.
    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-5)
    assert model_inputs["observation"].shape == (2, 1, 6)
    assert solver.kv_cache_rebuild_inputs is None
    assert solver.kv_cache is not None
    assert solver.kv_cache.lengths.tolist() == [5, 5]
    assert solver.kv_cache.next_positions.tolist() == [5, 6]


@pytest.mark.parametrize("method_name", ["forward_feature_cached", "forward_cached", "reset_transformer_cache"])
def test_solver_rejects_noncallable_cache_interfaces(monkeypatch, method_name):
    model = _mixed_input_model()
    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.neural_model = model
    solver.num_states_history = 5
    solver.use_kv_cache = method_name != "forward_feature_cached"
    solver.use_feature_cache = not solver.use_kv_cache
    solver.feature_cache = None
    solver.feature_cache_rebuild_env_ids = None
    solver.kv_cache = None
    solver.kv_cache_rebuild_env_ids = None
    solver.kv_cache_rebuild_inputs = None
    if method_name == "reset_transformer_cache":
        solver.kv_cache = model.init_transformer_cache(2, 5)
    monkeypatch.setattr(model, method_name, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match=method_name):
        with torch.inference_mode():
            if method_name == "reset_transformer_cache":
                solver.rebuild_history_rows(torch.tensor([0]))
            else:
                solver.predict({"observation": torch.randn(2, 1, 6)})


def test_solver_feature_cache_accepts_a_feature_only_custom_model():
    class FeatureOnlyModel(torch.nn.Module):
        def forward_feature_cached(self, inputs, cache, rebuild_batch_indices):
            return inputs["observation"][:, -1:], inputs["observation"].clone()

    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.neural_model = FeatureOnlyModel()
    solver.use_kv_cache = False
    solver.use_feature_cache = True
    solver.feature_cache = None
    solver.feature_cache_rebuild_env_ids = None
    observations = torch.randn(2, 3, 6)
    actual = solver.predict({"observation": observations})
    torch.testing.assert_close(actual, observations[:, -1:])
    torch.testing.assert_close(solver.feature_cache, observations)


def test_mixed_input_exposes_separate_cached_interface():
    model = _mixed_input_model()
    inputs = {"observation": torch.randn(2, 5, 6)}

    with torch.inference_mode():
        expected = model(inputs)
        cache = None
        predictions = []
        for step in range(5):
            prediction, cache = model.forward_cached(
                {"observation": inputs["observation"][:, step : step + 1]},
                cache,
                max_cache_length=8,
            )
            predictions.append(prediction)

    torch.testing.assert_close(torch.cat(predictions, dim=1), expected, atol=1e-6, rtol=1e-5)


def test_cached_interface_rejects_absolute_position_model():
    model = _mixed_input_model("learned_absolute")

    with pytest.raises(ValueError, match="position_encoding='rope'"):
        with torch.inference_mode():
            model.forward_cached({"observation": torch.randn(2, 1, 6)})


def test_feature_cache_matches_legacy_sliding_window():
    model = _mixed_input_model("learned_absolute")
    inputs = torch.randn(2, 12, 6)
    feature_cache = None

    with torch.inference_mode():
        for step in range(inputs.shape[1]):
            window_start = max(0, step - 4)
            window = {"observation": inputs[:, window_start : step + 1]}
            expected = model(window, single_step=True)
            actual, feature_cache = model.forward_feature_cached(window, feature_cache)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_feature_cache_encodes_only_latest_frame_after_initial_window():
    model = _mixed_input_model("learned_absolute")
    encoded_lengths = []

    def record_sequence_length(_module, args, _output):
        encoded_lengths.append(args[0].shape[1])

    hook = model.encoders["low_dim"].register_forward_hook(record_sequence_length)
    first_window = {"observation": torch.randn(2, 5, 6)}
    second_window = {"observation": torch.randn(2, 5, 6)}

    with torch.inference_mode():
        _, feature_cache = model.forward_feature_cached(first_window)
        model.forward_feature_cached(second_window, feature_cache)
    hook.remove()

    assert encoded_lengths == [5, 1]


def test_feature_cache_rebuild_after_reset_matches_legacy_forward():
    model = _mixed_input_model("learned_absolute")
    initial_window = {"observation": torch.randn(2, 5, 6)}
    reset_window = {"observation": torch.randn(2, 5, 6)}

    with torch.inference_mode():
        _, feature_cache = model.forward_feature_cached(initial_window)
        expected = model(reset_window, single_step=True)
        actual, rebuilt_cache = model.forward_feature_cached(reset_window, feature_cache=None)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert rebuilt_cache.shape == feature_cache.shape


def test_feature_cache_rebuilds_only_reset_batch_rows():
    model = _mixed_input_model("learned_absolute")
    initial_values = torch.randn(2, 5, 6)
    next_values = torch.cat((initial_values[:, 1:], torch.randn(2, 1, 6)), dim=1)
    next_values[0, :-1] = 0.0

    with torch.inference_mode():
        _, feature_cache = model.forward_feature_cached({"observation": initial_values})
        expected = model({"observation": next_values}, single_step=True)
        actual, _ = model.forward_feature_cached(
            {"observation": next_values},
            feature_cache,
            rebuild_batch_indices=[0],
        )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_feature_cache_matches_shared_per_body_contact_encoder():
    model = _mixed_contact_model()
    observations = torch.randn(2, 12, 6)
    contact_tokens = torch.randn(2, 12, 4, CONTACT_TOKEN_DIM)
    contact_tokens[..., 0] = 1.0
    contact_tokens[..., 1] = torch.arange(4)
    contact_tokens[..., 2] = -1.0
    feature_cache = None

    with torch.inference_mode():
        for step in range(observations.shape[1]):
            window_start = max(0, step - 4)
            window = {
                "observation": observations[:, window_start : step + 1],
                "contact_tokens": contact_tokens[:, window_start : step + 1],
            }
            expected = model(window, single_step=True)
            actual, feature_cache = model.forward_feature_cached(window, feature_cache)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("position_encoding", "spans"), [("learned_absolute", None), ("rope", None), ("relative_lag", (3, 2))]
)
@pytest.mark.parametrize("steps", [1, 3, 8])
def test_gpt_last_only_matches_last_row_of_full_window(position_encoding, spans, steps):
    model = _gpt(position_encoding, spans)
    tokens = torch.randn(3, steps, 6)
    with torch.inference_mode():
        torch.testing.assert_close(model(tokens, last_only=True), model(tokens)[:, -1:], atol=1e-6, rtol=1e-5)


def _frame_cache_solver(model: ModelMixedInput, history: int) -> _TransformerTestHarness:
    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.num_envs = 2
    solver.num_states_history = history
    solver.use_feature_cache = solver.use_kv_cache = False
    solver.use_frame_cache = True
    solver.neural_model = model
    solver.torch_device = "cpu"
    solver.predicted_body_net_contact_forces_b = None
    solver.feature_cache = solver.kv_cache = None
    solver.feature_cache_rebuild_env_ids = solver.kv_cache_rebuild_env_ids = None
    solver.frame_cache = deque(maxlen=history)
    solver.frame_cache_rebuild_env_ids = None
    solver.stream_cache = None
    solver.states_history = deque(maxlen=history)
    return solver


@pytest.mark.parametrize(("position_encoding", "spans"), [("learned_absolute", None), ("relative_lag", (3, 2))])
def test_solver_frame_cache_matches_full_window_through_growth_eviction_and_row_rebuild(position_encoding, spans):
    model = _mixed_input_model(position_encoding, spans)
    solver = _frame_cache_solver(model, history=4)
    frames = torch.randn(2, 9, 6)

    def full_window():
        return {"observation": torch.stack([entry["observation"] for entry in solver.states_history], dim=1)}

    with torch.inference_mode():
        for step in range(9):
            solver.states_history.append({"observation": frames[:, step].clone()})
            if step == 6:
                # A partial reset rewrites row 0's raw history; its cached encodings must be rebuilt.
                for entry in solver.states_history:
                    entry["observation"][0] = frames[0, step]
                solver.rebuild_history_rows(torch.tensor([0]))
            inputs = solver.get_neural_model_inputs()
            assert inputs["observation"].shape == (2, 1, 6)
            expected = model(full_window(), single_step=True)
            torch.testing.assert_close(solver.predict(inputs), expected, atol=1e-6, rtol=1e-5)
        assert len(solver.frame_cache) == 4

        # Externally loaded history (frame cache out of step) re-encodes the whole window.
        solver.frame_cache.clear()
        solver.states_history.append({"observation": torch.randn(2, 6)})
        expected = model(full_window(), single_step=True)
        torch.testing.assert_close(solver.predict(solver.get_neural_model_inputs()), expected, atol=1e-6, rtol=1e-5)


def test_solver_compiles_only_fixed_shape_steady_stream_steps(monkeypatch):
    model = _mixed_input_model("relative_lag", (3, 2))
    solver = _frame_cache_solver(model, history=4)
    solver.inference_compile = True
    calls, transforms = [], []

    def process(model_inputs):
        # Record which token transform preprocessing would use for this batch.
        transform = solver._contact_token_transform()
        transforms.append((model_inputs["observation"].shape[0], getattr(transform, "compiled", False)))
        return model_inputs

    solver.process_neural_model_inputs = process

    def fake_compile(function, dynamic):
        assert dynamic is False

        def wrapped(*args, **kwargs):
            calls.append(
                (function.__name__, args[0]["observation"].shape if isinstance(args[0], dict) else args[0].shape)
            )
            return function(*args, **kwargs)

        wrapped.compiled = True
        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    frames = torch.randn(2, 9, 6)
    with torch.inference_mode():
        for step in range(9):
            solver.states_history.append({"observation": frames[:, step].clone()})
            if step == 6:
                for entry in solver.states_history:
                    entry["observation"][0] = frames[0, step]
                solver.rebuild_history_rows(torch.tensor([0]))
            expected = model(
                {"observation": torch.stack([entry["observation"] for entry in solver.states_history], dim=1)},
                single_step=True,
            )
            torch.testing.assert_close(solver.predict(solver.get_neural_model_inputs()), expected, atol=1e-6, rtol=1e-5)
    # The first frame builds the stream eagerly; every later frame runs both compiled callables on full batches.
    assert [name for name, _ in calls] == ["encode_frames", "forward_stream"] * 8
    assert all(shape[0] == 2 and shape[1] == 1 for _, shape in calls)
    # Newest-frame preprocessing uses the compiled token transform; the reset-row rebuild at frame 6 stays eager.
    assert [batch for batch, _ in transforms] == [2] * 7 + [1, 2, 2]
    assert [compiled for _, compiled in transforms] == [True] * 7 + [False, True, True]
    compiled_model, compiled = solver._compiled_inference
    assert compiled_model is model and len(compiled) == 3


def test_frame_cache_is_exclusive_with_other_caches():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TransformerNeuralSolver(use_kv_cache=True, use_frame_cache=True)


@pytest.mark.parametrize(
    ("states_frame", "anchor", "accepted"), [("world", None, True), ("body", "every", True), ("body", "first", False)]
)
def test_frame_cache_requires_world_states_or_per_frame_anchoring(monkeypatch, states_frame, anchor, accepted):
    from isaaclab_neural.solvers.neural_solver import NeuralSolver

    def init(self, **kwargs):
        # World-frame solvers never set anchor_frame_step (the contact-free Reach and Cartpole recipes).
        self.states_frame = states_frame
        if anchor is not None:
            self.anchor_frame_step = anchor

    monkeypatch.setattr(NeuralSolver, "__init__", init)
    if accepted:
        assert TransformerNeuralSolver(use_frame_cache=True).use_frame_cache
    else:
        with pytest.raises(ValueError, match="per-frame anchoring"):
            TransformerNeuralSolver(use_frame_cache=True)


def test_relative_lag_stream_matches_a_sliding_window_over_its_receptive_field():
    # spans (3, 2): each output depends on the newest 1 + 2 + 1 = 4 inputs only.
    model = _gpt("relative_lag", (3, 2))
    inputs = torch.randn(3, 8, 6)
    with torch.inference_mode():
        full = model(inputs)
        _, cache = model.build_stream_cache(inputs[:, :1])
        for step in range(1, 8):
            streamed = model.forward_stream(inputs[:, step : step + 1], cache)
            window = inputs[:, max(step - 3, 0) : step + 1]
            torch.testing.assert_close(streamed, model(window)[:, -1:], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(streamed, full[:, step : step + 1], atol=1e-6, rtol=1e-5)
            rebuilt, _ = model.build_stream_cache(window)
            torch.testing.assert_close(rebuilt, streamed, atol=1e-6, rtol=1e-5)


def test_relative_lag_rejects_spans_that_do_not_fit():
    with pytest.raises(ValueError, match="one attention span"):
        _gpt("relative_lag", (3,))
    with pytest.raises(ValueError, match="receptive field"):
        _gpt("relative_lag", (6, 5))
