# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE.md for details].

from dataclasses import dataclass

import torch
import torch.nn as nn

from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.contacts.tensor_utils import (
    mask_inactive_contact_fields,
    normalize_contact_field,
    normalize_contact_tokens,
)
from isaaclab_neural.models.base_models import CNNBase, GRUBase, LSTMBase, MLPBase
from isaaclab_neural.models.contact_set_model import ContactSetEncoderBlock
from isaaclab_neural.models.model_kan import KAN
from isaaclab_neural.models.model_transformer import GPT, GPTConfig
from isaaclab_neural.models.shared_per_body_contact_encoder import SharedPerBodyContactEncoder


@dataclass
class ModelMixedInputOutput:
    """Structured state and auxiliary predictions from :class:`ModelMixedInput`."""

    state_prediction: torch.Tensor
    """Predicted state transition, shape ``[B, T, state_prediction_dim]``."""

    auxiliary_predictions: dict[str, torch.Tensor]
    """Auxiliary predictions keyed by target name."""


class MLPDeterministic(nn.Module):
    def __init__(self, input_shape, output_dim, network_cfg, device="cuda:0"):
        super().__init__()

        self.device = device

        self.feature_net = MLPBase(input_shape[0], network_cfg["mlp"], device=device)

        self.output_net = nn.Linear(self.feature_net.out_features, output_dim, device=device)

    def forward(self, inputs, deterministic=False):
        features = self.feature_net(inputs)
        output = self.output_net(features)
        return output

    def to(self, device):
        self.device = device
        self.feature_net.to(device)
        self.output_net.to(device)


class ModelMixedInput(nn.Module):
    def __init__(
        self,
        input_sample,
        output_dim,
        input_cfg,
        network_cfg,
        device="cuda:0",
        *,
        contact_mode="fixed_ground",
        num_bodies=None,
    ):
        super().__init__()

        self.device = device
        self.model = None

        self.input_rms = None
        self.normalize_input = network_cfg.get("normalize_input", False)
        self.use_native_contact_processing = contact_mode == "newton_native"
        self.use_contact_token_set = "contact_set" in input_cfg
        self.output_rms = None
        self.normalize_output = network_cfg.get("normalize_output", False)

        self.encoders, self.feature_dim = self.construct_input_encoders(
            input_cfg,
            network_cfg["encoder"],
            input_sample,
            device=device,
            transformer_cfg=network_cfg.get("transformer"),
            num_bodies=num_bodies,
        )

        if "rnn" in network_cfg:
            self.is_rnn = True
            if network_cfg["rnn"]["net"] == "lstm":
                self.rnn = LSTMBase(self.feature_dim, network_cfg["rnn"], device=self.device)
            elif network_cfg["rnn"]["net"] == "gru":
                self.rnn = GRUBase(self.feature_dim, network_cfg["rnn"], device=self.device)
            else:
                raise NotImplementedError
            self.feature_dim = self.rnn.hidden_size
        else:
            self.is_rnn = False
            self.rnn = None

        if "transformer" in network_cfg:
            model_args = dict(
                n_layer=network_cfg["transformer"]["n_layer"],
                n_head=network_cfg["transformer"]["n_head"],
                n_embd=network_cfg["transformer"]["n_embd"],
                block_size=network_cfg["transformer"]["block_size"],
                bias=network_cfg["transformer"]["bias"],
                vocab_size=self.feature_dim,
                dropout=network_cfg["transformer"]["dropout"],
                position_encoding=network_cfg["transformer"].get("position_encoding", "learned_absolute"),
                rope_base=network_cfg["transformer"].get("rope_base", 10000.0),
            )
            gptconf = GPTConfig(**model_args)

            self.transformer_model = GPT(gptconf)
            self.transformer_model.to(self.device)

            self.is_transformer = True
            self.feature_dim = self.transformer_model.config.n_embd
        else:
            self.is_transformer = False
            self.transformer_model = None

        if "kan" in network_cfg:
            self.model = KAN(
                layers_hidden=[self.feature_dim] + network_cfg["kan"]["layer_sizes"] + [output_dim],
                grid_num=network_cfg["kan"].get("grid_num", 5),
                order=network_cfg["kan"].get("order", 3),
                scale_noise=network_cfg["kan"].get("scale_noise", 0.1),
                scale_base=network_cfg["kan"].get("scale_base", 1.0),
                scale_spline=network_cfg["kan"].get("scale_spline", 1.0),
                enable_standalone_scale_spline=network_cfg["kan"].get("enable_standalone_scale_spline", True),
                base_activation=torch.nn.SiLU,
                grid_range=network_cfg["kan"].get("grid_range", [-1, 1]),
            )
            self.model.to(self.device)
            self.is_kan = True
        else:
            self.is_kan = False

        if self.model is None:
            self.model = MLPDeterministic((self.feature_dim,), output_dim, network_cfg["model"], device=device)

        force_head_cfg = network_cfg.get("body_net_contact_force_head", {})
        self.predict_body_net_contact_forces = bool(force_head_cfg.get("enabled", False))
        self.body_net_contact_force_rms = None
        self.num_bodies = int(num_bodies) if num_bodies is not None else None
        if self.predict_body_net_contact_forces:
            if self.num_bodies is None:
                raise ValueError("body_net_contact_force_head requires num_bodies.")
            head_network_cfg = force_head_cfg.get("model", network_cfg["model"])
            self.body_net_contact_force_head = MLPDeterministic(
                (self.feature_dim,),
                self.num_bodies * 3,
                head_network_cfg,
                device=device,
            )

        self.output_tanh = network_cfg.get("output_tanh", False)

    def construct_input_encoders(
        self,
        input_cfg,
        encoder_cfg,
        input_sample,
        device="cuda:0",
        *,
        transformer_cfg=None,
        num_bodies=None,
    ):
        encoders = nn.ModuleDict()

        """
        low-dim inputs
        """
        if len(input_cfg.get("low_dim", [])) > 0:
            low_dim_size = 0
            self.low_dim_input_names = input_cfg.get("low_dim")
            for low_dim_input_name in self.low_dim_input_names:
                assert len(input_sample[low_dim_input_name].shape) in [2, 3]  # (B, *) or (B, T, *)
                low_dim_size += input_sample[low_dim_input_name].shape[-1]

            assert "low_dim" in encoder_cfg
            low_dim_encoder = MLPBase(low_dim_size, encoder_cfg["low_dim"], device=device)
            encoders["low_dim"] = low_dim_encoder
        else:
            self.low_dim_input_names = []

        if "contact_set" in input_cfg:
            contact_cfg = input_cfg["contact_set"]
            contact_dim = int(contact_cfg.get("dim", input_sample["contact_tokens"].shape[-1]))
            encoder_type = contact_cfg.get("encoder_type", "global_attention")
            if encoder_type == "global_attention":
                hidden_size = int(contact_cfg.get("hidden_size", transformer_cfg["n_embd"] if transformer_cfg else 192))
                self.contact_set_encoder = ContactSetEncoderBlock(
                    contact_dim=contact_dim,
                    hidden_size=hidden_size,
                    num_layers=int(contact_cfg.get("encoder_layers", 2)),
                    num_heads=int(contact_cfg.get("encoder_heads", 4)),
                    num_latent_queries=int(contact_cfg.get("num_latent_queries", 8)),
                    max_bodies=int(contact_cfg.get("max_bodies", 32)),
                    max_other_bodies=int(contact_cfg.get("max_other_bodies", 32)),
                    dropout=float(contact_cfg.get("dropout", 0.0)),
                    device=device,
                )
            elif encoder_type == "shared_per_body":
                configured_num_bodies = contact_cfg.get("num_bodies")
                if num_bodies is not None and configured_num_bodies is not None:
                    if int(configured_num_bodies) != int(num_bodies):
                        raise ValueError(
                            "Configured contact_set num_bodies does not match runtime primary-body metadata."
                        )
                resolved_num_bodies = num_bodies if num_bodies is not None else configured_num_bodies
                if resolved_num_bodies is None:
                    raise ValueError(
                        "contact_set encoder_type='shared_per_body' requires num_bodies "
                        "from runtime metadata or config."
                    )
                if contact_dim != CONTACT_TOKEN_DIM:
                    raise ValueError("shared_per_body contact encoding requires the canonical 17-D token schema.")
                self.contact_set_encoder = SharedPerBodyContactEncoder(
                    num_bodies=int(resolved_num_bodies),
                    body_latent_dim=int(contact_cfg.get("body_latent_dim", 16)),
                    hidden_dim=int(contact_cfg.get("hidden_dim", 64)),
                    max_other_bodies=int(contact_cfg.get("max_other_bodies", 32)),
                    device=device,
                )
            else:
                raise ValueError(
                    f"Unsupported contact_set encoder_type '{encoder_type}'. "
                    "Expected 'global_attention' or 'shared_per_body'."
                )
            encoders["contact_set"] = self.contact_set_encoder

        """
        rgb inputs
        """
        rgb_input_names = input_cfg.get("rgb", [])
        for rgb_input_name in rgb_input_names:
            assert rgb_input_name in input_sample
            assert len(input_sample[rgb_input_name].shape) in [4, 5]  # (B, C, H, W) or (B, T, C, H, W)
            assert "rgb" in encoder_cfg
            if rgb_input_name in encoder_cfg["rgb"]:
                config = encoder_cfg["rgb"][rgb_input_name]
            elif "default" in encoder_cfg["rgb"]:
                config = encoder_cfg["rgb"]["default"]
            else:
                raise ValueError(f"No '{rgb_input_name}' nor 'default' in encoder_cfg['rgb']")

            rgb_encoder = CNNBase(input_sample[rgb_input_name].shape[1:], config, device=device)
            encoders[rgb_input_name] = rgb_encoder

        feature_dim = 0
        for input_name in encoders:
            feature_dim += encoders[input_name].out_features

        return encoders, feature_dim

    def set_input_rms(self, data_rms):
        rms_dict = {}
        for input_name in self.encoders:
            if input_name == "low_dim":
                for low_dim_input_name in self.low_dim_input_names:
                    if low_dim_input_name in data_rms:
                        rms_dict[low_dim_input_name] = data_rms[low_dim_input_name]
            elif input_name == "contact_set":
                if "contact_tokens" in data_rms:
                    rms_dict["contact_tokens"] = data_rms["contact_tokens"]
            else:
                rms_dict[input_name] = data_rms[input_name]
        self.input_rms = nn.ModuleDict(rms_dict)

    def set_output_rms(self, output_rms):
        self.output_rms = output_rms

    def set_body_net_contact_force_rms(self, force_rms):
        """Set RMS statistics for per-body resultant linear contact forces [N].

        These forces include normal and tangential/friction components and
        exclude contact moments and torques.
        """
        self.body_net_contact_force_rms = force_rms

    def get_rms_info(self):
        """Return RMS keys and shapes for checkpoint reconstruction."""
        info = {}
        if self.input_rms is not None:
            info["input_rms"] = {k: tuple(v.mean.shape) for k, v in self.input_rms.items()}
        if self.output_rms is not None:
            info["output_rms"] = tuple(self.output_rms.mean.shape)
        if self.body_net_contact_force_rms is not None:
            info["body_net_contact_force_rms"] = tuple(self.body_net_contact_force_rms.mean.shape)
        return info

    def extract_input_features(self, input_dict):  # input can be in shape (B, input_dim) or (B, T, input_dim)
        features = []
        for input_name in self.encoders:
            if input_name == "low_dim":
                low_dim_input_list = []
                for low_dim_input_name in self.low_dim_input_names:
                    low_dim_input_list.append(input_dict[low_dim_input_name])
                cur_input = torch.cat(low_dim_input_list, dim=-1)
                features.append(self.encoders[input_name](cur_input))
            elif input_name == "contact_set":
                contact_features = self.encoders[input_name](input_dict["contact_tokens"])
                if isinstance(self.encoders[input_name], SharedPerBodyContactEncoder):
                    contact_features = contact_features.flatten(start_dim=-2)
                features.append(contact_features)
            else:
                cur_input = input_dict[input_name]
                features.append(self.encoders[input_name](cur_input))  # each feature is (B, (T), feature_dim_i)
        features = torch.cat(features, dim=-1)
        return features

    def _prepare_inputs(self, input_dict, inject_noise=False):
        """Normalize, augment, and mask model inputs."""
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                if obs_key == "contact_tokens":
                    input_dict[obs_key] = normalize_contact_tokens(
                        input_dict[obs_key],
                        self.input_rms[obs_key],
                    )
                elif getattr(self, "use_native_contact_processing", False) and obs_key.startswith("contact_"):
                    contact_masks = input_dict.get("contact_masks")
                    if contact_masks is None:
                        raise ValueError("Contact RMS mode 'masked_shared' requires explicit contact_masks.")
                    input_dict[obs_key] = normalize_contact_field(
                        input_dict[obs_key], contact_masks, self.input_rms[obs_key]
                    )
                else:
                    input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        if inject_noise:
            for obs_key in input_dict.keys():
                if not torch.is_floating_point(input_dict[obs_key]):
                    continue
                if obs_key == "contact_tokens":
                    # Keep categorical identity channels (valid/slots/dynamic) clean.
                    noise = torch.randn_like(input_dict[obs_key]) * 0.01
                    noise[..., :4] = 0.0
                    input_dict[obs_key] = input_dict[obs_key] + noise
                else:
                    input_dict[obs_key] = input_dict[obs_key] + torch.randn_like(input_dict[obs_key]) * 0.01

        # Normalization and noise turn zero padding into nonzero values. Apply
        # the mask again so inactive contacts remain zero at the encoder input.
        if getattr(self, "use_native_contact_processing", False):
            mask_inactive_contact_fields(input_dict)
        if getattr(self, "use_contact_token_set", False) and "contact_tokens" in input_dict:
            from isaaclab_neural.contacts.contact_set_encoder import mask_invalid_contact_tokens

            input_dict["contact_tokens"] = mask_invalid_contact_tokens(input_dict["contact_tokens"])
        return input_dict

    def _process_output(self, output):
        """Apply configured output transforms."""
        if self.output_tanh:
            output = torch.tanh(output)
        if self.normalize_output:
            output = self.output_rms.normalize(output, un_norm=True)
        return output

    def _extract_trunk_features(self, input_dict, inject_noise=False):
        """Preprocess inputs and run the shared encoder and temporal trunk."""
        input_dict = self._prepare_inputs(input_dict, inject_noise)
        features = self.extract_input_features(input_dict)  # (B, T, feature_dim)

        if self.is_rnn:
            features = self.rnn(features)  # (B, T, rnn_hidden_dim)

        if self.is_transformer:
            features = self.transformer_model(features)  # (B, T, transform_embed_dim)

        return features

    def _decode_state(self, features, deterministic=False):
        """Decode the state prediction from shared trunk features."""
        output = self.model(features, deterministic=deterministic)
        return self._process_output(output)

    def forward(
        self,
        input_dict,
        single_step=False,
        deterministic=False,
        inject_noise=False,
        return_aux=False,
    ):
        """
        Forward pass through the model.

        Args:
            input_dict: Dictionary of input tensors, each shape (B, T, input_dim)
            single_step: If True, returns only the last timestep prediction (B, 1, output_dim)
                        If False, returns all timesteps (B, T, output_dim)
            deterministic: Whether to use deterministic mode (unused in current implementation)
            inject_noise: If True, adds Gaussian noise to inputs for data augmentation
            return_aux: If True, returns state and enabled auxiliary predictions.

        Returns:
            State tensor by default, or :class:`ModelMixedInputOutput` when
            ``return_aux`` is True.
        """
        if return_aux:
            return self.forward_with_aux(
                input_dict,
                single_step=single_step,
                deterministic=deterministic,
                inject_noise=inject_noise,
            )

        features = self._extract_trunk_features(input_dict, inject_noise=inject_noise)
        output = self._decode_state(features, deterministic=deterministic)

        if single_step:
            output = output[:, -1:, :]

        return output

    def forward_with_aux(self, input_dict, single_step=False, deterministic=False, inject_noise=False):
        """Return state and enabled auxiliary predictions from one trunk pass."""
        features = self._extract_trunk_features(input_dict, inject_noise=inject_noise)
        state_prediction = self._decode_state(features, deterministic=deterministic)
        auxiliary_predictions = {}
        if self.predict_body_net_contact_forces:
            force_prediction = self.body_net_contact_force_head(
                features,
                deterministic=deterministic,
            )
            force_prediction = force_prediction.unflatten(-1, (self.num_bodies, 3))
            if self.normalize_output:
                if self.body_net_contact_force_rms is None:
                    raise RuntimeError("Normalized force-head output requires body net contact-force RMS statistics.")
                force_prediction = self.body_net_contact_force_rms.normalize(
                    force_prediction,
                    un_norm=True,
                )
            auxiliary_predictions["body_net_contact_forces_b"] = force_prediction

        if single_step:
            state_prediction = state_prediction[:, -1:, :]
            auxiliary_predictions = {
                name: prediction[:, -1:, ...] for name, prediction in auxiliary_predictions.items()
            }

        return ModelMixedInputOutput(
            state_prediction=state_prediction,
            auxiliary_predictions=auxiliary_predictions,
        )

    def forward_feature_cached(
        self,
        input_dict,
        feature_cache=None,
        rebuild_batch_indices=None,
        deterministic=False,
    ):
        """Run full-window Transformer inference while encoding only the newest frame.

        This path preserves the existing Transformer and position-encoding
        semantics. On the first call, or after a reset, the complete input
        window is encoded. Subsequent calls append only the latest encoded
        frame and retain the preceding encoded features.

        Args:
            input_dict: Dictionary of full-window input tensors shaped ``(B, T, ...)``.
            feature_cache: Encoded feature window returned by the previous call.
            rebuild_batch_indices: Optional batch rows whose complete windows
                must be re-encoded after an environment reset.
            deterministic: Whether to use deterministic output mode.

        Returns:
            A tuple containing the latest prediction shaped ``(B, 1, output_dim)``
            and the updated encoded feature window.
        """
        if not self.is_transformer or self.is_rnn:
            raise RuntimeError("forward_feature_cached() requires a Transformer model without an RNN.")
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "forward_feature_cached() requires eval mode and torch.inference_mode() or torch.no_grad()."
            )
        if not input_dict:
            raise ValueError("forward_feature_cached() requires at least one input tensor.")

        first_input = next((value for value in input_dict.values() if isinstance(value, torch.Tensor)), None)
        if first_input is None:
            raise ValueError("forward_feature_cached() requires at least one tensor input.")
        if first_input.ndim < 2:
            raise ValueError("Feature-cache inputs must include batch and time dimensions.")
        batch_size, history_length = first_input.shape[:2]
        if history_length == 0:
            raise ValueError("Feature-cache inputs must include at least one timestep.")

        can_append = (
            feature_cache is not None
            and feature_cache.shape[0] == batch_size
            and feature_cache.shape[2] == self.transformer_model.config.vocab_size
            and feature_cache.shape[1] in {history_length - 1, history_length}
        )
        if can_append:
            latest_inputs = {
                key: value[:, -1:] if isinstance(value, torch.Tensor) else value for key, value in input_dict.items()
            }
            latest_inputs = self._prepare_inputs(latest_inputs, inject_noise=False)
            latest_features = self.extract_input_features(latest_inputs)
            if feature_cache.shape[1] == history_length:
                feature_cache = torch.cat((feature_cache[:, 1:], latest_features), dim=1)
            else:
                feature_cache = torch.cat((feature_cache, latest_features), dim=1)

            if rebuild_batch_indices is not None:
                rebuild_batch_indices = torch.as_tensor(
                    rebuild_batch_indices,
                    device=feature_cache.device,
                    dtype=torch.long,
                ).reshape(-1)
                if rebuild_batch_indices.numel() > 0:
                    rebuild_inputs = {
                        key: value.index_select(0, rebuild_batch_indices) if isinstance(value, torch.Tensor) else value
                        for key, value in input_dict.items()
                    }
                    rebuild_inputs = self._prepare_inputs(rebuild_inputs, inject_noise=False)
                    rebuilt_features = self.extract_input_features(rebuild_inputs)
                    feature_cache[rebuild_batch_indices] = rebuilt_features
        else:
            prepared_inputs = self._prepare_inputs(dict(input_dict), inject_noise=False)
            feature_cache = self.extract_input_features(prepared_inputs)

        features = self.transformer_model(feature_cache)
        output = self._process_output(self.model(features, deterministic=deterministic))
        return output[:, -1:, :], feature_cache

    def forward_cached(self, input_dict, cache=None, max_cache_length=None, deterministic=False):
        """Run one incremental Transformer inference step.

        Args:
            input_dict: Dictionary of latest-timestep input tensors, each shaped ``(B, 1, input_dim)``.
            cache: Cache returned by the previous call, or ``None`` to initialize one.
            max_cache_length: Maximum number of timesteps retained when initializing a cache.
            deterministic: Whether to use deterministic output mode.

        Returns:
            A tuple containing the prediction shaped ``(B, 1, output_dim)`` and updated KV cache.
        """
        if not self.is_transformer or self.is_rnn:
            raise RuntimeError("forward_cached() requires a Transformer model without an RNN.")

        input_dict = self._prepare_inputs(dict(input_dict), inject_noise=False)
        features = self.extract_input_features(input_dict)
        features, cache = self.transformer_model.forward_cached(features, cache, max_cache_length)
        output = self._process_output(self.model(features, deterministic=deterministic))
        return output, cache

    def init_transformer_cache(self, batch_size, max_cache_length=None):
        """Create an empty cache for :meth:`forward_cached`."""
        if not self.is_transformer:
            raise RuntimeError("Cannot create a Transformer cache for a non-Transformer model.")
        return self.transformer_model.init_kv_cache(
            batch_size,
            max_cache_length,
            device=self.device,
        )

    def reset_transformer_cache(self, cache, batch_indices=None):
        """Reset all cache rows or only selected batch rows."""
        cache.reset(batch_indices)

    def get_rnn_hidden_states(self):
        assert self.rnn is not None
        return self.rnn.get_hidden_states()

    def rnn_hidden_states_size(self):
        if self.is_rnn:
            return self.rnn.hidden_states_size()
        else:
            return None

    def to(self, device):
        for _, encoder in self.encoders.items():
            encoder.to(device)
        if self.rnn is not None:
            self.rnn.to(device)
        if self.transformer_model is not None:
            self.transformer_model.to(device)
        self.model.to(device)
        if self.predict_body_net_contact_forces:
            self.body_net_contact_force_head.to(device)

    def init_rnn(self, batch_size):
        if self.is_rnn:
            return self.rnn.initialize_hidden_states(batch_size)
        return None

    def reset_rnn_hidden_states(self, batch_indices=None, hidden_states=None):
        if self.is_rnn:
            self.rnn.reset_hidden_states(batch_indices, hidden_states)

    def reset(self, batch_size):
        self.init_rnn(batch_size)


# class ModelVectorInput(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         network_cfg,
#         device = 'cuda:0'
#     ):

#         self.device = device

#         self.input_rms = None
#         if network_cfg.get('input_rms', False):
#             self.input_rms = RunningMeanStd(
#                 shape = (input_dim, ),
#                 device = device
#             )

#         self.encoder, self.feature_dim =\
#             self.construct_input_encoder(
#                 network_cfg['encoder'],
#                 input_dim,
#                 device = device
#             )

#         self.model = MLPDeterministic(
#             (self.feature_dim, ),
#             output_dim,
#             network_cfg['model'],
#             device = device
#         )

#         self.output_tanh = network_cfg.get('output_tanh', False)

#     def construct_input_encoder(self,
#                                 encoder_cfg,
#                                 input_dim,
#                                 device = 'cuda:0'):

#         encoder = MLPBase(input_dim, encoder_cfg, device = device)

#         return encoder, encoder.out_features

#     def extract_input_features(self, x): # input can be in shape (B, input_dim) or (T, B, input_dim)
#         features = self.encoder(x)
#         return features

#     def evaluate(self, x, deterministic = False): # Single-step forward, input in shape (B, input_dim)
#         with torch.no_grad():
#             if hasattr(self, "input_rms") and self.input_rms is not None: # NOTE: remove the first condition later
#                 if self.training:
#                     self.input_rms.update(x, batch_dim = True, time_dim = False)
#                 x = self.input_rms.normalize(x)

#         features = self.extract_input_features(x)
#         if self.is_rnn:
#             features = self.rnn(features.unsqueeze(1)).squeeze(1)

#         output = self.model(features, deterministic = deterministic)

#         if self.output_tanh:
#             output = torch.tanh(output)

#         return output

#     def forward(self, input_dict, deterministic = False):
#         # Multi-step sequence forward, input in shape (T, B, input_dim)
#         with torch.no_grad():
#             input_dict = self.preprocess_input(input_dict)

#             if self.input_rms is not None:
#                 if self.training:
#                     self.input_rms.update(input_dict, batch_dim = True, time_dim = True)
#                 input_dict = self.input_rms.normalize(input_dict)

#         features = self.extract_input_features(input_dict) # (T, B, feature_dim)
#         if self.is_rnn:
#             features = self.rnn(features) # (T, B, rnn_hidden_dim)

#         T, B, feature_dim = features.shape[0], features.shape[1], features.shape[2]
#         features_flatten = features.view(-1, feature_dim)
#         output_flatten = self.model(features_flatten, deterministic = deterministic)
#         output = output_flatten.view(T, B, -1)

#         if self.output_tanh:
#             output = torch.tanh(output)

#         return output

#     def get_rnn_hidden_states(self):
#         assert self.rnn is not None
#         return self.rnn.get_hidden_states()

#     def rnn_hidden_states_size(self):
#         if self.is_rnn:
#             return self.rnn.hidden_states_size()
#         else:
#             return None

#     def to(self, device):
#         self.encoder.to(device)
#         if self.rnn is not None:
#             self.rnn.to(device)
#         self.model.to(device)

#     def init_rnn(self, batch_size):
#         if self.is_rnn:
#             return self.rnn.initialize_hidden_states(batch_size)

#     def reset_rnn_hidden_states(self, batch_indices=None, hidden_states=None):
#         if self.is_rnn:
#             self.rnn.reset_hidden_states(batch_indices, hidden_states)

#     def reset(self, batch_size):
#         self.init_rnn(batch_size)
