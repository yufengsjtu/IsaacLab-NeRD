# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE.md for details].

import torch
import torch.nn as nn
from isaaclab_neural.models.base_models import MLPBase, CNNBase, LSTMBase, GRUBase
from isaaclab_neural.models.model_kan import KAN
from isaaclab_neural.models.model_transformer import GPT, GPTConfig
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
import numpy as np

class MLPDeterministic(nn.Module):
    def __init__(self,
                input_shape,
                output_dim,
                network_cfg,
                device='cuda:0'):
        super().__init__()
        
        self.device = device

        self.feature_net = MLPBase(
            input_shape[0], 
            network_cfg['mlp'], 
            device=device
        )

        self.output_net = nn.Linear(
            self.feature_net.out_features, 
            output_dim, 
            device=device
        )
    
    def forward(self, inputs, deterministic = False):
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
        device = 'cuda:0'
    ):
        
        super().__init__()

        self.device = device
        self.model = None
        
        self.input_rms = None
        self.normalize_input = network_cfg.get('normalize_input', False)
        self.output_rms = None
        self.normalize_output = network_cfg.get('normalize_output', False)

        self.encoders, self.feature_dim = self.construct_input_encoders(
            input_cfg, 
            network_cfg['encoder'], 
            input_sample, 
            device = device
        )
        
        if "rnn" in network_cfg:
            self.is_rnn = True
            if network_cfg['rnn']['net'] == 'lstm':
                self.rnn = LSTMBase(
                    self.feature_dim, 
                    network_cfg['rnn'], 
                    device=self.device
                )
            elif network_cfg['rnn']['net'] == 'gru':
                self.rnn = GRUBase(
                    self.feature_dim, 
                    network_cfg['rnn'], 
                    device=self.device
                )
            else:
                raise NotImplementedError
            self.feature_dim = self.rnn.hidden_size
        else:
            self.is_rnn = False
            self.rnn = None

        if "transformer" in network_cfg:
            model_args = dict(
                n_layer=network_cfg['transformer']['n_layer'],
                n_head=network_cfg['transformer']['n_head'],
                n_embd=network_cfg['transformer']['n_embd'],
                block_size=network_cfg['transformer']['block_size'],
                bias=network_cfg['transformer']['bias'],
                vocab_size=self.feature_dim,
                dropout=network_cfg['transformer']['dropout'],
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
                layers_hidden=[self.feature_dim] + network_cfg['kan']['layer_sizes'] + [output_dim],
                grid_num=network_cfg['kan'].get('grid_num', 5),
                order=network_cfg['kan'].get('order', 3),
                scale_noise=network_cfg['kan'].get('scale_noise', 0.1),
                scale_base=network_cfg['kan'].get('scale_base', 1.0),
                scale_spline=network_cfg['kan'].get('scale_spline', 1.0),
                enable_standalone_scale_spline=network_cfg['kan'].get(
                    'enable_standalone_scale_spline', True
                ),
                base_activation=torch.nn.SiLU,
                grid_range=network_cfg['kan'].get('grid_range', [-1, 1]),
            )
            self.model.to(self.device)
            self.is_kan = True
        else:
            self.is_kan = False

        if self.model is None:
            self.model = MLPDeterministic(
                (self.feature_dim, ), 
                output_dim, 
                network_cfg['model'], 
                device = device
            )
        
        self.output_tanh = network_cfg.get('output_tanh', False)
    
    def construct_input_encoders(
        self,
        input_cfg,
        encoder_cfg,
        input_sample,
        device = 'cuda:0'
    ):
        encoders = nn.ModuleDict()
        
        '''
        low-dim inputs
        '''
        if len(input_cfg.get('low_dim', [])) > 0:
            low_dim_size = 0
            self.low_dim_input_names = input_cfg.get('low_dim')
            for low_dim_input_name in self.low_dim_input_names:
                assert len(input_sample[low_dim_input_name].shape) in [2, 3] # (B, *) or (B, T, *)
                low_dim_size += input_sample[low_dim_input_name].shape[-1]
            
            assert 'low_dim' in encoder_cfg
            low_dim_encoder = MLPBase(
                low_dim_size, 
                encoder_cfg['low_dim'], 
                device = device
            )
            encoders['low_dim'] = low_dim_encoder
            
        '''
        rgb inputs
        '''
        rgb_input_names = input_cfg.get('rgb', [])
        for rgb_input_name in rgb_input_names:
            assert rgb_input_name in input_sample
            assert len(input_sample[rgb_input_name].shape) in [4, 5] # (B, C, H, W) or (B, T, C, H, W)
            assert 'rgb' in encoder_cfg
            if rgb_input_name in encoder_cfg['rgb']:
                config = encoder_cfg['rgb'][rgb_input_name]
            elif 'default' in encoder_cfg['rgb']:
                config = encoder_cfg['rgb']['default']
            else:
                raise ValueError(
                    f"No '{rgb_input_name}' nor 'default' in encoder_cfg['rgb']"
                )

            rgb_encoder = CNNBase(
                input_sample[rgb_input_name].shape[1:], 
                config, 
                device = device
            )
            encoders[rgb_input_name] = rgb_encoder
        
        feature_dim = 0
        for input_name in encoders:
            feature_dim += encoders[input_name].out_features

        return encoders, feature_dim

    def set_input_rms(self, data_rms):
        rms_dict = {}
        for input_name in self.encoders:
            if input_name == 'low_dim':
                for low_dim_input_name in self.low_dim_input_names:
                    if low_dim_input_name in data_rms:
                        rms_dict[low_dim_input_name] = data_rms[low_dim_input_name]
            else:
                rms_dict[input_name] = data_rms[input_name]
        self.input_rms = nn.ModuleDict(rms_dict)

    def set_output_rms(self, output_rms):
        self.output_rms = output_rms

    def get_rms_info(self):
        """Return RMS keys and shapes for checkpoint reconstruction."""
        info = {}
        if self.input_rms is not None:
            info['input_rms'] = {
                k: tuple(v.mean.shape) for k, v in self.input_rms.items()
            }
        if self.output_rms is not None:
            info['output_rms'] = tuple(self.output_rms.mean.shape)
        return info

    def extract_input_features(self, input_dict): # input can be in shape (B, input_dim) or (B, T, input_dim)
        features = []
        for input_name in self.encoders:
            if input_name == 'low_dim':
                low_dim_input_list = []
                for low_dim_input_name in self.low_dim_input_names:
                    low_dim_input_list.append(input_dict[low_dim_input_name])
                cur_input = torch.cat(low_dim_input_list, dim = -1)
            else:
                cur_input = input_dict[input_name]
            features.append(self.encoders[input_name](cur_input)) # each feature is (B, (T), feature_dim_i)
        features = torch.cat(features, dim = -1)
        return features

    def forward(
        self, 
        input_dict, 
        single_step = False,
        deterministic = False, 
        inject_noise = False
    ): 
        """
        Forward pass through the model.
        
        Args:
            input_dict: Dictionary of input tensors, each shape (B, T, input_dim)
            single_step: If True, returns only the last timestep prediction (B, 1, output_dim)
                        If False, returns all timesteps (B, T, output_dim)
            deterministic: Whether to use deterministic mode (unused in current implementation)
            inject_noise: If True, adds Gaussian noise to inputs for data augmentation
        
        Returns:
            Tensor of shape (B, 1, output_dim) if single_step=True, else (B, T, output_dim)
        """
        
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        if inject_noise:
            for obs_key in input_dict.keys():
                input_dict[obs_key] = (
                    input_dict[obs_key] + 
                    torch.randn_like(input_dict[obs_key]) * 0.01
                )

        features = self.extract_input_features(input_dict) # (B, T, feature_dim)

        if self.is_rnn:
            features = self.rnn(features) # (B, T, rnn_hidden_dim)

        if self.is_transformer:
            features = self.transformer_model(features) # (B, T, transform_embed_dim)

        output = self.model(features, deterministic = deterministic)

        if self.output_tanh:
            output = torch.tanh(output)

        if self.normalize_output:
            output = self.output_rms.normalize(output, un_norm = True)
        
        if single_step:
            output = output[:, -1:, :]
        
        return output

    def get_rnn_hidden_states(self):
        assert self.rnn is not None
        return self.rnn.get_hidden_states()
    
    def rnn_hidden_states_size(self):
        if self.is_rnn:
            return self.rnn.hidden_states_size()
        else:
            return None
        
    def to(self, device):
        for (_, encoder) in self.encoders.items():
            encoder.to(device)
        if self.rnn is not None:
            self.rnn.to(device)
        self.model.to(device)

    def init_rnn(self, batch_size):
        if self.is_rnn:
            return self.rnn.initialize_hidden_states(batch_size)
    
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

#     def forward(self, input_dict, deterministic = False): # Multi-step sequence forward, input in shape (T, B, input_dim)
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