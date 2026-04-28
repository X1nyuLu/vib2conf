import sys 

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_modules import (LayerNorm, 
                                  SublayerConnection, 
                                  PositionwiseFeedForward, 
                                  ContinuousSpectralEncoding)
from .transformer_modules import clones

class SpectralEncoding(nn.Module):
    def __init__(self,
                 d_model=512,
                 max_pos_len=256,
                 continuous_max_len=1024,
                 continuous_patch_size=8,
                 no_bias=False,
                 concat_spectrum=False,
                 ):
        super().__init__()
 
        self.spectral_encoding = ContinuousSpectralEncoding(continuous_max_len, continuous_patch_size, d_model, no_bias, concat_spectrum=concat_spectrum)
        self.spectral_pos_embedding = nn.Embedding(max_pos_len, d_model)
        self.num_tokens = self.spectral_encoding.num_tokens    
        
    def forward(self, inputs):
        spectral_embeds = self.spectral_encoding(inputs)
        spectral_pos_embeds = self.spectral_pos_embedding(torch.arange(spectral_embeds.size(1)).to(spectral_embeds.device))
        return spectral_embeds + spectral_pos_embeds

class SpectralEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.ffn = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.ffn)
    
class SpectralEncoder(nn.Module):
    def __init__(self, nheads=8, nlayers=6, d_model=512, dropout=0.1, attn_fn=False):
        super().__init__()
        self_attn = attn_fn(nheads, d_model)
        ffn = PositionwiseFeedForward(d_model, d_model*4, dropout)
        layer = SpectralEncoderLayer(d_model, self_attn, ffn, dropout)
        self.layers = clones(layer, nlayers)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class AttentionalPoolingLayer(nn.Module):
    def __init__(self,
                 self_attn,
                 src_attn,
                 ffn,
                 d_model,
                 dropout=0.1,
                 num_experts=1,
                 ):
        super().__init__()
        
        self.num_experts = num_experts
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        
        if num_experts > 1:
            self.ffns = nn.ModuleList([copy.deepcopy(ffn) for _ in range(num_experts)])
            self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_experts))
        else:
            self.feed_forward = ffn

    def forward(self, x, memory, src_mask=None):
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x, None))
        
        # MoE
        def ffn_process(input_tensor):
            gate_logits = self.gate(input_tensor)
            weights = F.softmax(gate_logits, dim=-1)
            
            expert_outputs = torch.stack([ffn(input_tensor) for ffn in self.ffns], dim=0)
            weights = weights.permute(2, 0, 1).unsqueeze(-1)
            # weighted addition (dim=0)
            combined_output = torch.sum(weights * expert_outputs, dim=0)
            return combined_output
        
        if self.num_experts > 1:
            return self.sublayer[2](x, ffn_process)
        else:
            return self.sublayer[2](x, self.feed_forward)
    
class AttentionalPooling(nn.Module):
    def __init__(self,
                 num_queries,
                 d_model,
                 nheads=8,
                 dropout=0.1,
                 nlayers=3,
                 use_cls_token=False,
                 attn_fn=None,
                 num_experts=1):
        super().__init__()
        # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.num_queries = num_queries + 1 if use_cls_token else num_queries
        self.queries = nn.Parameter(torch.randn(self.num_queries, d_model)) 
        
        self_attn = attn_fn(nheads, d_model)
        src_attn = attn_fn(nheads, d_model)
        ffn = PositionwiseFeedForward(d_model, d_model*4, dropout)
        attn_pooling_layer = AttentionalPoolingLayer(self_attn, src_attn, ffn, d_model, dropout, num_experts)
        self.layers = clones(attn_pooling_layer, nlayers)
        self.norm = LayerNorm(d_model)
        
    def forward(self, memory, src_mask=None):
        
        batch_size = memory.shape[0]
        queries = self.queries.repeat(batch_size, 1).view(batch_size, -1, memory.size(-1))
        
        for layer in self.layers:
            queries = layer(queries, memory, src_mask)
        return self.norm(queries)
