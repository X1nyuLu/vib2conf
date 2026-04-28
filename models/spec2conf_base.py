import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_base import SpectralEncoding, SpectralEncoder, AttentionalPooling
from .transformer_modules import MultiHeadedAttention, PositionwiseFeedForward, DecoderLayer, LayerNorm
from .transformer_modules import cl_loss, clones

class MatchingEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, nlayers=6, dropout=0.1, attn_fn=None):
        super().__init__()
        self_attn = attn_fn(nhead, d_model, dropout)
        src_attn = attn_fn(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_model*4, dropout)
        matching_layer = DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout)
        self.layers = clones(matching_layer, nlayers)
        self.norm = LayerNorm(d_model)

    def forward(self, input_embeds, memory=None, mask=None, src_mask=None, tgt_mask=None):
        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, memory, src_mask, tgt_mask)
        layer_output = self.norm(layer_output)
        return layer_output

class Spec2ConfBase(nn.Module):
    def __init__(self,
                 nheads=8, 
                 nlayers=6, 
                 encoder_layers=None,
                 pooling_layers=4,
                 matching_layers=2,
                 d_model=512, 
                 d_proj=256,
                 pooling_queries=128,
                 concat_spectrum=False,
                 use_matching_loss=False,
                 attn_fn=MultiHeadedAttention,
                 num_experts=1,
                 mask_ratio=0.0,
                 balance_loss_weight=0.0,
                 **kwargs
                 ):
        super().__init__()
        
        self.d_model = d_model
        self.attn_fn = attn_fn
        self.nheads = nheads
        
        self.use_matching_loss = use_matching_loss
        self.concat_spectrum = concat_spectrum
        self.pooling_layers = pooling_layers
        self.balance_loss_weight = balance_loss_weight
        self.mask_ratio = mask_ratio
        self.logit_scale = nn.Parameter(torch.rand([]))
        
        assert encoder_layers + pooling_layers == nlayers
        self.spectral_encoding = SpectralEncoding(d_model, concat_spectrum=concat_spectrum)
        self.spectral_encoder = SpectralEncoder(nheads=nheads, 
                                                nlayers=nlayers-pooling_layers,
                                                d_model=d_model,
                                                attn_fn=attn_fn)

        if pooling_layers > 0:
            self.attn_pool = AttentionalPooling(num_queries=pooling_queries, 
                                                d_model=d_model, 
                                                nheads=nheads, 
                                                nlayers=pooling_layers,
                                                use_cls_token=True, 
                                                attn_fn=attn_fn,
                                                num_experts=num_experts,)
        else:
            self.spectral_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.trunc_normal_(self.spectral_cls_token, std=0.02) # better initialization
            
        self.spectral_cls_proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.molecular_cls_proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
    
        self.molecular_encoder = None # please replace this with specific GNN
        self.matching_encoder = MatchingEncoder(d_model=d_model, nhead=nheads, nlayers=matching_layers, attn_fn=attn_fn) if use_matching_loss else None
        self.matching_head = nn.Sequential(nn.Linear(d_model, d_proj), nn.Tanh(), nn.Linear(d_proj, 2)) if use_matching_loss else None
        self.matching_token = nn.Parameter(torch.randn(1, d_model)) if use_matching_loss else None
            
               
    def get_spectral_embedding(self, inputs):
        ref_tensor = inputs.get('raman', inputs.get('ir'))
        batch_size, seq_len = ref_tensor.shape[0], ref_tensor.shape[-1]
        
        if not self.concat_spectrum:
            spectral_inputs = ref_tensor.unsqueeze(1)
        else:
            spectral_inputs = torch.stack([inputs.get('raman'), inputs.get('ir')], dim=1)        
            if self.training:
                # 以 self.mask_ratio 的概率在训练后期随机关掉一个模态
                mask = torch.rand(1)
                if mask < self.mask_ratio / 2: # 只给 Raman
                    spectral_inputs[:, 1, :] = 0
                elif mask < self.mask_ratio: # 只给 IR
                    spectral_inputs[:, 0, :] = 0
            
        spectral_embeds = self.spectral_encoding(spectral_inputs)
        
        if self.pooling_layers > 0:
            spectral_mask = torch.ones((batch_size, self.spectral_encoding.num_tokens), dtype=spectral_inputs.dtype, device=spectral_inputs.device)
        else:
            cls_tokens = self.spectral_cls_token.expand(batch_size, -1, -1)
            spectral_embeds = torch.cat([cls_tokens, spectral_embeds], dim=1)
            spectral_mask = torch.ones((batch_size, spectral_embeds.size(1)), dtype=spectral_inputs.dtype, device=spectral_inputs.device)
            
        spectral_outputs = self.spectral_encoder(spectral_embeds, spectral_mask)  
        return spectral_outputs, spectral_mask
    
    def get_molecular_embedding(self, inputs, return_node_features):
        outputs = self.molecular_encoder(pos=inputs.pos, batch=inputs.batch, z=inputs.x, edge_index=inputs.edge_index, return_node_features=return_node_features)
        return outputs
    
    def compute_cl_loss(self, molecular_output, spectral_output, return_sim=False):
        molecular_output = F.normalize(molecular_output, p=2, dim=1)
        spectral_output = F.normalize(spectral_output, p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        logits_per_smiles = torch.matmul(
            molecular_output, spectral_output.t()) * logit_scale
        logits_per_spectrum = logits_per_smiles.T
        loss = cl_loss(logits_per_spectrum)
        
        if return_sim:
            return loss, logits_per_smiles, logits_per_spectrum
        else:
            return loss
        
    
    def forward(self):
        raise TypeError('employ _forward method')
    
    def _forward(self,
                inputs,
                return_loss=True,
                return_proj_output=False
                ):
        
        spectral_embeds, spectral_mask = self.get_spectral_embedding(inputs)
        
        if self.pooling_layers > 0:
            spectral_outputs = self.attn_pool(spectral_embeds, src_mask=spectral_mask)
        
        spectral_cls_token = spectral_outputs[:, 0]
        spectral_contra_token = self.spectral_cls_proj(spectral_cls_token)

        molecular_outputs = self.get_molecular_embedding(inputs, return_node_features=self.use_matching_loss)
        
        if self.use_matching_loss:
            molecular_cls_token, molecular_outputs, molecular_mask = molecular_outputs
        else:
            molecular_cls_token = molecular_outputs
            
        molecular_contra_token = self.molecular_cls_proj(molecular_cls_token)
        
        result_dict = {}
        loss = torch.tensor(0, device=spectral_cls_token.device, dtype=spectral_cls_token.dtype)
        
        cl_loss, sim_m2s, sim_s2m = self.compute_cl_loss(molecular_contra_token, spectral_contra_token, return_sim=True)
        result_dict['cl_loss'] = cl_loss
        loss += cl_loss
        
        total_aux_loss = sum(m.aux_loss for m in self.modules() if hasattr(m, 'aux_loss'))
        
        if isinstance(total_aux_loss, torch.Tensor):
            result_dict['aux_loss'] = total_aux_loss
            loss += self.balance_loss_weight * total_aux_loss

        if return_loss:
            result_dict['loss'] = loss
        
        if return_proj_output:
            result_dict['molecular_proj_output'] = molecular_contra_token
            result_dict['spectral_proj_output'] = spectral_contra_token
            result_dict['spectral_embeds'] = spectral_embeds
            result_dict['spectral_outputs'] = spectral_outputs

        return result_dict
    
    
    def matching(self, inputs):
        
        spectral_outputs, spectral_mask = self.get_spectral_embedding(inputs)
        spectral_outputs = self.attn_pool(spectral_outputs, src_mask=spectral_mask)
        spectral_attention_mask_with_cls = torch.ones(size=(spectral_outputs.size(0), self.attn_pool.num_queries), device=spectral_outputs.device)

        molecular_outputs = self.get_molecular_embedding(inputs, return_node_features=True)
        _, molecular_outputs, molecular_mask = molecular_outputs
        molecular_attention_mask_with_cls = torch.cat([torch.ones(size=(molecular_mask.size(0), 1), device=molecular_mask.device), molecular_mask], dim=1)
        molecular_embeds = torch.cat([self.matching_token.repeat(molecular_mask.size(0), 1).unsqueeze(1), molecular_outputs], dim=1)
        
        multimodal_outputs = self.matching_encoder(molecular_embeds, 
                                                     memory=spectral_outputs,  
                                                     src_mask=spectral_attention_mask_with_cls,
                                                     tgt_mask=molecular_attention_mask_with_cls,
                                                     )
        
        matching_outputs = self.matching_head(multimodal_outputs[:, 0, :])
        return matching_outputs