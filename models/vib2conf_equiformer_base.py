import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .equiformer_base import equiformer_base
from .vib2conf_base import Vib2ConfBase


class Vibc2ConfEquiformerBase(Vib2ConfBase):
    def __init__(self,
                 use_matching_loss=False,
                 encoder_layers=2,
                 pooling_layers=4,
                 pooling_queries=128,
                 num_experts=1,
                 concat_spectrum=False,
                 balance_loss_weight=0.0,
                 mask_ratio=0.0,
                 gpu_align=False,
                 **kwargs
                 ):
        super().__init__(pooling_queries=pooling_queries,
                         encoder_layers=encoder_layers,
                         pooling_layers=pooling_layers,
                         use_matching_loss=use_matching_loss,
                         num_experts=num_experts,
                         concat_spectrum=concat_spectrum,
                         balance_loss_weight=balance_loss_weight,
                         mask_ratio=mask_ratio,
                         gpu_align=gpu_align,
                         **kwargs)
        
        self.molecular_encoder = equiformer_base(irreps_feature=f"{self.d_model}x0e")
        
    def forward(self,
                inputs,
                return_loss=True,
                return_proj_output=False
                ):
        
        result_dict = self._forward(inputs, return_loss, return_proj_output)
        return result_dict

    
@register_model
def vib2conf_equiformer_base(**kwargs):
    model = Vibc2ConfEquiformerBase(**kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool1(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=5, pooling_layers=1, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool2(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=4, pooling_layers=2, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool3(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=3, pooling_layers=3, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool4(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool5(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=1, pooling_layers=5, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool4_sample256(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=256, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool4_sample64(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=64, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool4_sample32(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=32, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool4_sample16(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=16, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_pool4_sample8(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=8, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_cls(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, **kwargs)
    return model


# MoE for spectral encoding
@register_model
def vib2conf_equiformer_base_moe2(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, num_experts=2, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_moe3(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, num_experts=3, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_moe4(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, num_experts=4, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_moe5(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, num_experts=5, **kwargs)
    return model

@register_model
def vib2conf_equiformer_base_moe6(**kwargs):
    model = Vibc2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, num_experts=6, **kwargs)
    return model