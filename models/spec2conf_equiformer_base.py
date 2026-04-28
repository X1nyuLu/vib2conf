import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .equiformer_base import equiformer_base
from .spec2conf_base import Spec2ConfBase


class Spec2ConfEquiformerBase(Spec2ConfBase):
    def __init__(self,
                 use_matching_loss=False,
                 encoder_layers=2,
                 pooling_layers=4,
                 pooling_queries=128,
                 num_experts=1,
                 **kwargs
                 ):
        super().__init__(pooling_queries=pooling_queries, 
                         encoder_layers=encoder_layers, 
                         pooling_layers=pooling_layers, 
                         use_matching_loss=use_matching_loss,
                         num_experts=num_experts,)
        
        self.molecular_encoder = equiformer_base(irreps_feature=f"{self.d_model}x0e")
        
    def forward(self,
                inputs,
                return_loss=True,
                return_proj_output=False
                ):
        
        result_dict = self._forward(inputs, return_loss, return_proj_output)
        return result_dict

    
@register_model
def spec2conf_equiformer_base(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(**kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool1(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=5, pooling_layers=1, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool2(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=4, pooling_layers=2, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool3(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=3, pooling_layers=3, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool4(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool5(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=1, pooling_layers=5, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool4_sample256(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=256, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool4_sample64(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=64, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool4_sample32(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=32, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool4_sample16(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=16, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_pool4_sample8(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, pooling_queries=8, **kwargs)
    return model

@register_model
def spec2conf_equiformer_base_cls(pretrained=False, **kwargs):
    model = Spec2ConfEquiformerBase(encoder_layers=2, pooling_layers=4, **kwargs)
    return model

