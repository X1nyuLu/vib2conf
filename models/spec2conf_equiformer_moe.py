import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .equiformer_moe import equiformer_moe
from .spec2conf_base import Spec2ConfBase


class Spec2ConfEquiformerMoE(Spec2ConfBase):
    def __init__(self,
                 use_matching_loss=False,
                 encoder_layers=2,
                 pooling_layers=4,
                 pooling_queries=64,
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
                         gpu_align=gpu_align)
        
        self.molecular_encoder = equiformer_moe(irreps_feature=f"{self.d_model}x0e", num_experts=num_experts)
        
    def forward(self,
                inputs,
                return_loss=True,
                return_proj_output=False
                ):
        
        result_dict = self._forward(inputs, return_loss, return_proj_output)
        return result_dict


# MoE for molecular encoding
@register_model
def spec2conf_equiformer_moe2(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=2, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe3(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe4(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=4, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe5(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=5, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe6(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=6, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_balance01(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, balance_loss_weight=0.1, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_balance001(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, balance_loss_weight=0.01, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_balance0001(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, balance_loss_weight=0.001, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_balance00001(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, balance_loss_weight=0.0001, **kwargs)
    return model

# Best setting for single modality
@register_model
def spec2conf_equiformer_moe_balance0001_align(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, balance_loss_weight=0.001, gpu_align=True, **kwargs)
    return model


@register_model
def spec2conf_equiformer_moe_concat(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance01(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.1, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance001(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.01, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance0001(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance00001(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.0001, **kwargs)
    return model

# Best setting for double modality
@register_model
def spec2conf_equiformer_moe_concat_balance0001_align(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, gpu_align=True, **kwargs)
    return model


@register_model
def spec2conf_equiformer_moe_concat_balance0001_mask15(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, mask_ratio=0.15, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance0001_mask30(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, mask_ratio=0.30, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance0001_mask45(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, mask_ratio=0.45, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance0001_mask60(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, mask_ratio=0.60, **kwargs)
    return model

@register_model
def spec2conf_equiformer_moe_concat_balance0001_mask75(**kwargs):
    model = Spec2ConfEquiformerMoE(num_experts=3, concat_spectrum=True, balance_loss_weight=0.001, mask_ratio=0.75, **kwargs)
    return model