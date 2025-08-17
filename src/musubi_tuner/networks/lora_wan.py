# LoRA module for Wan2.1

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]


class WanLoRANetwork(lora.LoRANetwork):
    """WAN-specific LoRA network with bulk dtype conversion optimization."""
    
    def prepare_grad_etc(self, unet):
        """Override to add bulk dtype conversion for WAN networks."""
        super().prepare_grad_etc(unet)
        
        # Perform bulk dtype conversion to avoid per-layer conversions during training
        if hasattr(unet, 'dtype') and unet.dtype != torch.float32:
            target_dtype = unet.dtype
            logger.info(f"SWAP: Converting LoRA network to dtype {target_dtype} to avoid per-layer conversions")
            self.to(target_dtype)
            logger.info(f"SWAP: LoRA network conversion to {target_dtype} completed")


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'img_mod', 'txt_mod' or 'modulation' in the name
    exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")

    kwargs["exclude_patterns"] = exclude_patterns

    # Create the base network first
    base_network = lora.create_network(
        WAN_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
    
    # Create our custom WAN network with the same parameters
    wan_network = WanLoRANetwork(
        WAN_TARGET_REPLACE_MODULES,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=kwargs.get("rank_dropout", None),
        module_dropout=kwargs.get("module_dropout", None),
        conv_lora_dim=kwargs.get("conv_dim", None),
        conv_alpha=kwargs.get("conv_alpha", None),
        exclude_patterns=exclude_patterns,
        include_patterns=kwargs.get("include_patterns", None),
        verbose=kwargs.get("verbose", False),
    )
    
    # Set LoRA+ ratio if specified
    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    if loraplus_lr_ratio is not None:
        wan_network.set_loraplus_lr_ratio(float(loraplus_lr_ratio))
    
    return wan_network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        WAN_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
