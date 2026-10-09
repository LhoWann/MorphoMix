"""ConvNeXt V2 Atto (timm `convnextv2_atto.fcmae_ft_in1k`), the only backbone this repo builds."""
import timm
import torch.nn as nn

try:
    import huggingface_hub.utils.logging as hlog
    hlog.set_verbosity_error()
except Exception:
    pass

TIMM_ID = "convnextv2_atto.fcmae_ft_in1k"


def build_model(
        model_name: str = "convnextv2_atto",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        head_init_scale: float = 1.0,
) -> nn.Module:
    """ConvNeXt V2 Atto with a fresh `num_classes` head; any other name is refused.

    `drop_path` (stochastic depth) and `head_init_scale` only matter for training a fresh head; models that load
    a checkpoint for evaluation can keep the defaults.
    """
    if model_name.lower().replace("-", "_") != "convnextv2_atto":
        raise ValueError(f"Unsupported model name {model_name!r}: MorphoMix is fixed to 'convnextv2_atto'.")
    return timm.create_model(TIMM_ID, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout,
                             drop_path_rate=drop_path, head_init_scale=head_init_scale)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def get_target_cam_layer(model: nn.Module, stage_idx: int = -3) -> nn.Module:
    """Last block of `stages[stage_idx]` (-3 = Stage 2, 80 x 28 x 28)."""
    return unwrap_model(model).stages[stage_idx].blocks[-1]
