"""AdamW with layer-wise lr decay (ConvNeXt V2, Woo et al., CVPR 2023, Table 9), warm-up and cosine decay."""
import math
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from timm.optim import param_groups_layer_decay


def setup_optimizer(
    model: nn.Module,
    lr: float = 1.0e-4,
    weight_decay: float = 0.05,
    layer_decay: float = 0.9,
    total_epochs: int = 25,
    warmup_epochs: int = 2
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW and a per-epoch LambdaLR: warm-up 0.2 + 0.8 * (epoch + 1) / warmup_epochs (36 % at epoch 0 for 5
    warm-up epochs), then cosine down to a 2 % floor at the last epoch.

    `lr` is the peak rate of the head; each earlier layer gets `layer_decay` times the rate of the layer after it
    (stem = 0.9^13 of the head for Atto). Biases and norm parameters get no weight decay.
    """
    groups = param_groups_layer_decay(model, weight_decay=weight_decay, layer_decay=layer_decay)
    for g in groups:
        g["lr"] = g["initial_lr"] = lr * g.pop("lr_scale")
    optimizer = AdamW(groups, betas=(0.9, 0.999))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return 0.2 + 0.8 * float(epoch + 1) / float(warmup_epochs)

        eff_epoch = epoch - warmup_epochs
        eff_total = max(1, total_epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, eff_epoch / float(eff_total)))
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = 0.02
        return max(floor, floor + (1.0 - floor) * decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(groups))
    return optimizer, scheduler
