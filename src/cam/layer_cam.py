"""Layer-CAM (Jiang et al., IEEE TIP 2021)."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import unwrap_model


class LayerCAM:
    """Layer-CAM heatmaps from pointwise positive gradients at one target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, chunk_size: int = 32):
        self.raw_model = unwrap_model(model)
        self.target_layer = target_layer
        self.chunk_size = max(1, int(chunk_size))
        self.activations: Optional[torch.Tensor] = None
        self.enabled = False
        self.handlers = [target_layer.register_forward_hook(self._store)]

    def _store(self, module, inputs, output):
        if self.enabled:
            self.activations = output

    def remove_hooks(self) -> None:
        for h in self.handlers:
            h.remove()
        self.handlers.clear()
        self.activations = None
        self.enabled = False

    def __del__(self):
        self.remove_hooks()

    def _generate_chunk(self, input_tensor: torch.Tensor, target_class: Optional[torch.Tensor] = None) -> torch.Tensor:
        was_training = self.raw_model.training
        self.raw_model.eval()
        h, w = input_tensor.shape[-2:]
        with torch.enable_grad():
            # the input carries the graph even when no parameter requires grad
            x = input_tensor.clone().detach().requires_grad_(True)
            self.enabled = True
            logits = self.raw_model(x)
            self.enabled = False
            acts = self.activations
            self.activations = None
            if target_class is None:
                target_class = logits.argmax(dim=1)
            one_hot = torch.zeros_like(logits).scatter_(1, target_class.view(-1, 1), 1.0)
            # gradient w.r.t. the activation only: no weight gradients, nothing below the layer
            grads, = torch.autograd.grad(logits, acts, grad_outputs=one_hot)
        cam = F.relu((F.relu(grads) * acts.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        diff = cam.amax(dim=(2, 3), keepdim=True) - cam_min
        cam = torch.where(diff > 1e-8, (cam - cam_min) / (diff + 1e-8), torch.zeros_like(cam))
        self.raw_model.train(was_training)
        return torch.nan_to_num(cam.detach(), nan=0.0, posinf=1.0, neginf=0.0)

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Heatmaps [B, 1, H, W] in [0, 1]."""
        cams = []
        for s in range(0, input_tensor.shape[0], self.chunk_size):
            sub_target = target_class[s:s + self.chunk_size] if target_class is not None else None
            cams.append(self._generate_chunk(input_tensor[s:s + self.chunk_size], sub_target))
        return torch.cat(cams, dim=0)
