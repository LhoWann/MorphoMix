"""Train/val transforms, the batch-level resolution degradation and ImageNet (de)normalisation."""
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224) -> v2.Compose:
    """Resize, flips, rotation up to 180 degrees and mild colour jitter."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=180, interpolation=v2.InterpolationMode.BILINEAR),  # default NEAREST aliases
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


def get_val_transforms(img_size: int = 224) -> v2.Compose:
    """Resize and normalise only."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


@torch.no_grad()
def degrade_resolution(x: torch.Tensor, prob: float, scale: Tuple[float, float]) -> torch.Tensor:
    """Low-resolution capture of the normalised GPU batch, for every arm: each sample is selected with `prob`, area
    downsampled to s * its side (s ~ U[scale], side rounded to 8 px so a batch needs few resampling calls) and
    bilinearly upsampled back. It mimics a small cell (Aria: about 27 px at 224, against about 100 px for a C-NMC cell)
    cropped from a field and resized to the training size. Both resamplings are linear, so they commute with the
    ImageNet normalisation. Two draws of B values per batch, whatever the data, keep the RNG stream fixed."""
    b = x.shape[0]
    selected = torch.rand(b, device=x.device) < prob
    s = scale[0] + (scale[1] - scale[0]) * torch.rand(b, device=x.device)
    return resample_down_up(x, selected, s)


@torch.no_grad()
def resample_down_up(x: torch.Tensor, selected: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The selected samples area-downsampled to s x their side (rounded to 8 px) and bilinearly upsampled back; the
    core of `degrade_resolution` and of MorphoMix C2' (low-detail cell). No RNG draw."""
    _, _, h, w = x.shape
    sides = ((s * h / 8).round() * 8).clamp(min=8).long()
    out = x.clone()
    for side in sorted(set(sides[selected].tolist())):
        idx = (selected & (sides == side)).nonzero().squeeze(1)
        small = F.interpolate(x[idx], size=(side, max(1, round(side * w / h))), mode="area")
        out[idx] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return out


def denormalize_batch(tensor: torch.Tensor) -> np.ndarray:
    """[B, 3, H, W] normalized tensor -> [B, H, W, 3] uint8."""
    x = tensor.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    return np.clip(np.rint((x * std + mean) * 255.0), 0, 255).astype(np.uint8)


def denormalize_image(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """One normalised image [3, H, W] (or [1, 3, H, W]) -> uint8 RGB [H, W, 3]."""
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = np.array(tensor, copy=True)

    if img.ndim == 4:
        img = img[0]

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    img = img * std + mean
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
