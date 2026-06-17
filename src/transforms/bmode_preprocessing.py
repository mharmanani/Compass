"""B-mode ultrasound image preprocessing and augmentation.

Handles loading from file paths or numpy arrays, converting to float tensors,
normalizing, resizing, and optional augmentations.

Uses shared helpers from ``sweep_preprocessing`` for input normalisation so
that sweep and single-frame paths stay in sync.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask

from src.transforms.sweep_preprocessing import (
    ensure_3d_float_tensor,
    _minmax_normalize,
)


@dataclass
class PreprocessBmode:
    """Preprocess a single B-mode frame (or a batch of frames).

    Converts a raw image (file path, numpy array, or tensor) to a normalised
    ``(C, H, W)`` float tensor suitable for model input.

    The pipeline is:
        1. Load (file path → ndarray if needed)
        2. Convert to ``(C, H, W)`` float tensor (int→float /255, expand dims,
           repeat channels).  Uses :func:`ensure_3d_float_tensor`.
        3. Optional per-image min-max normalisation to [0, 1].
        4. ``T.Normalize(mean, std)`` — default ``(1,1,1)/(1,1,1)`` is a no-op.
        5. Resize.
        6. Augmentations (if ``train``).

    Args:
        image_size: Resize target ``(H, W)`` or a single int for square.
        minmax_normalize: Scale to [0, 1] using per-image min/max.
        mean: Per-channel mean for ``T.Normalize``. Default is a no-op.
        std: Per-channel std for ``T.Normalize``. Default is a no-op.
        num_channels: Target channel count (e.g. 3 to repeat greyscale).
        train: Enable stochastic augmentations.
        rand_resize_crop: True (defaults) or dict of extra kwargs.  None = off.
        rand_affine: True (defaults) or dict of extra kwargs.  None = off.
        rand_gamma: Apply random gamma augmentation.
        rand_contrast: Apply random contrast augmentation.
    """

    image_size: int | tuple[int, int] = 1024
    minmax_normalize: bool = True
    mean: tuple = (0, 0, 0)
    std: tuple = (1, 1, 1)
    num_channels: int = 3
    train: bool = True

    # augmentations
    rand_resize_crop: Optional[bool | dict] = None
    rand_affine: Optional[bool | dict] = None
    rand_gamma: bool = False
    rand_contrast: bool = False

    def __call__(self, bmode, mask: Optional[torch.Tensor] = None):
        """Process a single B-mode image, optionally with a paired mask.

        Args:
            bmode: File path (str), numpy array ``(H, W)`` or ``(H, W, C)``,
                or torch tensor.
            mask: Optional ``(1, H, W)`` float tensor in ``[0, 1]``.  Spatial
                augmentations are applied jointly so image and mask stay
                synchronized.  Pixel-level augmentations are not applied to
                the mask.

        Returns:
            ``(C, H, W)`` tensor, or ``((C, H, W), (1, H, W))`` tuple when
            *mask* is provided.
        """
        bmode = ensure_3d_float_tensor(bmode, num_channels=self.num_channels)
        if self.minmax_normalize:
            bmode = _minmax_normalize(bmode)
        bmode = T.Normalize(self.mean, self.std)(bmode)
        bmode = self._resize(bmode)
        if mask is not None:
            mask = self._resize_mask(mask)
        bmode, mask = self._augment(bmode, mask)
        if mask is not None:
            return bmode, mask
        return bmode

    # --- internals -----------------------------------------------------------

    def _resize(self, bmode: torch.Tensor) -> torch.Tensor:
        size = (
            (self.image_size, self.image_size)
            if isinstance(self.image_size, int)
            else self.image_size
        )
        return T.Resize(size, antialias=True)(bmode)

    def _resize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        size = (
            (self.image_size, self.image_size)
            if isinstance(self.image_size, int)
            else self.image_size
        )
        return T.Resize(
            size, antialias=False, interpolation=InterpolationMode.NEAREST
        )(mask)

    def _augment(self, bmode: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if not self.train:
            return bmode, mask

        # Spatial augments: apply jointly to image + mask so they stay in sync.
        # Wrap in tv_tensors so torchvision v2 applies identical random params.
        if self.rand_resize_crop is not None:
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            crop_kw = {"scale": (0.5, 1.0)}
            if isinstance(self.rand_resize_crop, dict):
                crop_kw.update(self.rand_resize_crop)
            transform = T.RandomResizedCrop(size, **crop_kw)
            if mask is not None:
                result = transform({"image": TVImage(bmode), "mask": TVMask(mask)})
                bmode = result["image"].as_subclass(torch.Tensor)
                mask  = result["mask"].as_subclass(torch.Tensor)
            else:
                bmode = transform(bmode)

        if self.rand_affine is not None:
            affine_kw = {"degrees": [0, 0], "translate": [0.1, 0.1]}
            if isinstance(self.rand_affine, dict):
                affine_kw.update(self.rand_affine)
            transform = T.RandomAffine(**affine_kw)
            if mask is not None:
                result = transform({"image": TVImage(bmode), "mask": TVMask(mask)})
                bmode = result["image"].as_subclass(torch.Tensor)
                mask  = result["mask"].as_subclass(torch.Tensor)
            else:
                bmode = transform(bmode)

        # Pixel-level augments: image only, mask unaffected.
        if self.rand_gamma:
            try:
                from src.transforms.pixel_augmentations import RandomGamma
                bmode = T.RandomApply([RandomGamma((0.6, 3))])(bmode)
            except ImportError:
                pass

        if self.rand_contrast:
            try:
                from src.transforms.pixel_augmentations import RandomContrast
                bmode = T.RandomApply([RandomContrast((0.7, 2))])(bmode)
            except ImportError:
                pass

        return bmode, mask


@dataclass
class PreprocessMask:
    """Resize a binary mask to a target size using nearest interpolation.

    Args:
        mask_size: Target ``(H, W)`` or int for square.
    """

    mask_size: int | tuple[int, int] = 256

    def __call__(self, mask) -> torch.Tensor:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.copy()).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        size = (
            (self.mask_size, self.mask_size)
            if isinstance(self.mask_size, int)
            else self.mask_size
        )
        mask = T.Resize(
            size, antialias=False, interpolation=InterpolationMode.NEAREST
        )(mask)
        return TVMask(mask)
