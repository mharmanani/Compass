from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from monai import transforms as MT
from torchvision.transforms import v2 as T
from src.transforms.select_frame_subsequence import SelectFrameSubsequence


# =============================================================================
# Helpers
# =============================================================================


def _to_float_tensor(x) -> torch.Tensor:
    """Convert ndarray/path/tensor to float, int→float /255."""
    if isinstance(x, str):
        from PIL import Image as PILImage
        x = np.array(PILImage.open(x))
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    if not x.is_floating_point():
        x = x.float() / 255.0
    else:
        x = x.float()
    return x


def _expand_channels(x: torch.Tensor, dim: int, num_channels: int) -> torch.Tensor:
    """Repeat along ``dim`` to reach ``num_channels``."""
    if x.shape[dim] < num_channels:
        repeats = [1] * x.ndim
        repeats[dim] = -(-num_channels // x.shape[dim])  # ceiling division
        x = x.repeat(*repeats)
        # Trim if overshoot
        if x.shape[dim] != num_channels:
            x = x.narrow(dim, 0, num_channels)
    return x


def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale to [0, 1] using per-tensor min/max."""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin > 0:
        x = (x - xmin) / (xmax - xmin)
    return x


def ensure_3d_float_tensor(
    img,
    num_channels: int | None = None,
) -> torch.Tensor:
    """Normalize input to a float tensor of shape (C, H, W).

    Accepts:
        - File path (str) → loaded via PIL
        - np.ndarray or torch.Tensor of shape (H, W) → (1, H, W)
        - np.ndarray or torch.Tensor of shape (H, W, C) → (C, H, W)
        - np.ndarray or torch.Tensor of shape (C, H, W) → passthrough

    If the input has an integer dtype (uint8, int32, etc.), it is converted
    to float and scaled to [0, 1] by dividing by 255.

    If num_channels is set and the result has fewer channels, the channel
    dimension is repeated to match (e.g. grayscale → 3-channel).
    """
    img = _to_float_tensor(img)

    # (H, W, C) → (C, H, W)
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)
    # (H, W) → (1, H, W)
    elif img.ndim == 2:
        img = img.unsqueeze(0)

    if num_channels is not None:
        img = _expand_channels(img, dim=0, num_channels=num_channels)

    return img


def ensure_4d_float_tensor(
    sweep,
    num_channels: int | None = None,
) -> torch.Tensor:
    """Normalize input to a float tensor of shape (T, C, H, W).

    Accepts:
        - np.ndarray or torch.Tensor of shape (T, H, W) → (T, 1, H, W)
        - np.ndarray or torch.Tensor of shape (T, C, H, W) → passthrough
        - np.ndarray or torch.Tensor of shape (H, W) → (1, 1, H, W)

    If the input has an integer dtype (uint8, int32, etc.), it is converted
    to float and scaled to [0, 1] by dividing by 255.

    If num_channels is set and the result has fewer channels, the channel
    dimension is repeated to match (e.g. grayscale → 3-channel).
    """
    sweep = _to_float_tensor(sweep)

    if sweep.ndim == 2:
        sweep = sweep.unsqueeze(0).unsqueeze(0)
    elif sweep.ndim == 3:
        sweep = sweep.unsqueeze(1)
    elif sweep.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Expected sweep with 2, 3, or 4 dimensions, got {sweep.ndim}D "
            f"with shape {tuple(sweep.shape)}"
        )

    if num_channels is not None:
        sweep = _expand_channels(sweep, dim=1, num_channels=num_channels)

    return sweep


def ensure_temporal_tensor(seq, dim: int = 0) -> torch.Tensor:
    """Normalize a 1-D or N-D temporal sequence to a torch float tensor.

    Accepts np.ndarray or torch.Tensor of any shape. Integer dtypes are
    converted to float (no /255 — use for non-image sequences like roll angles).
    """
    if isinstance(seq, np.ndarray):
        seq = torch.from_numpy(seq)
    return seq.float()


# =============================================================================
# Synchronized frame selection for dict-based samples
# =============================================================================


@dataclass
class SynchronizedFrameSelection:
    """Select a temporal subsequence from multiple dict entries in sync.

    Given a dict sample, this selects frames along the temporal dimension of a
    primary key (``sweep_key``) and applies the *same* indices to every key
    listed in ``sync_keys``.  Keys that are missing from the dict are silently
    skipped.

    Each sync key can have a *different* temporal dimension from the sweep
    (specified via ``sync_dims``), but they must have the same number of frames
    along that dimension.

    Args:
        sweep_key: Dict key for the primary sweep tensor.
        sync_keys: Additional dict keys whose temporal axis should be subsampled
            with the same indices.
        sweep_dim: Temporal dimension of the sweep tensor.  Default 0.
        sync_dims: Per-key temporal dimension overrides.  Keys not listed here
            default to ``sweep_dim``.
        subsequence_kw: Kwargs forwarded to ``SelectFrameSubsequence``
            (e.g. ``mode``, ``n_frames``, ``jitter_strength``).
    """

    sweep_key: str = "sweep"
    sync_keys: list[str] = field(default_factory=list)
    sweep_dim: int = 0
    sync_dims: dict[str, int] = field(default_factory=dict)
    subsequence_kw: dict = field(default_factory=lambda: {"mode": "uniform", "n_frames": 16})

    def __call__(self, sample: dict) -> dict:
        out = sample.copy()

        sweep = out[self.sweep_key]
        if not isinstance(sweep, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor for '{self.sweep_key}', "
                f"got {type(sweep).__name__}"
            )

        selector = SelectFrameSubsequence(**self.subsequence_kw, dim=self.sweep_dim)
        indices = selector.get_indices(sweep)
        idx_tensor = torch.tensor(indices)

        # Sub-select sweep
        out[self.sweep_key] = sweep.index_select(
            self.sweep_dim, idx_tensor.to(sweep.device)
        )

        # Sub-select synchronized sequences
        for key in self.sync_keys:
            if key not in out:
                continue
            seq = out[key]
            if not isinstance(seq, torch.Tensor):
                seq = ensure_temporal_tensor(seq)
            dim = self.sync_dims.get(key, self.sweep_dim)
            out[key] = seq.index_select(dim, idx_tensor.to(seq.device))

        return out


# =============================================================================
# Spatial / volume augmentations
# =============================================================================


@dataclass
class SweepSpatialTransforms:
    """Per-frame 2-D spatial transforms for a (T, C, H, W) sweep tensor.

    Args:
        image_size: Resize target.  None = no resize.
        minmax_normalize: Scale each frame to [0, 1] using min/max.
        rand_resize_crop: True for defaults, or dict of extra kwargs.  None = off.
        mean / std: ``T.Normalize`` params. Default ``(1,1,1)/(1,1,1)`` is a no-op.
        train: Enable stochastic augmentations.
    """

    image_size: Optional[int | tuple[int, int]] = None
    minmax_normalize: bool = False
    rand_resize_crop: Optional[bool | dict] = None
    mean: tuple = (1, 1, 1)
    std: tuple = (1, 1, 1)
    train: bool = True

    def __call__(self, sweep: torch.Tensor) -> torch.Tensor:
        if self.image_size is not None:
            sweep = T.Resize(self.image_size)(sweep)

        if self.minmax_normalize:
            sweep = _minmax_normalize(sweep)

        if self.train and self.rand_resize_crop is not None:
            if self.image_size is None:
                raise ValueError("rand_resize_crop requires image_size to be set.")
            crop_params = {"scale": (0.5, 1.0)}
            if isinstance(self.rand_resize_crop, dict):
                crop_params.update(self.rand_resize_crop)
            sweep = T.RandomResizedCrop(self.image_size, **crop_params)(sweep)

        sweep = T.Normalize(mean=self.mean, std=self.std)(sweep)

        return sweep


@dataclass
class SweepVolumeTransforms:
    """3-D volume-level augmentations for a (C, T, H, W) tensor.

    Args:
        volume_size: Resize target.  None = no resize.
        rand_histogram_shift: Toggle.
        rand_affine: Toggle.
        rand_affine_kw: Extra kwargs for monai RandAffine.
        rand_coarse_dropout: Toggle.
        rand_coarse_dropout_kw: Extra kwargs for monai RandCoarseDropout.
        train: Enable stochastic augmentations.
    """

    volume_size: Optional[tuple[int, int, int]] = None
    rand_histogram_shift: bool = False
    rand_affine: bool = False
    rand_affine_kw: dict = field(default_factory=lambda: {
        "prob": 0.5,
        "shear_range": (0.5, 0.5),
        "translate_range": (0.1, 0.1),
    })
    rand_coarse_dropout: bool = False
    rand_coarse_dropout_kw: dict = field(default_factory=lambda: {
        "prob": 0.2,
        "holes": 100,
        "spatial_size": 20,
        "fill_value": 0,
    })
    train: bool = True

    def __call__(self, sweep: torch.Tensor) -> torch.Tensor:
        if self.volume_size is not None:
            sweep = MT.Resize(self.volume_size)(sweep)

        if self.train and self.rand_histogram_shift:
            sweep = MT.RandHistogramShift(prob=0.5)(sweep)

        if self.train and self.rand_affine:
            sweep = MT.RandAffine(**self.rand_affine_kw)(sweep)

        if self.train and self.rand_coarse_dropout:
            sweep = MT.RandCoarseDropout(**self.rand_coarse_dropout_kw)(sweep)

        return sweep


# =============================================================================
# Main pipeline
# =============================================================================


@dataclass
class PreprocessSweep:
    """Full preprocessing pipeline for ultrasound sweep volumes.

    Handles:
        1. Input normalisation  (int→float /255, expand dims, repeat channels)
        2. Synchronized frame sub-selection across multiple dict keys
        3. Per-frame 2-D spatial transforms
        4. Layout permute  (T,C,H,W) → (C,T,H,W)
        5. 3-D volume-level augmentations
        6. Permute layout back (C,T,H,W) -> (T,C,H,W)

    Can be called with:
        - A raw tensor / ndarray  → returns (C, T, H, W) tensor.
        - A dict with ``sweep_key`` → transforms in-place and returns the dict.
          Synchronized keys are sub-selected but not otherwise augmented.

    Args:
        sweep_key: Dict key for the main sweep data.  Default ``"sweep"``.
        sync_keys: Additional temporal keys to sub-select in sync with the sweep.
        sync_dims: Per-key temporal dimension override (default same as sweep_dim).
        select_frame_subsequence: Kwargs for ``SelectFrameSubsequence``.  None = skip.
        image_size: Resize target for each frame.
        minmax_normalize: Scale each frame to [0, 1] using min/max.
        rand_resize_crop: True / dict / None.
        mean / std: ``T.Normalize`` params. Default ``(1,1,1)/(1,1,1)`` is a no-op.
        volume_size: 3-D resize target after layout permute.
        rand_histogram_shift / rand_affine / rand_coarse_dropout: Toggles.
        rand_affine_kw / rand_coarse_dropout_kw: Extra kwargs.
        num_channels: Repeat channels to reach this count.  None = no change.
        train: Enable stochastic augmentations.
    """

    # --- dict-mode keys ---
    sweep_key: str = "sweep"
    sync_keys: list[str] = field(default_factory=list)
    sync_dims: dict[str, int] = field(default_factory=dict)

    # --- frame selection ---
    select_frame_subsequence: Optional[dict] = None

    # --- spatial ---
    image_size: Optional[int | tuple[int, int]] = None
    minmax_normalize: bool = False
    rand_resize_crop: Optional[bool | dict] = None
    mean: tuple = (0, 0, 0)
    std: tuple = (1, 1, 1)

    # --- volume ---
    volume_size: Optional[tuple[int, int, int]] = None
    rand_histogram_shift: bool = False
    rand_affine: bool = False
    rand_affine_kw: dict = field(default_factory=lambda: {
        "prob": 0.5,
        "shear_range": (0.5, 0.5),
        "translate_range": (0.1, 0.1),
    })
    rand_coarse_dropout: bool = False
    rand_coarse_dropout_kw: dict = field(default_factory=lambda: {
        "prob": 0.2,
        "holes": 100,
        "spatial_size": 20,
        "fill_value": 0,
    })

    # --- general ---
    num_channels: Optional[int] = None
    train: bool = True

    def __call__(self, inp):
        """Accept a dict or a raw tensor/ndarray."""
        if isinstance(inp, dict):
            return self._transform_dict(inp)
        return self._transform_tensor(inp)

    # ----- dict path --------------------------------------------------------

    def _transform_dict(self, sample: dict) -> dict:
        out = sample.copy()

        # 1. Normalise the sweep tensor
        out[self.sweep_key] = ensure_4d_float_tensor(
            out[self.sweep_key], num_channels=self.num_channels,
        )

        # 2. Normalise sync tensors (keep them as float, no /255 image scaling)
        for key in self.sync_keys:
            if key in out:
                out[key] = ensure_temporal_tensor(out[key])

        # 3. Synchronized frame selection
        if self.select_frame_subsequence is not None:
            frame_selector = SynchronizedFrameSelection(
                sweep_key=self.sweep_key,
                sync_keys=self.sync_keys,
                sweep_dim=0,
                sync_dims=self.sync_dims,
                subsequence_kw=self.select_frame_subsequence,
            )
            out = frame_selector(out)

        # 4. Spatial transforms on sweep only
        out[self.sweep_key] = self._spatial(out[self.sweep_key])

        # 5. Layout permute (T, C, H, W) → (C, T, H, W)
        out[self.sweep_key] = out[self.sweep_key].permute(1, 0, 2, 3)

        # 6. Volume transforms on sweep only
        out[self.sweep_key] = self._volume(out[self.sweep_key])
        out[self.sweep_key] = out[self.sweep_key].permute(1, 0, 2, 3)

        return out

    # ----- raw tensor path ---------------------------------------------------

    def _transform_tensor(self, sweep) -> torch.Tensor:
        sweep = ensure_4d_float_tensor(sweep, num_channels=self.num_channels)

        if self.select_frame_subsequence is not None:
            result = SelectFrameSubsequence(
                **self.select_frame_subsequence, dim=0,
            )(sweep)
            sweep = result if isinstance(result, torch.Tensor) else result[0]

        sweep = self._spatial(sweep)
        sweep = sweep.permute(1, 0, 2, 3)
        sweep = self._volume(sweep)
        sweep = sweep.permute(1, 0, 2, 3)
        return sweep

    # ----- sub-pipelines -----------------------------------------------------

    def _spatial(self, sweep: torch.Tensor) -> torch.Tensor:
        return SweepSpatialTransforms(
            image_size=self.image_size,
            minmax_normalize=self.minmax_normalize,
            rand_resize_crop=self.rand_resize_crop,
            mean=self.mean,
            std=self.std,
            train=self.train,
        )(sweep)

    def _volume(self, sweep: torch.Tensor) -> torch.Tensor:
        return SweepVolumeTransforms(
            volume_size=self.volume_size,
            rand_histogram_shift=self.rand_histogram_shift,
            rand_affine=self.rand_affine,
            rand_affine_kw=self.rand_affine_kw,
            rand_coarse_dropout=self.rand_coarse_dropout,
            rand_coarse_dropout_kw=self.rand_coarse_dropout_kw,
            train=self.train,
        )(sweep)
