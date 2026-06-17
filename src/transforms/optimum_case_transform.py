"""Case-level transform for the OPTIMUM dataset with sweeps and biopsies.

Processes each sweep (with its synchronized roll series), each biopsy image
(with its grade group, PRI-MUS, cine_id), and case-level clinical prompts,
producing a model-ready dict.

Supports random subsampling / dropping of sweeps and biopsies.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from src.transforms.bmode_preprocessing import PreprocessBmode
from src.transforms.clinical_prompt_preprocessing import PreprocessClinicalPrompts
from src.transforms.sweep_preprocessing import PreprocessSweep


# =============================================================================
# Subsample helpers
# =============================================================================


def _subsample_indices(
    n: int,
    mode: str = "all",
    max_count: Optional[int] = None,
    drop_prob: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> list[int]:
    """Return a list of indices to keep from ``n`` items.

    Args:
        n: Total number of items.
        mode: One of ``"all"``, ``"random_one"``, ``"first"``, ``"last"``.
        max_count: If set and mode is ``"all"``, cap the number of kept items.
        drop_prob: Per-item independent drop probability (applied after mode
            selection, except when only one item remains).
        rng: Optional numpy RandomState for reproducibility.

    Returns:
        Sorted list of kept indices (may be empty if everything is dropped;
        caller should handle that).
    """
    if rng is None:
        rng = np.random.RandomState()

    if n == 0:
        return []

    if mode == "random_one":
        return [int(rng.randint(0, n))]
    elif mode == "first":
        return [0]
    elif mode == "last":
        return [n - 1]
    elif mode == "all":
        indices = list(range(n))
        # Random drop
        if drop_prob > 0 and len(indices) > 1:
            indices = [i for i in indices if rng.rand() > drop_prob]
            # Always keep at least one
            if len(indices) == 0:
                indices = [int(rng.randint(0, n))]
        # Cap
        if max_count is not None and len(indices) > max_count:
            indices = sorted(rng.choice(indices, size=max_count, replace=False).tolist())
        return indices
    else:
        raise ValueError(f"Unknown subsample mode: {mode!r}")


def _select_from_list(lst: list, indices: list[int]) -> list:
    return [lst[i] for i in indices]


def _safe_int(x, default: int = 0) -> int:
    """Convert to int, treating None and NaN as ``default``."""
    if x is None:
        return default
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return default
    return int(x)


def _safe_float(x, default: float = 0.0) -> float:
    """Convert to float, treating None and NaN as ``default``."""
    if x is None:
        return default
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return default
    return x


# =============================================================================
# Main transform
# =============================================================================


@dataclass
class OPTIMUMCaseTransform:
    """Case-level transform for the OPTIMUM sweeps + biopsies dataset.

    Processes the output of ``OPTIMUMCasesWithSweepsAndBiopsiesJSONDataset``
    into model-ready tensors.

    The output dict will contain:

    **Sweeps** (lists, one element per kept sweep):
        - ``sweep``: list of ``(C, T, H, W)`` float tensors
        - ``sweep_roll_series``: list of ``(T',)`` float tensors
        - ``sweep_id``: list of str

    **Biopsies** (stacked tensors, one row per kept biopsy):
        - ``biopsy_image``: ``(N, C, H, W)`` float tensor
        - ``biopsy_roll_angle``: ``(N,)`` float tensor
        - ``biopsy_grade_group``: ``(N,)`` long tensor
        - ``biopsy_label``: ``(N,)`` long tensor — per-biopsy csPC (GG ≥ 2)
        - ``biopsy_primus``: ``(N,)`` long tensor (−1 for missing)
        - ``biopsy_cine_id``: list of str

    **Labels**:
        - ``label``: ``(,)`` long tensor — 1 if any biopsy GG ≥ ``min_gg_for_positive``
        - ``max_gg``: ``(,)`` long tensor

    **Clinical prompts** (if present in input):
        - ``psa``, ``age``, etc. — case-level normalised float tensors
        - ``biopsy_psa``, ``biopsy_age``, etc. — ``(N, 1)`` broadcast copies

    **Case metadata** (passthrough):
        - ``case``, ``center``, ``study_arm``, etc.

    Args:
        sweep_preprocess: Kwargs forwarded to ``PreprocessSweep``.
        bmode_preprocess: Kwargs forwarded to ``PreprocessBmode``.
        prompt_preprocess: Kwargs forwarded to ``PreprocessClinicalPrompts``.
        sweep_subsample_mode: ``"all"``, ``"random_one"``, ``"first"``, ``"last"``.
        sweep_max_count: Max sweeps to keep (None = no cap).
        sweep_drop_prob: Per-sweep independent drop probability.
        biopsy_subsample_mode: Same modes as sweeps.
        biopsy_max_count: Max biopsies to keep.
        biopsy_drop_prob: Per-biopsy independent drop probability.
        min_gg_for_positive: Minimum Gleason grade group for positive label.
        train: Enable stochastic augmentations + random subsampling.
    """

    # --- sub-transforms config ---
    sweep_preprocess: dict = field(default_factory=dict)
    bmode_preprocess: dict = field(default_factory=dict)
    prompt_preprocess: dict = field(default_factory=dict)

    # --- sweep subsampling ---
    sweep_subsample_mode: str = "all"
    sweep_max_count: Optional[int] = None
    sweep_drop_prob: float = 0.0

    # --- biopsy subsampling ---
    biopsy_subsample_mode: str = "all"
    biopsy_max_count: Optional[int] = None
    biopsy_drop_prob: float = 0.0

    # --- labels ---
    min_gg_for_positive: int = 2

    # --- general ---
    train: bool = True

    def __post_init__(self):
        self._sweep_transform = PreprocessSweep(
            train=self.train,
            **self.sweep_preprocess,
        )
        self._bmode_transform = PreprocessBmode(
            train=self.train,
            **self.bmode_preprocess,
        )
        self._prompt_transform = PreprocessClinicalPrompts(
            **self.prompt_preprocess,
        )
        self._rng = np.random.RandomState()

    def __call__(self, sample: dict) -> dict:
        out = {}

        # ---- sweeps ---------------------------------------------------------
        sweeps_raw = sample.get("sweep", [])
        roll_series_raw = sample.get("sweep_roll_series", [])
        sweep_ids = sample.get("sweep_id", [])
        sweep_embs_raw = sample.get("sweep_backbone_embeddings", [])
        # Determine sweep count from whichever source is available.
        n_sweeps = max(len(sweeps_raw), len(roll_series_raw), len(sweep_embs_raw))

        if n_sweeps > 0:
            keep = _subsample_indices(
                n_sweeps,
                mode=self.sweep_subsample_mode if self.train else "all",
                max_count=self.sweep_max_count,
                drop_prob=self.sweep_drop_prob if self.train else 0.0,
                rng=self._rng,
            )

            if sweeps_raw:
                out["sweep"] = []
            out["sweep_roll_series"] = []
            out["sweep_id"] = _select_from_list(sweep_ids, keep)
            if sweep_embs_raw:
                out["sweep_backbone_embeddings"] = []

            for i in keep:
                sweep_i = sweeps_raw[i] if i < len(sweeps_raw) else None
                roll_i = roll_series_raw[i] if i < len(roll_series_raw) else None
                emb_i = sweep_embs_raw[i] if i < len(sweep_embs_raw) else None

                if sweep_i is not None:
                    # Full path: run PreprocessSweep on images, with sync
                    sweep_dict = {"sweep": sweep_i}
                    if roll_i is not None:
                        sweep_dict["sweep_roll_series"] = roll_i
                    if emb_i is not None:
                        sweep_dict["sweep_backbone_embeddings"] = emb_i

                    # Make sure sync_keys includes roll_series and embeddings
                    prev_sync = self._sweep_transform.sync_keys
                    extra_sync = []
                    if roll_i is not None and "sweep_roll_series" not in prev_sync:
                        extra_sync.append("sweep_roll_series")
                    if emb_i is not None and "sweep_backbone_embeddings" not in prev_sync:
                        extra_sync.append("sweep_backbone_embeddings")
                    if extra_sync:
                        self._sweep_transform.sync_keys = list(prev_sync) + extra_sync

                    processed = self._sweep_transform(sweep_dict)

                    self._sweep_transform.sync_keys = prev_sync  # restore

                    out["sweep"].append(processed["sweep"])
                    if "sweep_roll_series" in processed:
                        out["sweep_roll_series"].append(processed["sweep_roll_series"])
                    elif roll_i is not None:
                        out["sweep_roll_series"].append(
                            torch.from_numpy(np.asarray(roll_i)).float()
                        )
                    if "sweep_backbone_embeddings" in processed:
                        out["sweep_backbone_embeddings"].append(
                            processed["sweep_backbone_embeddings"]
                        )
                else:
                    # Embeddings-only path: no sweep images — do frame
                    # selection directly on embeddings + roll_series.
                    if emb_i is not None:
                        emb_t = torch.from_numpy(np.asarray(emb_i)).float()
                    else:
                        emb_t = None

                    if roll_i is not None:
                        roll_t = torch.from_numpy(np.asarray(roll_i)).float()
                    else:
                        roll_t = None

                    # Apply frame subsequence selection using embeddings as
                    # the primary temporal reference.
                    if self._sweep_transform.select_frame_subsequence is not None and emb_t is not None:
                        from src.transforms.sweep_preprocessing import (
                            SelectFrameSubsequence,
                        )
                        selector = SelectFrameSubsequence(
                            **self._sweep_transform.select_frame_subsequence, dim=0,
                        )
                        indices = selector.get_indices(emb_t)
                        idx_tensor = torch.tensor(indices)
                        emb_t = emb_t.index_select(0, idx_tensor)
                        if roll_t is not None:
                            roll_t = roll_t.index_select(0, idx_tensor)

                    if emb_t is not None:
                        out["sweep_backbone_embeddings"].append(emb_t)
                    if roll_t is not None:
                        out["sweep_roll_series"].append(roll_t)

        # ---- biopsies -------------------------------------------------------
        biopsy_images_raw = sample.get("biopsy_images", [])
        biopsy_rolls = sample.get("biopsy_roll_angle", [])
        biopsy_ggs = sample.get("biopsy_grade_group", [])
        biopsy_primus = sample.get("biopsy_primus", [])
        biopsy_cine_ids = sample.get("biopsy_cine_id", [])
        biopsy_pct_cancer = sample.get("biopsy_pct_cancer", [])
        biopsy_embs_raw = sample.get("biopsy_backbone_embeddings", [])
        # Determine biopsy count from whichever source is available.
        n_biopsies = max(
            len(biopsy_images_raw), len(biopsy_rolls), len(biopsy_embs_raw)
        )
        n_kept = 0

        if n_biopsies > 0:
            keep = _subsample_indices(
                n_biopsies,
                mode=self.biopsy_subsample_mode if self.train else "all",
                max_count=self.biopsy_max_count,
                drop_prob=self.biopsy_drop_prob if self.train else 0.0,
                rng=self._rng,
            )
            n_kept = len(keep)

            biopsy_needle_masks_raw = sample.get("biopsy_needle_masks", [])
            if biopsy_images_raw:
                if biopsy_needle_masks_raw:
                    # Process image + mask jointly so spatial augments stay in sync
                    processed = [
                        self._bmode_transform(
                            biopsy_images_raw[i],
                            mask=torch.from_numpy(
                                np.asarray(biopsy_needle_masks_raw[i])
                            ).float().unsqueeze(0) / 255.0,
                        )
                        for i in keep
                    ]
                    out["biopsy_image"] = torch.stack([p[0] for p in processed])
                    out["biopsy_needle_mask"] = torch.stack([p[1] for p in processed])
                else:
                    out["biopsy_image"] = torch.stack(
                        [self._bmode_transform(biopsy_images_raw[i]) for i in keep]
                    )
            elif biopsy_needle_masks_raw:
                # Embeddings-only path: no images, but needle masks still present.
                # Resize masks to the configured image_size so they are spatially consistent.
                target_size = self._bmode_transform.image_size  # (H, W) or int
                if isinstance(target_size, int):
                    target_size = (target_size, target_size)
                masks = []
                for i in keep:
                    m = torch.from_numpy(np.asarray(biopsy_needle_masks_raw[i])).float().unsqueeze(0) / 255.0
                    if tuple(m.shape[-2:]) != tuple(target_size):
                        m = F.interpolate(m.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
                    masks.append(m)
                out["biopsy_needle_mask"] = torch.stack(masks)
            out["biopsy_roll_angle"] = torch.tensor(
                [_safe_float(biopsy_rolls[i] if i < len(biopsy_rolls) else 0.0) for i in keep]
            )
            out["biopsy_grade_group"] = torch.tensor(
                [_safe_int(biopsy_ggs[i] if i < len(biopsy_ggs) else 0) for i in keep]
            ).long()
            out["biopsy_label"] = (out["biopsy_grade_group"] >= 2).long()
            out["biopsy_cspca"] = (out["biopsy_grade_group"] >= 2).long()
            out["biopsy_pca"] = (out["biopsy_grade_group"] >= 1).long()
            out["biopsy_primus"] = torch.tensor(
                [_safe_int(biopsy_primus[i] if i < len(biopsy_primus) else None, default=-1)
                 for i in keep]
            ).long()
            out["biopsy_cine_id"] = _select_from_list(biopsy_cine_ids, keep)
            out["biopsy_pct_cancer"] = torch.tensor(
                [_safe_float(biopsy_pct_cancer[i] if i < len(biopsy_pct_cancer) else 0.0) for i in keep]
            )

            # Pre-extracted biopsy backbone embeddings (subsample in sync)
            if biopsy_embs_raw:
                out["biopsy_backbone_embeddings"] = torch.from_numpy(
                    np.stack([biopsy_embs_raw[i] for i in keep])
                ).float()

            # Pre-extracted biopsy image feature maps (subsample in sync)
            biopsy_img_feats_raw = sample.get("biopsy_image_feats", [])
            if biopsy_img_feats_raw:
                out["biopsy_image_feats"] = torch.from_numpy(
                    np.stack([np.asarray(biopsy_img_feats_raw[i]) for i in keep])
                ).float()  # (N_kept, 256, H, W) — convert float16 → float32 here

            # Legacy aliases for backward compatibility with old tracker/loss
            out["grade_group"] = out["biopsy_grade_group"]
            out["cspca_core"] = out["biopsy_label"]
            out["cine_id"] = out["biopsy_cine_id"]

        # ---- labels ---------------------------------------------------------
        all_ggs = biopsy_ggs if biopsy_ggs else [0]
        clean_ggs = [_safe_int(g) for g in all_ggs]
        max_gg = max(clean_ggs)
        out["label"] = torch.tensor(max_gg >= self.min_gg_for_positive).long()
        out["max_gg"] = torch.tensor(max_gg).long()

        # Legacy alias
        out["cspca"] = torch.tensor(max_gg >= 2).long()

        # ---- clinical prompts -----------------------------------------------
        prompt_keys = ("psa", "age", "approx_psa_density", "family_history")
        prompt_source = {
            k: v for k, v in sample.items() if k in prompt_keys
        }
        out = self._prompt_transform(out | prompt_source)

        # Broadcast case-level prompts to per-biopsy tensors
        if n_biopsies > 0 and n_kept > 0:
            for pk in prompt_keys:
                if pk in out and isinstance(out[pk], torch.Tensor):
                    val = out[pk]
                    # Expand to (n_kept, ...) preserving trailing dims.
                    # psa is (1,) → (n_kept, 1); scalar → (n_kept,)
                    if val.dim() == 0:
                        out[f"biopsy_{pk}"] = val.unsqueeze(0).expand(n_kept)
                    elif val.dim() >= 1:
                        out[f"biopsy_{pk}"] = val.unsqueeze(0).expand(n_kept, *val.shape)
                    else:
                        out[f"biopsy_{pk}"] = val

        # ---- case-level metadata (passthrough) ------------------------------
        for key in ("case", "center", "study_arm", "max_grade_group"):
            if key in sample:
                out[key] = sample[key]

        return out
