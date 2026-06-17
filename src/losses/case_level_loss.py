import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CaseLevelLoss:
    """Combined case-level (sequence), optional biopsy-level (token), and
    optional heatmap loss.

    Args:
        label_key: key for case-level labels (default ``"label"``).
        token_label_key: key for per-biopsy labels.  ``None`` disables
            token-level loss entirely.
        token_logits_key: key for per-biopsy logits produced by the model
            (default ``"biopsy_logits"``; v2 models use ``"token_logits"``).
        token_pct_cancer_key: key for per-biopsy pct cancer values (0–100).
        token_loss_weight: weighting factor for the token loss term.
        minimum_pct_cancer_for_loss: if set, benign cores and cores with
            ``pct_cancer < threshold`` are excluded from the token loss.
        heatmap_logits_key: key for per-biopsy heatmap logits
            ``list[(N_i, 1, H, W)]``.  ``None`` disables the heatmap loss.
        heatmap_needle_mask_key: key for per-biopsy needle masks
            ``list[(N_i, 1, H, W)]``.  Predictions are restricted to the
            needle region before computing the proportion.
        heatmap_loss_weight: weighting factor for the heatmap loss term.
        minimum_pct_cancer_for_heatmap_loss: same semantics as
            ``minimum_pct_cancer_for_loss`` but applied to heatmap biopsies.
    """

    label_key: str = "label"
    token_label_key: str | None = None
    token_logits_key: str = "biopsy_logits"
    token_pct_cancer_key: str = "pct_cancer"
    token_loss_weight: float = 1.0
    minimum_pct_cancer_for_loss: float | None = None
    heatmap_logits_key: str | None = None
    heatmap_needle_mask_key: str = "biopsy_needle_mask"
    heatmap_loss_weight: float = 1.0
    minimum_pct_cancer_for_heatmap_loss: float | None = None

    def __call__(self, data):
        logits = data["logits"]
        device = logits.device
        loss = torch.tensor(0.0, device=device)

        # case-level loss
        labels = torch.stack(data[self.label_key]).to(logits.device)
        loss_sequence = torch.nn.CrossEntropyLoss()(logits, labels)
        data["sequence_loss"] = loss_sequence
        loss = loss + loss_sequence

        # biopsy-level (token) loss
        if self.token_label_key is not None:
            token_logits = data[self.token_logits_key]
            token_logits = torch.cat(token_logits, dim=0)
            token_labels = data[self.token_label_key]
            token_labels = torch.cat(token_labels).to(device)
            token_pct_cancer = torch.cat(
                data[self.token_pct_cancer_key], dim=0
            ).to(device)

            if self.minimum_pct_cancer_for_loss is not None:
                valid_mask = (token_labels == 0) | (
                    token_pct_cancer >= self.minimum_pct_cancer_for_loss
                )
                token_logits = token_logits[valid_mask]
                token_labels = token_labels[valid_mask]

            token_loss = torch.nn.CrossEntropyLoss()(token_logits, token_labels)
            loss = loss + self.token_loss_weight * token_loss
            data["token_loss"] = token_loss

        # heatmap loss — proportion BCE within needle mask region
        # For each biopsy: mean sigmoid over needle-masked pixels should match
        # biopsy_pct_cancer / 100  (mirrors CancerDetectionMILLoss + ProportionBCE)
        if self.heatmap_logits_key is not None:
            heatmap_logits = data.get(self.heatmap_logits_key)
            if heatmap_logits is not None and any(h is not None for h in heatmap_logits):
                needle_masks   = data[self.heatmap_needle_mask_key]  # list[(N_i, 1, H, W)]
                pct_list       = data[self.token_pct_cancer_key]     # list[(N_i,)] 0–100
                label_list     = data.get(self.token_label_key)      # list[(N_i,)] or None

                heatmap_loss = torch.tensor(0.0, device=device)
                n_valid = 0

                for i, hmap in enumerate(heatmap_logits):
                    if hmap is None:
                        continue
                    # hmap:   (N_i, 1, H_out, W_out)
                    # nmask:  (N_i, 1, H, W)
                    pct        = pct_list[i].to(device)        # (N_i,)
                    proportion = pct / 100.0                   # (N_i,) in [0, 1]
                    nmask      = needle_masks[i].float().to(device)  # (N_i, 1, H, W)

                    # Optional: exclude low-involvement positives
                    if self.minimum_pct_cancer_for_heatmap_loss is not None and label_list is not None:
                        labels_i = label_list[i].to(device)
                        keep = (labels_i == 0) | (pct >= self.minimum_pct_cancer_for_heatmap_loss)
                        hmap       = hmap[keep]
                        nmask      = nmask[keep]
                        proportion = proportion[keep]

                    if hmap.shape[0] == 0:
                        continue

                    # Resize needle mask to heatmap spatial dims if they differ
                    if nmask.shape[-2:] != hmap.shape[-2:]:
                        nmask = F.interpolate(
                            nmask, size=hmap.shape[-2:], mode="bilinear", align_corners=False
                        )
                    nmask_bin = (nmask > 0.5).float()  # (N_i, 1, H_out, W_out)

                    # Mean sigmoid over needle-masked pixels per biopsy
                    pred_sig  = hmap.sigmoid() * nmask_bin           # zero outside mask
                    mask_area = nmask_bin.sum(dim=(-1, -2))          # (N_i, 1)
                    pred_sum  = pred_sig.sum(dim=(-1, -2))           # (N_i, 1)

                    # Only use biopsies where the mask has at least one pixel
                    valid = mask_area[:, 0] > 0                      # (N_i,)
                    if not valid.any():
                        continue

                    pred_prop = (pred_sum[:, 0] / mask_area[:, 0].clamp(min=1))[valid]
                    pred_prop = pred_prop.clamp(1e-6, 1 - 1e-6)
                    p         = proportion[valid]

                    # Proportion BCE: -p*log(p_hat) - (1-p)*log(1-p_hat)
                    bce = -(p * pred_prop.log() + (1 - p) * (1 - pred_prop).log())
                    heatmap_loss = heatmap_loss + bce.mean()
                    n_valid += 1

                if n_valid > 0:
                    heatmap_loss = heatmap_loss / n_valid
                loss = loss + self.heatmap_loss_weight * heatmap_loss
                data["heatmap_loss"] = heatmap_loss

        data["loss"] = loss
        return data
