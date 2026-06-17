import torch 
from torch import nn 
from einops import repeat, rearrange


def get_bags_of_predictions(predictions_map, *masks):
    """Accepts maps of logits and a list of masks. Extracts the `bags` of logits, i.e., the logits inside the masked regions
    for each batch item.
    """

    masks = [mask.to(predictions_map.device) for mask in masks]
    _, _, h, w = predictions_map.shape 
    masks = [nn.functional.interpolate(mask, size=(h, w), mode='nearest') for mask in masks]

    final_masks = []

    for i in range(len(predictions_map)):
        final_mask = torch.ones(
            predictions_map[i].shape, device=predictions_map[i].device
        ).bool()
        for mask in masks:
            final_mask &= (mask[i] > 0.5)
        
        final_masks.append(final_mask)

    masks = torch.stack(final_masks)
    predictions, batch_idx = MaskedPredictionModule()(predictions_map, masks)

    outputs = []
    for i in range(len(predictions_map)): 
        outputs.append(predictions[batch_idx == i])
    return outputs


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx