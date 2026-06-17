# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from re import L
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Literal, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_extra_upscale_layers: int = 0,
        output_upscaling_version: Literal['default', 'interpolate'] = 'default'
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        if output_upscaling_version == 'default':
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
                ),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(
                    transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
                ),
                activation(),
            )
            self._output_upscaling = self.output_upscaling
    
        elif output_upscaling_version == 'interpolate': 
            self.output_upscaling_interpolate = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, mode='nearest'
                ), 
                nn.Conv2d(
                    transformer_dim, transformer_dim // 4, kernel_size=3, padding=1
                ),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.Upsample(
                    scale_factor=2, mode='nearest'
                ),
                nn.Conv2d(
                    transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1
                ),
                activation(),
            )
            self._output_upscaling = self.output_upscaling_interpolate

        elif output_upscaling_version == 'none':
            self._output_upscaling = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )

        else: 
            raise ValueError(
                f"Unknown output_upscaling_version: {output_upscaling_version}. "
                "Supported versions are 'default', 'interpolate', and 'none'."
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        if num_extra_upscale_layers > 0:
            extra_upscaling_layers = []
            # Add extra upscaling layers if specified
            for _ in range(num_extra_upscale_layers):
                extra_upscaling_layers.append(
                    LayerNorm2d(self.transformer_dim // 8)
                )
                extra_upscaling_layers.append(
                    nn.ConvTranspose2d(
                        self.transformer_dim // 8, self.transformer_dim // 8, kernel_size=2, stride=2
                    )
                )
                extra_upscaling_layers.append(nn.GELU())
            self.extra_upscaling = nn.Sequential(*extra_upscaling_layers)
        else: 
            self.extra_upscaling = None

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_patch_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
          output_patch_features (bool): If True, also return spatially-rich
            post-transformer image token features (B, h*w, C).

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          (optional) torch.Tensor: patch features (B, h*w, C) if output_patch_features
        """
        result = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            output_patch_features=output_patch_features,
        )

        if output_patch_features:
            masks, iou_pred, patch_features = result
        else:
            masks, iou_pred, *out = result
            patch_features = None

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        if output_patch_features:
            return (masks, iou_pred, patch_features)
        return (masks, iou_pred, *out)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        output_patch_features: bool = False,
    ):
        """Predicts masks. See 'forward' for more details.

        Args:
            output_patch_features: If True, also return the spatially-rich
                post-transformer image token features ``(B, h*w, C)`` before
                upscaling.  Useful for downstream spatiotemporal attention.
        """
        
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        num_sparse_prompts = sparse_prompt_embeddings.shape[1]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # src is now (B, h*w, C) — the post-cross-attention image features
        # Save a copy before reshaping if the caller wants patch-level features
        patch_features = src if output_patch_features else None  # (B, h*w, C)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self._output_upscaling(src)
        if self.extra_upscaling is not None:
            upscaled_embedding = self.extra_upscaling(upscaled_embedding)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        if output_patch_features:
            return masks, iou_pred, patch_features
        return masks, iou_pred

    def forward_embeddings_list(self, image_embeddings, image_pe,
                                sparse_prompt_embeddings, dense_prompt_embeddings):
        """
        Predicts masks given image and prompt embeddings, returning the
        embeddings used to predict masks.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
            list[tensor]: the embeddings used to predict masks
        """
        masks, iou_pred, *out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        return [masks, iou_pred, *out]


class ClassDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_cls_tokens=1, 
        cls_output_dims=[2],
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_cls_tokens = num_cls_tokens

        self.cls_tokens = nn.Embedding(self.num_cls_tokens, transformer_dim)
        self.output_cls_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, cls_output_dims[i], 3)
                for i in range(self.num_cls_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        output_hidden_states=False,
        output_patch_features=False,
    ):
        """
        Predict class logits given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          output_hidden_states (bool): If True, return hidden states dict.
          output_patch_features (bool): If True, include post-transformer
              image token features ``(B, h*w, C)`` in the returned dict
              (only when output_hidden_states is True).

        Returns:
            list[tensor] or dict: the class predictions, or dict with extras.
        """
        # Concatenate output tokens

        output_tokens = self.cls_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        cls_token_out = hs[:, :self.num_cls_tokens, :]

        cls_out_list = []
        for i in range(self.num_cls_tokens): 
            cls_out_list.append(self.output_cls_mlps[i](cls_token_out[:, i, :]))
            
        if output_hidden_states: 
            result = {
                "class_logits": cls_out_list, 
                "hidden_states": hs, 
                "class_token_hidden_states": [
                    hs[:, i, :] for i in range(self.num_cls_tokens)
                ],
            }
            if output_patch_features:
                result["patch_features"] = src  # (B, h*w, C)
            return result

        return cls_out_list


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
