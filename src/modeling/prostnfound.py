from argparse import ArgumentParser, Action, _StoreAction, _StoreTrueAction, FileType
import os
import typing as tp
from warnings import warn
import torch
from torch import nn
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm

# from .sam_wrappers import build_medsam, build_sam, build_sammed_2d, build_adapter_medsam_256, build_adapter_sam, build_adapter_sammed_2d
from src.layers.common import LayerNorm2d
from src.modeling._medsam.segment_anything.modeling.transformer import (
    TwoWayTransformer,
)
from src.utils.argparse import UpdateDictAction
def _get_patch_view():
    from src.utils.data.patch_extraction import PatchView
    return PatchView


from itertools import chain
import logging
from .registry import list_models, register_model, create_model
from ._medsam.segment_anything.modeling.sam import Sam
from ._medsam.segment_anything.modeling.mask_decoder import ClassDecoder


class ProstNFound(nn.Module):

    BACKBONE_OPTIONS = [
        "sam",
        "medsam",
        "sam_med2d",
        "adapter_medsam",
        "adapter_sam",
        "adapter_sammed_2d",
    ]
    _WARNED_RESIZE = False

    def __init__(
        self,
        sam_backbone: Sam,
        floating_point_prompts: list[str] = [],
        discrete_prompts: list[str] = [],
        discrete_prompt_nvals: list[int] = [],
        use_sparse_cnn_patch_features: bool = False,
        use_sparse_cnn_patch_features_rf: bool = False,
        use_prostate_mask_prompt: bool = False,
        num_data_independent_prompts: int = 0,
        prompt_dropout: float = 0.0,  # dropout for prompt embeddings
        replace_patch_embed: bool = False,
        sparse_cnn_backbone_path: str | None = None,
        freeze_mask_decoder: bool = False,
        freeze_image_encoder: bool = False,
        freeze_cnn: bool = False,
        img_emb_dropout: float = 0.0,
        cnn_patches_whole_prostate: bool = False,
        pos_embed_cnn_patch: bool = True,
        pool_patch_features: bool | None = None,
        prompt_embedding_dim=256,
        use_class_decoder: bool = False,
        use_cnn_dense_prompt: bool = False,
        output_mode: tp.Literal["heatmaps", "classifier", "all"] = "all",
        upsample_image_encoder_features: tp.Literal["interpolate", "conv", None] = None,
        upsample_image_encoder_features_factor: int = 4,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.use_cnn_dense_prompt = use_cnn_dense_prompt
        self.floating_point_prompts = floating_point_prompts
        self.discrete_prompts = discrete_prompts
        self.discrete_prompt_nvals = discrete_prompt_nvals
        self.prompts = self.floating_point_prompts + self.discrete_prompts
        self.use_sparse_cnn_patch_features = use_sparse_cnn_patch_features
        self.use_sparse_cnn_patch_features_rf = use_sparse_cnn_patch_features_rf
        self.num_data_independent_prompts = num_data_independent_prompts
        self.use_prostate_mask_prompt = use_prostate_mask_prompt
        self.upsample_image_encoder_features = upsample_image_encoder_features

        if use_sparse_cnn_patch_features and use_sparse_cnn_patch_features_rf:
            raise ValueError(
                "Both sparse_cnn_patch_features and sparse_cnn_patch_features_rf cannot be True"
            )

        self.prompt_dropout = prompt_dropout
        self.replace_patch_embed = replace_patch_embed
        self.cnn_patches_whole_prostate = cnn_patches_whole_prostate
        self.pos_embed_cnn_patch = pos_embed_cnn_patch
        self.pool_patch_features = pool_patch_features
        if replace_patch_embed and sam_backbone != "sam_med2d":
            raise ValueError(
                "replace_patch_embed is only supported for sam_med2d backbone"
            )

        self.sparse_cnn_backbone_path = sparse_cnn_backbone_path
        self.medsam_model = sam_backbone
        self.img_emb_dropout = nn.Dropout(img_emb_dropout)

        if freeze_image_encoder:
            logging.debug("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            logging.debug("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        # ====================================================
        # BUILD PROMPT MODULES
        # ==================================================

        # null prompt - used for prompt dropout
        self.null_prompt = nn.Parameter(torch.zeros(1, prompt_embedding_dim))

        # floating point prompts
        self.floating_point_prompt_modules = torch.nn.ModuleDict()
        for prompt in self.floating_point_prompts:
            self.floating_point_prompt_modules[prompt] = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, prompt_embedding_dim),
            )

        # integer prompts
        self.integer_prompt_modules = torch.nn.ModuleDict()
        for prompt, num_categories in zip(
            self.discrete_prompts, self.discrete_prompt_nvals
        ):
            self.integer_prompt_modules[prompt] = nn.Embedding(
                num_categories, prompt_embedding_dim
            )

        # data independent prompts
        if self.num_data_independent_prompts > 0:
            self.data_independent_prompts = nn.Parameter(
                torch.randn(1, num_data_independent_prompts, prompt_embedding_dim)
            )

        # CNN for extracting patch features
        if self.use_sparse_cnn_patch_features or self.use_sparse_cnn_patch_features_rf:
            from timm.models.resnet import resnet10t

            model = resnet10t(
                in_chans=3,
            )
            model.fc = nn.Identity()
            model = nn.Sequential(nn.InstanceNorm2d(3), model)
            if sparse_cnn_backbone_path is not None:
                state = torch.load(sparse_cnn_backbone_path, map_location="cpu")
                model.load_state_dict(
                    {
                        k.replace("backbone.", ""): v
                        for k, v in state.items()
                        if "backbone" in k
                    }
                )
            self.patch_feature_cnn = model
            if freeze_cnn:
                for param in self.patch_feature_cnn.parameters():
                    param.requires_grad = False

            # project the CNN features to the prompt space
            # self.patch_feature_prompt_module = nn.Linear(512, EMBEDDING_DIM)
            self.patch_feature_prompt_module = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, prompt_embedding_dim),
            )
            self.pad_token = nn.Parameter(
                torch.zeros(prompt_embedding_dim)
            )  # for padding the number of patches to a fixed number in a minibatch

        # CNN for dense prompt
        if self.use_cnn_dense_prompt:
            from src.layers.samus_cnn_embed import SingleCNNEmbed

            self.cnn_for_dense_embedding = SingleCNNEmbed(
                patchsize=16,
                in_chans=3,
                embed_dim=prompt_embedding_dim,
                norm_layer=nn.BatchNorm2d,
            )
        else:
            self.cnn_for_dense_embedding = None

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))
        self.use_tc = False

        # ====================================================
        # Build additional modules
        # ==================================================

        self.class_decoder = (
            ClassDecoder(
                transformer_dim=256,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                num_cls_tokens=1,
                cls_output_dims=[2],
            )
            if use_class_decoder
            else None
        )

        if self.upsample_image_encoder_features == "conv":
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    prompt_embedding_dim, prompt_embedding_dim, kernel_size=2, stride=2
                ),
                LayerNorm2d(prompt_embedding_dim),
                nn.GELU(),
                nn.ConvTranspose2d(
                    prompt_embedding_dim, prompt_embedding_dim, kernel_size=2, stride=2
                ),
                nn.GELU(),
            )
        elif self.upsample_image_encoder_features == "interpolate":
            self.upsample = nn.Upsample(
                scale_factor=upsample_image_encoder_features_factor,
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.upsample = None

    def forward(
        self,
        image,
        rf_image=None,
        prostate_mask=None,
        needle_mask=None,
        output_mode=None,
        output_patch_features=False,
        output_image_feats=False,
        **prompts,
    ):

        output_mode = output_mode or self.output_mode

        B, C, H, W = image.shape

        image_feats = self.medsam_model.image_encoder(image)
        image_feats = self.img_emb_dropout(image_feats)
        if self.upsample is not None:
            image_feats = self.upsample(image_feats)

        if self.use_prostate_mask_prompt:
            if (
                prostate_mask is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                mask = None
            else:
                B, C, H, W = prostate_mask.shape
                if H != 256 or W != 256:
                    prostate_mask = torch.nn.functional.interpolate(
                        prostate_mask, size=(256, 256)
                    )
                mask = prostate_mask
        else:
            mask = None

        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, mask
        )
        if (dense_embedding.shape[-2] != image_feats.shape[-2]) or (
            dense_embedding.shape[-1] != image_feats.shape[-1]
        ):
            dense_embedding = torch.nn.functional.interpolate(
                dense_embedding,
                size=image_feats.shape[-2:],
            )

        dense_embedding = dense_embedding.repeat_interleave(len(image), 0)

        if self.use_cnn_dense_prompt:
            # extract dense prompt features from the image
            assert self.cnn_for_dense_embedding is not None
            cnn_emb = self.cnn_for_dense_embedding(image).permute(0, 3, 1, 2)
            dense_embedding = dense_embedding + cnn_emb

        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)

        for prompt_name, prompt_value in prompts.items():
            if (
                prompt_value is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                # skip this prompt and use the 'null embedding' instead
                prompt_embedding = self.null_prompt.repeat_interleave(len(image), 0)
            else:
                if prompt_name in self.floating_point_prompts:
                    prompt_embedding = self.floating_point_prompt_modules[prompt_name](
                        prompt_value
                    )
                elif prompt_name in self.discrete_prompts:
                    prompt_embedding = self.integer_prompt_modules[prompt_name](
                        prompt_value
                    )
                else:
                    raise ValueError(f"Unknown prompt: {prompt_name}")

            prompt_embedding = prompt_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, prompt_embedding], dim=1)

        if self.num_data_independent_prompts > 0:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )

        if self.use_sparse_cnn_patch_features:
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_bmode(
                image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        if self.use_sparse_cnn_patch_features_rf:
            patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_rf(
                rf_image, needle_mask, prostate_mask
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        pe = self.medsam_model.prompt_encoder.get_dense_pe()
        if (pe.shape[-2] != image_feats.shape[-2]) or (
            pe.shape[-1] != image_feats.shape[-1]
        ):
            pe = torch.nn.functional.interpolate(
                pe,
                size=image_feats.shape[-2:],
            )

        mask_decoder_result = self.medsam_model.mask_decoder(
            image_feats,
            pe,
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
            output_patch_features=output_patch_features,
        )
        if output_patch_features:
            mask_logits, iou, mask_decoder_patch_features = mask_decoder_result
        else:
            mask_logits, iou = mask_decoder_result[:2]
            mask_decoder_patch_features = None

        mask_logits = (
            mask_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )

        if self.class_decoder is not None:
            class_decoder_all_outputs = self.class_decoder(
                image_feats,
                pe,
                sparse_embedding,
                dense_embedding,
                output_hidden_states=True,
                output_patch_features=output_patch_features,
            )
            cls_outputs = class_decoder_all_outputs.pop("class_logits")
        else:
            cls_outputs = None
            class_decoder_all_outputs = None

        if output_mode == "heatmaps":
            return mask_logits
        elif output_mode == "classifier":
            assert cls_outputs is not None
            return cls_outputs[0]
        else:
            outputs = {
                "mask_logits": mask_logits,
                "cls_outputs": cls_outputs,
                "class_logits_list": cls_outputs,  # more descriptive alias
                "iou": iou,
            }
            if class_decoder_all_outputs:
                outputs['class_decoder_hidden_states'] = class_decoder_all_outputs['class_token_hidden_states']
                outputs['class_decoder_last_hidden_state'] = outputs['class_decoder_hidden_states'][-1]
                if 'patch_features' in class_decoder_all_outputs:
                    outputs['class_decoder_patch_features'] = class_decoder_all_outputs['patch_features']
            if mask_decoder_patch_features is not None:
                outputs['mask_decoder_patch_features'] = mask_decoder_patch_features
            if output_image_feats:
                outputs['image_feats'] = image_feats
            return outputs

    def get_cnn_patch_embedding_bmode(self, image, needle_mask, prostate_mask):
        # we need to extract patches from the images.
        DEVICE = image.device
        patches = []
        batch_indices = []
        positions = []
        B = len(image)
        for i in range(B):

            im = image[i].permute(1, 2, 0).cpu().numpy()
            mask = needle_mask[i].permute(1, 2, 0).cpu().numpy()
            prostate_mask_ = prostate_mask[i].permute(1, 2, 0).cpu().numpy()

            if self.cnn_patches_whole_prostate:
                masks = [prostate_mask_]
                thresholds = [0.9]
            else:
                masks = [mask, prostate_mask_]
                thresholds = [0.3, 0.9]

            pv = _get_patch_view().from_sliding_window(
                im,
                window_size=(128, 128),
                stride=(64, 64),
                masks=masks,
                thresholds=thresholds,
            )
            for position, patch in zip(pv.positions, pv):
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                positions.append(torch.from_numpy(position))
                batch_indices.append(i)

        patches = torch.stack(patches).to(DEVICE)
        positions = torch.stack(positions).to(DEVICE)
        positions = positions[:, [1, 0]]
        batch_indices = torch.tensor(batch_indices)

        patch_cnn_output = self.patch_feature_cnn(patches)
        patch_cnn_output = self.patch_feature_prompt_module(patch_cnn_output)
        if self.pos_embed_cnn_patch:
            position_encoding_outputs = (
                self.medsam_model.prompt_encoder.pe_layer.forward_with_coords(
                    positions[None, ...], image_size=(1024, 1024)
                )[0]
            )
            patch_cnn_output = patch_cnn_output + position_encoding_outputs

        sparse_embeddings_by_batch = []
        for i in range(B):
            patch_embeddings_for_batch = patch_cnn_output[batch_indices == i]
            if self.pool_patch_features == "mean":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.mean(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                )
            elif self.pool_patch_features == "max":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                ).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        if self.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.prompt_dropout:
                    patch_cnn_sparse_embeddings[:, i, :] = (
                        self.null_prompt.repeat_interleave(B, 0)
                    )

        return patch_cnn_sparse_embeddings

    def get_cnn_patch_embedding_rf(self, image, needle_mask, prostate_mask):
        # we need to extract patches from the images.
        DEVICE = image.device
        patches = []
        batch_indices = []
        positions = []

        im_size_mm = 28, 46.06
        B, C, H, W = image.shape
        logging.debug(f"RF shape: {image.shape}")
        im_size_px = H, W
        patch_size_mm = 5, 5
        if not self.cnn_patches_whole_prostate:
            patch_stride_mm = 1, 1
        else:
            patch_stride_mm = 2, 2
        patch_size_px = int(patch_size_mm[0] / im_size_mm[0] * im_size_px[0]), int(
            patch_size_mm[1] / im_size_mm[1] * im_size_px[1]
        )
        patch_stride_px = int(patch_stride_mm[0] / im_size_mm[0] * im_size_px[0]), int(
            patch_stride_mm[1] / im_size_mm[1] * im_size_px[1]
        )
        logging.debug(f"Patch size: {patch_size_px}")

        B = len(image)
        for i in range(B):

            im = image[i].permute(1, 2, 0).cpu().numpy()
            mask = needle_mask[i].permute(1, 2, 0).cpu().numpy()
            prostate_mask_ = prostate_mask[i].permute(1, 2, 0).cpu().numpy()

            if self.cnn_patches_whole_prostate:
                masks = [prostate_mask_]
                thresholds = [0.9]
            else:
                masks = [mask]
                thresholds = [0.6]

            pv = _get_patch_view().from_sliding_window(
                im,
                window_size=patch_size_px,
                stride=patch_stride_px,
                masks=masks,
                thresholds=thresholds,
                align_to="topright",
            )
            for position, patch in zip(pv.positions, pv):
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                positions.append(torch.from_numpy(position))
                batch_indices.append(i)

        logging.debug(f"Extracted {len(patches)} patches from {B} rf images")
        if len(patches) == 0:
            return None

        patches = torch.stack(patches).to(self.device)
        # patches should be resized to 256 by 256 as used in the RF CNNs
        patches = torch.nn.functional.interpolate(
            patches, size=(256, 256), mode="bilinear"
        )

        positions = torch.stack(positions).to(DEVICE)
        positions = positions[:, [1, 0]]
        batch_indices = torch.tensor(batch_indices)

        patch_cnn_output = self.patch_feature_cnn(patches)
        patch_cnn_output = self.patch_feature_prompt_module(patch_cnn_output)

        if self.pos_embed_cnn_patch:
            position_encoding_outputs = (
                self.medsam_model.prompt_encoder.pe_layer.forward_with_coords(
                    positions[None, ...], image_size=(1024, 1024)
                )[0]
            )
            patch_cnn_output = patch_cnn_output + position_encoding_outputs

        sparse_embeddings_by_batch = []
        for i in range(B):
            patch_embeddings_for_batch = patch_cnn_output[batch_indices == i]  # N x 256
            if self.pool_patch_features == "mean":
                if len(patch_embeddings_for_batch) == 0:
                    return None  # no patches found
                patch_embeddings_for_batch = torch.mean(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                )
            elif self.pool_patch_features == "max":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                ).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        if self.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.prompt_dropout:
                    patch_cnn_sparse_embeddings[:, i, :] = (
                        self.null_prompt.repeat_interleave(B, 0)
                    )

        B, N, C = patch_cnn_sparse_embeddings.shape
        if self.pool_patch_features == "transformer":
            patch_cnn_sparse_embeddings = self.patch_aggregator(
                patch_cnn_sparse_embeddings
            )
            patch_cnn_sparse_embeddings = patch_cnn_sparse_embeddings.mean(
                dim=1, keepdim=True
            )

        return patch_cnn_sparse_embeddings

    def train(self, mode: bool = True):
        super().train(mode)
        # always keep cnn in eval mode - otherwise batch norm might interfere.
        if (
            (
                self.use_sparse_cnn_patch_features
                or self.use_sparse_cnn_patch_features_rf
            )
            and self.sparse_cnn_backbone_path is not None
            and self.patch_feature_cnn is not None
        ):
            self.patch_feature_cnn.eval()

    def get_params_groups(self):

        encoder_parameters = [
            p
            for (k, p) in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]

        warmup_parameters = []
        # warmup components from backbone
        warmup_parameters = chain(
            warmup_parameters, self.medsam_model.mask_decoder.parameters()
        )
        warmup_parameters = chain(
            warmup_parameters, self.medsam_model.prompt_encoder.parameters()
        )
        if hasattr(self.medsam_model.image_encoder, "neck"):
            warmup_parameters = chain(
                warmup_parameters, self.medsam_model.image_encoder.neck.parameters()
            )
        # null prompt
        warmup_parameters = chain(warmup_parameters, [self.null_prompt])
        # floating point prompts
        for module in self.floating_point_prompt_modules.values():
            warmup_parameters = chain(warmup_parameters, module.parameters())
        # data independent prompts
        for module in self.integer_prompt_modules.values():
            warmup_parameters = chain(warmup_parameters, module.parameters())
        # patch prompts
        if self.use_sparse_cnn_patch_features or self.use_sparse_cnn_patch_features_rf:
            warmup_parameters = chain(
                warmup_parameters, self.patch_feature_prompt_module.parameters()
            )
        # data independent prompts
        if self.num_data_independent_prompts > 0:
            warmup_parameters = chain(
                warmup_parameters, [self.data_independent_prompts]
            )
        if self.class_decoder is not None:
            warmup_parameters = chain(
                warmup_parameters, self.class_decoder.parameters()
            )
        if self.cnn_for_dense_embedding is not None:
            warmup_parameters = chain(
                warmup_parameters, self.cnn_for_dense_embedding.parameters()
            )
        if self.upsample is not None:
            warmup_parameters = chain(warmup_parameters, self.upsample.parameters())

        cnn_parameters = (
            self.patch_feature_cnn.parameters()
            if self.use_sparse_cnn_patch_features
            or self.use_sparse_cnn_patch_features_rf
            else []
        )

        return encoder_parameters, warmup_parameters, cnn_parameters

    @property
    def device(self):

        return next(self.parameters()).device


@register_model
def prostnfound(
    *, backbone=None, backbone_kw: dict = {}, backbone_cfg: dict = {}, prompts=[], **kwargs
):
    """
    Builds the ProstNFound model.
    """

    from . import sam # trigger import to ensure sam is registered

    floating_point_prompts = []
    for prompt in prompts:
        if prompt in [
            "age",
            "psa",
            "approx_psa_density",
            "base_apex_encoding",
            "mid_lateral_encoding",
            "family_history",
        ]:
            floating_point_prompts.append(prompt)
        elif prompt == 'pos': 
            floating_point_prompts.append("base_apex_encoding")
            floating_point_prompts.append("mid_lateral_encoding")
        elif prompt == 'psad': 
            floating_point_prompts.append('approx_psa_density')
        else: 
            raise ValueError(f"Unknown prompt {prompt}")

    if backbone_cfg: 
        backbone = create_model(**backbone_cfg)
    else: 
        backbone = create_model(backbone, **backbone_kw)
    return ProstNFound(
        backbone, floating_point_prompts=floating_point_prompts, **kwargs
    )


@register_model
def prostnfound_medsam_noprompt(backbone="medsam_vit_b", **kwargs):
    """
    ProstNFound model with medsam backbone.
    """
    return prostnfound(backbone=backbone, **kwargs)


@register_model
def prostnfound_medsam_noprompt_adapter(**kwargs):
    backbone_kw = kwargs.pop("backbone_kw", {})
    backbone_kw["adapter_dim"] = 256
    return prostnfound(backbone="medsam_vit_b", backbone_kw=backbone_kw, **kwargs)


@register_model
def prostnfound_noprompt_adapter_medsam_legacy(adapter_dim=256):
    from . import sam

    return prostnfound(
        backbone="medsam_adapter", backbone_kw=dict(adapter_dim=adapter_dim)
    )


@register_model
def prostnfound_noprompt_adapter_medsam_legacy_ssl_adapt(
    enc_ckpt=None,
    adapter_dim=256,
    **kwargs,
):
    assert enc_ckpt is not None
    from . import sam

    return prostnfound(
        backbone="medsam_adapter",
        backbone_kw=dict(
            encoder_checkpoint=enc_ckpt,
            adapter_dim=adapter_dim,
            **kwargs.get("backbone_kw", {}),
        ),
        **kwargs,
    )


@register_model
def prostnfound_adapter_medsam_legacy(
    adapter_dim=256,
    prompts=["age", "psa", "approx_psa_density"],
    upsample=False,
    **kwargs,
):
    from . import sam

    if upsample:
        backbone = "medsam_adapter_upsample"
    else:
        backbone = "medsam_adapter"

    return prostnfound(
        backbone=backbone,
        backbone_kw=dict(adapter_dim=adapter_dim, **kwargs.pop("backbone_kw", {})),
        prompts=prompts,
        **kwargs,
    )


@register_model
def prostnfound_sam_noprompt(
    **kwargs,
):
    from .sam import build_medsam

    return ProstNFound(
        sam_backbone=build_medsam(),
    )


@register_model
def prostnfound_adapter_medsam_legacy_ssl_adapt(
    enc_ckpt=None,
    adapter_dim=256,
    prompts=["age", "psa", "approx_psa_density"],
    upsample=False,
    **kwargs,
):
    assert enc_ckpt is not None
    from . import sam

    backbone = "medsam_adapter_upsample" if upsample else "medsam_adapter"

    return prostnfound(
        backbone=backbone,
        backbone_kw=dict(
            encoder_checkpoint=enc_ckpt,
            adapter_dim=adapter_dim,
            **kwargs.pop("backbone_kw", {}),
        ),
        prompts=prompts,
        **kwargs,
    )


@register_model
def prostnfound_medsam_ssl_adapt(
    enc_ckpt=None, prompts=["age", "psa", "approx_psa_density"], **kwargs
):
    assert enc_ckpt is not None
    from . import sam

    return prostnfound(
        backbone="medsam",
        backbone_kw=dict(encoder_checkpoint=enc_ckpt, **kwargs.pop("backbone_kw", {})),
        prompts=prompts,
        **kwargs,
    )


@register_model 
def prostnfound_adapter_medsam_legacy_lowres(
    adapter_dim=256,
    prompts=["age", "psa"],
    upsample=False,
    **kwargs,
):
    from . import sam

    backbone = "medsam_adapter"
    
    return prostnfound(
        backbone=backbone,
        backbone_kw=dict(
            adapter_dim=adapter_dim, 
            mask_decoder_kw=dict(
                output_upscaling_version="none",
            )    
        ),
        prompts=prompts,
        **kwargs,
    )


@register_model 
def prostnfound_plus_final(strict=True, **kwargs):
    path = os.getenv("PNF_PLUS_CHECKPOINT", None)
    if path is None:
        raise ValueError(
            "PNF_PLUS_CHECKPOINT environment variable is not set. "
            "Please set it to the path of the PNF+ checkpoint."
        )
    state = torch.load(path, weights_only=False)
    args = state['args']

    from src.factories.prostnfound.models import get_model as create_prostnfound_model
    model = create_prostnfound_model(args, **kwargs)

    # model = create_model(
    #     args['model'], **args['model_kw']
    # )
    model_sd = state['model'] 
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    consume_prefix_in_state_dict_if_present(model_sd, "model.")
    msg = model.load_state_dict(model_sd, strict=strict)
    print(f"Loaded PNF+ model from {path} with msg: {msg}")

    return model


@register_model 
def pnf_plus_final(**kwargs): 
    return prostnfound_plus_final(**kwargs)


@register_model 
def pnf_sam_backbone(*, pnf_config):
    model: ProstNFound = create_model(**pnf_config)
    return model.medsam_model


class ProstNFoundWrapperForVideo(nn.Module): 
    def __init__(self, module, return_key=None):
        super().__init__()
        self.module = module
        self.return_key = return_key

    def forward(self, video_data, **prompts): 
        B, N, C, H, W = video_data.shape
        video_data = video_data.reshape(B * N, C, H, W) 

        repeated_prompts = {k: v.repeat(N, 1) for k, v in prompts.items()}
        outputs_dict = self.module(video_data, **repeated_prompts)

        def reshape_tensor_pyobj(obj):
            if isinstance(obj, torch.Tensor):
                return obj.reshape(B, N, *obj.shape[1:])
            elif isinstance(obj, dict):
                return {
                    k: reshape_tensor_pyobj(v) for k, v in obj.items()
                }
            else: 
                return [reshape_tensor_pyobj(v) for v in obj]

        outputs_dict = reshape_tensor_pyobj(outputs_dict)
        if self.return_key is not None:
            return outputs_dict[self.return_key]
        else:
            return outputs_dict


if __name__ == "__main__":

    device = "cuda"
    from . import sam 

    model = prostnfound_adapter_medsam_legacy_lowres().to(device)

    inputs = {
        "image": torch.randn(1, 3, 224, 224).to(device),
        # "hello": torch.randn(1, 1).to(device),
        # "world": torch.randint(0, 9, (1,)).to(device),
    }

    for _ in tqdm(range(5)):
        out = model.forward(**inputs, output_mode='all')

    breakpoint()

    print(f"Output shape: {out['mask_logits'].shape}")
