import copy
import torch
import torch.nn.functional as F
from torch import nn

from src.modeling import create_model, register_model
from src.modeling.utils import (
    collate_mil_batch,
    concat_variable_length_batch,
    unstack_variable_length_batch,
)
from src.modeling.wrappers import FrozenModuleWrapper


# =============================================================================
# Prompt key mapping helpers
# =============================================================================

# Maps canonical biopsy_* data keys to the inner ProstNFound kwarg names.
_DEFAULT_PROMPT_KEY_MAP = {
    "biopsy_psa": "psa",
    "biopsy_age": "age",
    "biopsy_approx_psa_density": "approx_psa_density",
    "biopsy_family_history": "family_history",
}


def _remap_prompts(data_kwargs: dict, key_map: dict, strict=True) -> dict:
    """Extract and rename prompt keys from *data_kwargs*.

    Returns a new dict with only the recognised prompt keys, renamed to the
    inner model's expected names.  Keys not in *key_map* are ignored.
    """
    out = {}
    for data_key, model_key in key_map.items():
        if data_key in data_kwargs:
            out[model_key] = data_kwargs[data_key]
        elif strict:
            raise ValueError(
                f"Looked for {data_key} in dictionary for input as {model_key}, but it is not present."
            )
    return out


@register_model
def prostnfound_mil(
    freeze_backbone=False, bert_kw={}, pnf_cfg={"name": "prostnfound_plus_final"}
):
    from src.modeling import create_model

    pnf = create_model(**pnf_cfg)
    wrapped_pnf = ProstNFoundWrapperForMILBatches(pnf)
    if freeze_backbone:
        wrapped_pnf = FrozenModuleWrapper(wrapped_pnf)

    from src.modeling.multiple_instance_learning.bert_based_mil_aggregator import (
        BertBasedMILAggregatorForSequenceClassification,
    )

    bert_kw_defaults = dict(
        num_classes=2,
        num_attention_heads=8,
        num_hidden_layers=2,
        intermediate_size=1024,
    )
    bert_kw_defaults.update(bert_kw)
    agg = BertBasedMILAggregatorForSequenceClassification(
        embedding_dim=256, **bert_kw_defaults
    )

    class CombinedModel(nn.Module):
        def __init__(self, wrapped_pnf, agg):
            super().__init__()
            self.wrapped_pnf = wrapped_pnf
            self.agg = agg

        def forward(self, *args, **kwargs):
            outputs = self.wrapped_pnf(*args, **kwargs)
            return self.agg(outputs)

    return CombinedModel(wrapped_pnf=wrapped_pnf, agg=agg)


@register_model
def prostnfound_mil_v2(
    freeze_backbone=False,
    num_classes=2,
    num_token_classes=2,
    pnf_cfg={"name": "prostnfound_plus_final"},
    pooling_strategy: str = "cls",
    **bert_kwargs,
):
    from src.modeling import create_model

    pnf = create_model(**pnf_cfg)
    wrapped_pnf = ProstNFoundWrapperForMILBatches(pnf)
    if freeze_backbone:
        wrapped_pnf = FrozenModuleWrapper(wrapped_pnf)

    from src.modeling.multiple_instance_learning.bert_based_mil_aggregator import (
        BertBasedMILAggregatorForTokenAndSequenceClassification,
    )

    bert_kw_defaults = dict(
        num_attention_heads=8, num_hidden_layers=2, intermediate_size=1024
    )
    bert_kw_defaults.update(bert_kwargs)
    agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
        embedding_dim=256,
        num_classes=num_classes,
        num_token_classes=num_token_classes,
        pooling_strategy=pooling_strategy,
        **bert_kw_defaults,
    )

    class CombinedModel(nn.Module):
        def __init__(self, wrapped_pnf, agg):
            super().__init__()
            self.wrapped_pnf = wrapped_pnf
            self.agg = agg

        def forward(self, *args, **kwargs):
            outputs = self.wrapped_pnf(*args, **kwargs)
            return self.agg(outputs)

    return CombinedModel(wrapped_pnf=wrapped_pnf, agg=agg)


# =============================================================================
# v3 — new canonical API: forward(*, biopsy_image, biopsy_psa=None, ...)
# =============================================================================


@register_model
def prostnfound_mil_v3(
    freeze_backbone=False,
    num_classes=2,
    num_token_classes=2,
    pnf_cfg={"name": "prostnfound_plus_final"},
    pooling_strategy: str = "cls",
    prompt_key_map: dict | None = None,
    **bert_kwargs,
):
    """Like ``prostnfound_mil_v2`` but with a dict-in / dict-out forward API.

    The returned model's ``forward`` accepts ``**data`` directly from the
    collated batch dict.  It looks for ``biopsy_image`` (list of per-case
    image tensors) and optional ``biopsy_psa``, ``biopsy_age``, etc.
    Unknown keys are silently ignored.

    Returns a dict with ``logits``, ``biopsy_logits``, and other keys.

    Args:
        prompt_key_map: Override mapping from data keys to inner ProstNFound
            kwarg names.  Defaults to ``_DEFAULT_PROMPT_KEY_MAP``.
    """
    from src.modeling import create_model as _create_model

    pnf = _create_model(**pnf_cfg)
    wrapped_pnf = ProstNFoundWrapperForMILBatches(pnf)
    if freeze_backbone:
        wrapped_pnf = FrozenModuleWrapper(wrapped_pnf)

    from src.modeling.multiple_instance_learning.bert_based_mil_aggregator import (
        BertBasedMILAggregatorForTokenAndSequenceClassification,
    )

    bert_kw_defaults = dict(
        num_attention_heads=8, num_hidden_layers=2, intermediate_size=1024
    )
    bert_kw_defaults.update(bert_kwargs)
    agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
        embedding_dim=256,
        num_classes=num_classes,
        num_token_classes=num_token_classes,
        pooling_strategy=pooling_strategy,
        **bert_kw_defaults,
    )

    return ProstNFoundMILV3(
        wrapped_pnf=wrapped_pnf,
        agg=agg,
        prompt_key_map=prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP),
    )


class ProstNFoundMILV3(nn.Module):
    """Dict-in / dict-out ProstNFound MIL model.

    ``forward`` accepts ``**kwargs`` from the collated batch dict:

    - ``biopsy_image``: list of ``(N_i, C, H, W)`` tensors (mutually exclusive with ``biopsy_backbone_embeddings``)
    - ``biopsy_backbone_embeddings``: list of ``(N_i, D)`` tensors; when provided the backbone is skipped entirely
    - ``biopsy_psa``, ``biopsy_age``, …: list of ``(N_i, 1)`` tensors (optional, used only when running backbone)
    - All other keys are silently ignored.

    Returns a dict::

        {
            "logits":         Tensor (B, num_classes),
            "biopsy_logits":  list[Tensor (N_i, num_token_classes)],
            "biopsy_hidden":  list[Tensor (N_i, D)],
        }
    """

    def __init__(self, wrapped_pnf, agg, prompt_key_map: dict | None = None):
        super().__init__()
        self.wrapped_pnf = wrapped_pnf
        self.agg = agg
        self.prompt_key_map = prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP)

    def forward(self, *, biopsy_image=None, biopsy_backbone_embeddings=None, **kwargs):
        if biopsy_backbone_embeddings is not None:
            # Skip backbone — embeddings already extracted
            hidden_states = list(biopsy_backbone_embeddings)
        else:
            # Remap biopsy_psa → psa, biopsy_age → age, etc.
            prompts = _remap_prompts(kwargs, self.prompt_key_map)
            # Backbone: list[Tensor(N_i, C, H, W)] → list[Tensor(N_i, D)]
            hidden_states = self.wrapped_pnf(biopsy_image, **prompts)

        # Aggregator: list[Tensor(N_i, D)] → dict
        agg_out = self.agg(hidden_states)

        return {
            "logits": agg_out["logits"],
            "biopsy_logits": agg_out["token_logits"],
            "biopsy_hidden": agg_out.get("token_hidden_states"),
        }


# =============================================================================
# v1/v2 backbone wrapper (unchanged — kept for backward compat)
# =============================================================================


class ProstNFoundWrapperForMILBatches(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bmode_list, return_image_feats=False, **prompts_list):
        bmode, batch_idx = concat_variable_length_batch(bmode_list)
        prompts = {
            k: concat_variable_length_batch(v)[0] for k, v in prompts_list.items()
        }
        outputs = self.model(bmode, output_image_feats=return_image_feats, **prompts)
        hidden = unstack_variable_length_batch(
            outputs["class_decoder_hidden_states"][0], batch_idx
        )
        if return_image_feats:
            image_feats = unstack_variable_length_batch(outputs["image_feats"], batch_idx)
            return hidden, image_feats
        return hidden


# =============================================================================
# Roll-angle positional embedding modules
# =============================================================================


class MLPRollAngleEmbedding(nn.Module):
    """Embed a scalar roll angle via a small MLP → ``(*, D)``."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, roll_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roll_angle: ``(N,)`` or ``(N, 1)`` float tensor.
        Returns:
            ``(N, D)`` embedding.
        """
        if roll_angle.dim() == 1:
            roll_angle = roll_angle.unsqueeze(-1)  # (N,) → (N, 1)
        return self.mlp(roll_angle)


class SinusoidalRollAngleEmbedding(nn.Module):
    """Embed a scalar roll angle via fixed sinusoidal frequencies → ``(*, D)``.

    Uses the same sin/cos interleaving as the original Transformer PE,
    but applied to a single continuous scalar (the roll angle) rather than
    discrete integer positions.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even for sinusoidal PE"
        self.embedding_dim = embedding_dim
        # Pre-compute inverse frequencies: shape (D/2,)
        half = embedding_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, roll_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roll_angle: ``(N,)`` or ``(N, 1)`` float tensor.
        Returns:
            ``(N, D)`` embedding.
        """
        if roll_angle.dim() == 1:
            roll_angle = roll_angle.unsqueeze(-1)  # (N,) → (N, 1)
        # (N, 1) @ (1, D/2) → (N, D/2)
        freqs = roll_angle @ self.inv_freq.unsqueeze(0)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)


class ZeroRollAngleEmbedding(nn.Module):
    """No-op roll-angle embedding — always returns zeros (disables roll PE)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, roll_angle: torch.Tensor) -> torch.Tensor:
        n = roll_angle.shape[0]
        return torch.zeros(n, self.embedding_dim, device=roll_angle.device, dtype=roll_angle.dtype)


# =============================================================================
# v3 + roll PE — new model class
# =============================================================================


class ProstNFoundMILV3WithRollPE(nn.Module):
    """Dict-in / dict-out ProstNFound MIL model with roll-angle positional
    embedding injected before the BERT aggregator.

    Like ``ProstNFoundMILV3``, but **requires** ``biopsy_roll_angle`` in the
    input dict.  The roll angle is embedded and added to each biopsy's hidden
    state before aggregation, giving the BERT layer information about the
    spatial position of the imaging plane.

    ``forward`` accepts ``**kwargs`` from the collated batch dict:

    - ``biopsy_image``: list of ``(N_i, C, H, W)`` tensors (required)
    - ``biopsy_roll_angle``: list of ``(N_i,)`` float tensors (required)
    - ``biopsy_psa``, ``biopsy_age``, …: list of ``(N_i, 1)`` tensors (optional)
    - All other keys are silently ignored.

    Returns a dict::

        {
            "logits":         Tensor (B, num_classes),
            "biopsy_logits":  list[Tensor (N_i, num_token_classes)],
            "biopsy_hidden":  list[Tensor (N_i, D)],
        }
    """

    def __init__(
        self,
        wrapped_pnf,
        agg,
        roll_embedding: nn.Module,
        prompt_key_map: dict | None = None,
    ):
        super().__init__()
        self.wrapped_pnf = wrapped_pnf
        self.agg = agg
        self.roll_embedding = roll_embedding
        self.prompt_key_map = prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP)

    def forward(self, *, biopsy_image, biopsy_roll_angle, **kwargs):
        # Remap biopsy_psa → psa, biopsy_age → age, etc.
        prompts = _remap_prompts(kwargs, self.prompt_key_map)

        # Backbone: list[Tensor(N_i, C, H, W)] → list[Tensor(N_i, D)]
        hidden_states = self.wrapped_pnf(biopsy_image, **prompts)

        # Add roll-angle positional embedding to each case's hidden states
        hidden_states = [
            h + self.roll_embedding(r)
            for h, r in zip(hidden_states, biopsy_roll_angle)
        ]

        # Aggregator: list[Tensor(N_i, D)] → dict
        agg_out = self.agg(hidden_states)

        return {
            "logits": agg_out["logits"],
            "biopsy_logits": agg_out["token_logits"],
            "biopsy_hidden": agg_out.get("token_hidden_states"),
        }


# =============================================================================
# Factory functions for roll PE variants
# =============================================================================


@register_model
def prostnfound_mil_v3_roll_mlp(
    freeze_backbone=False,
    num_classes=2,
    num_token_classes=2,
    pnf_cfg={"name": "prostnfound_plus_final"},
    pooling_strategy: str = "cls",
    prompt_key_map: dict | None = None,
    roll_hidden_dim: int = 128,
    **bert_kwargs,
):
    """ProstNFound MIL v3 with **MLP** roll-angle positional embedding.

    Same as ``prostnfound_mil_v3`` but requires ``biopsy_roll_angle`` and
    injects a learned MLP embedding (roll → Linear → ReLU → Linear → D) as an
    additive positional encoding to the biopsy hidden states before BERT.

    Extra args:
        roll_hidden_dim: Hidden dimension of the roll MLP (default 128).
    """
    from src.modeling import create_model as _create_model

    pnf = _create_model(**pnf_cfg)
    wrapped_pnf = ProstNFoundWrapperForMILBatches(pnf)
    if freeze_backbone:
        wrapped_pnf = FrozenModuleWrapper(wrapped_pnf)

    from src.modeling.multiple_instance_learning.bert_based_mil_aggregator import (
        BertBasedMILAggregatorForTokenAndSequenceClassification,
    )

    bert_kw_defaults = dict(
        num_attention_heads=8, num_hidden_layers=2, intermediate_size=1024
    )
    bert_kw_defaults.update(bert_kwargs)
    agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
        embedding_dim=256,
        num_classes=num_classes,
        num_token_classes=num_token_classes,
        pooling_strategy=pooling_strategy,
        **bert_kw_defaults,
    )

    roll_emb = MLPRollAngleEmbedding(embedding_dim=256, hidden_dim=roll_hidden_dim)

    return ProstNFoundMILV3WithRollPE(
        wrapped_pnf=wrapped_pnf,
        agg=agg,
        roll_embedding=roll_emb,
        prompt_key_map=prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP),
    )


@register_model
def prostnfound_mil_v3_roll_sinusoidal(
    freeze_backbone=False,
    num_classes=2,
    num_token_classes=2,
    pnf_cfg={"name": "prostnfound_plus_final"},
    pooling_strategy: str = "cls",
    prompt_key_map: dict | None = None,
    **bert_kwargs,
):
    """ProstNFound MIL v3 with **sinusoidal** roll-angle positional embedding.

    Same as ``prostnfound_mil_v3`` but requires ``biopsy_roll_angle`` and
    injects a fixed sinusoidal encoding of the roll angle as an additive
    positional encoding to the biopsy hidden states before BERT.
    """
    from src.modeling import create_model as _create_model

    pnf = _create_model(**pnf_cfg)
    wrapped_pnf = ProstNFoundWrapperForMILBatches(pnf)
    if freeze_backbone:
        wrapped_pnf = FrozenModuleWrapper(wrapped_pnf)

    from src.modeling.multiple_instance_learning.bert_based_mil_aggregator import (
        BertBasedMILAggregatorForTokenAndSequenceClassification,
    )

    bert_kw_defaults = dict(
        num_attention_heads=8, num_hidden_layers=2, intermediate_size=1024
    )
    bert_kw_defaults.update(bert_kwargs)
    agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
        embedding_dim=256,
        num_classes=num_classes,
        num_token_classes=num_token_classes,
        pooling_strategy=pooling_strategy,
        **bert_kw_defaults,
    )

    roll_emb = SinusoidalRollAngleEmbedding(embedding_dim=256)

    return ProstNFoundMILV3WithRollPE(
        wrapped_pnf=wrapped_pnf,
        agg=agg,
        roll_embedding=roll_emb,
        prompt_key_map=prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP),
    )


# =============================================================================
# "Compass" — unified biopsy + sweep model with type & roll embeddings
# =============================================================================
#
# Name rationale: A compass needle tells you where you are oriented — just like
# the roll angle tells the model where each imaging plane sits.  Plus it sounds
# cool.
# =============================================================================


class ProstNFoundSweepEncoder(nn.Module):
    """Encode sweep frames through ProstNFound, yielding one D-dim vector per
    frame (spatial dimensions are already pooled by ProstNFound).

    Input:
        sweep_list:  list of ``(T_i, C, H, W)`` tensors — one per sweep.
        prompts:     dict of ``{name: list[(T_i, 1)]}`` — optional prompts
                     broadcast to every frame of each sweep.

    Output:
        list of ``(T_i, D)`` tensors — one per sweep.
    """

    def __init__(self, pnf_model):
        super().__init__()
        self.pnf_model = pnf_model

    def forward(self, sweep_list, **prompts_list):
        # Treat each sweep's frames as independent images.
        all_frames, batch_idx = concat_variable_length_batch(sweep_list)
        prompts = {
            k: concat_variable_length_batch(v)[0] for k, v in prompts_list.items()
        }
        outputs = self.pnf_model(all_frames, **prompts)
        hidden = outputs["class_decoder_hidden_states"][0]  # (N_total, D)
        return unstack_variable_length_batch(hidden, batch_idx)


class Compass(nn.Module):
    """Unified biopsy + sweep MIL model with type and roll-angle positional
    embeddings.

    Architecture::

        biopsies ──► ProstNFound ──► (N_b, D) ──┐
                                                 ├──► + type_emb + roll_emb ──► BERT ──► case logits
        sweeps   ──► ProstNFound ──► (T_s, D) ──┘                                   ──► biopsy logits
                   (per-frame)

    The BERT aggregator sees a single mixed-token sequence per case:
    ``[CLS, biopsy_1, …, biopsy_Nb, sweep1_t1, …, sweep1_Tt, sweep2_t1, …]``

    Each token receives an additive **type embedding** (biopsy vs sweep-frame)
    and an additive **roll-angle positional embedding** (from the MLP roll
    encoder).

    Only biopsy-positioned tokens produce token-level logits; sweep tokens
    contribute to attention context but are not classified individually.

    ``forward`` accepts ``**data`` from the collated batch dict:

    Required:
        - ``biopsy_image``: list of ``(N_i, C, H, W)`` — OR —
          ``biopsy_backbone_embeddings``: list of ``(N_i, D_backbone)``
          (pre-extracted backbone features; skips the biopsy encoder)
        - ``biopsy_roll_angle``: list of ``(N_i,)``
        - ``sweep``: list of list of ``(T_j, C, H, W)`` — OR —
          ``sweep_backbone_embeddings``: list of list of ``(T_j, D_backbone)``
          (pre-extracted backbone features; skips the sweep encoder)
        - ``sweep_roll_series``: list of list of ``(T_j,)``

    Optional (prompt keys):
        - ``biopsy_psa``, ``biopsy_age``, …

    Returns::

        {
            "logits":         (B, num_classes),
            "biopsy_logits":  list[(N_i, num_token_classes)],
            "biopsy_hidden":  list[(N_i, D)],
        }

    If ``detached_token_classifier`` is provided, the output dict also
    contains ``"detached_biopsy_logits"`` — per-biopsy logits produced from
    the **pre-BERT** backbone features (with ``.detach()`` so gradients do
    not flow back into BERT).  This allows a biopsy-level auxiliary loss
    that trains only the token classifier head without distorting the
    aggregator's attention patterns.
    """

    # Type indices
    BIOPSY_TYPE = 0
    SWEEP_TYPE = 1

    def __init__(
        self,
        pnf_backbone: nn.Module,
        agg,
        roll_embedding: nn.Module,
        embedding_dim: int = 256,
        backbone_dim: int | None = None,
        prompt_key_map: dict | None = None,
        detached_token_classifier: nn.Module | None = None,
        force_ignore_biopsy_images_or_embeddings = False,
        biopsy_agg=None,
        heatmap_decoder: nn.Module | None = None,
        detach_biopsy_embeddings_in_heatmap_decoder: bool = False
    ):
        super().__init__()
        self.pnf_backbone = pnf_backbone  # raw ProstNFound model (shared)
        self.biopsy_encoder = ProstNFoundWrapperForMILBatches(pnf_backbone)
        self.sweep_encoder = ProstNFoundSweepEncoder(pnf_backbone)
        self.agg = agg
        self.biopsy_agg = biopsy_agg
        self.roll_embedding = roll_embedding
        self.type_embedding = nn.Embedding(2, embedding_dim)
        self.prompt_key_map = prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP)
        self.detached_token_classifier = detached_token_classifier
        self.ignore_biopsy_images_or_embeddings = force_ignore_biopsy_images_or_embeddings
        self.detach_biopsy_embeddings_in_heatmap_decoder = detach_biopsy_embeddings_in_heatmap_decoder

        # Optional projection if backbone output dim != transformer width
        backbone_dim = backbone_dim or embedding_dim
        self.input_proj = (
            nn.Linear(backbone_dim, embedding_dim)
            if backbone_dim != embedding_dim
            else nn.Identity()
        )

        # Optional heatmap decoder — deepcopy of the ProstNFound mask decoder,
        # fine-tuned independently from the frozen backbone.
        self.heatmap_decoder = heatmap_decoder
        if heatmap_decoder is not None:
            # Projects BERT-enriched biopsy token (D) → SAM prompt space (256)
            self.heatmap_context_proj = nn.Linear(embedding_dim, 256)

    @property
    def _raw_pnf(self):
        """Unwrap FrozenModuleWrapper to get the underlying ProstNFound model."""
        pnf = self.pnf_backbone
        if isinstance(pnf, FrozenModuleWrapper):
            return pnf.module
        return pnf

    @torch.no_grad()
    def extract_backbone_embeddings(self, *, biopsy_image, sweep, **kwargs):
        """Pre-extract backbone features for biopsies and sweeps.

        Useful for caching embeddings to disk so subsequent training runs
        skip the expensive backbone forward pass.

        Args:
            biopsy_image: list of ``(N_i, C, H, W)`` tensors.
            sweep: list of list of ``(T_j, C, H, W)`` tensors.
            **kwargs: prompt keys (``biopsy_psa``, ``biopsy_age``, …).

        Returns:
            dict with:
                - ``biopsy_backbone_embeddings``: list of ``(N_i, D_backbone)``
                - ``sweep_backbone_embeddings``: list of list of ``(T_j, D_backbone)``
        """
        prompts = _remap_prompts(kwargs, self.prompt_key_map, strict=False)

        # Biopsy embeddings — optionally also grab image feature maps
        if self.heatmap_decoder is not None:
            biopsy_embs, biopsy_img_feats = self.biopsy_encoder(
                biopsy_image, return_image_feats=True, **prompts
            )
        else:
            biopsy_embs = self.biopsy_encoder(biopsy_image, **prompts)
            biopsy_img_feats = None

        # Sweep embeddings
        sweep_embs = []
        for case_sweeps in sweep:
            if len(case_sweeps) == 0:
                sweep_embs.append([])
            else:
                sweep_embs.append(self.sweep_encoder(case_sweeps))

        out = {
            "biopsy_backbone_embeddings": biopsy_embs,
            "sweep_backbone_embeddings": sweep_embs,
        }
        if biopsy_img_feats is not None:
            out["biopsy_image_feats"] = biopsy_img_feats  # list[(N_i, 256, H, W)]
        return out

    def forward(
        self,
        *,
        biopsy_image=None,
        biopsy_roll_angle,
        sweep=None,
        sweep_roll_series,
        biopsy_backbone_embeddings=None,
        sweep_backbone_embeddings=None,
        biopsy_image_feats=None,
        output_attentions=False,
        **kwargs,
    ):
        """
        Args:
            biopsy_image: list of ``(N_i, C, H, W)`` tensors. Required unless
                ``biopsy_backbone_embeddings`` is provided.
            biopsy_roll_angle: list of ``(N_i,)`` tensors.
            sweep: list of list of ``(T_j, C, H, W)`` tensors. Required unless
                ``sweep_backbone_embeddings`` is provided.
            sweep_roll_series: list of list of ``(T_j,)`` tensors.
            biopsy_backbone_embeddings: Pre-extracted biopsy backbone features,
                list of ``(N_i, D_backbone)`` tensors. When provided,
                ``biopsy_image`` is ignored and the biopsy encoder is skipped.
            sweep_backbone_embeddings: Pre-extracted sweep backbone features,
                list of list of ``(T_j, D_backbone)`` tensors (outer list = cases,
                inner list = sweeps per case). When provided, ``sweep`` is
                ignored and the sweep encoder is skipped.
            biopsy_image_feats: Pre-extracted SAM neck image features,
                list of ``(N_i, 256, H, W)`` tensors. When provided alongside
                ``biopsy_backbone_embeddings``, the heatmap decoder can run
                even in the cached-embeddings regime.
        """
        if self.ignore_biopsy_images_or_embeddings: 
            biopsy_image = biopsy_backbone_embeddings = None

        batch_size = len(biopsy_roll_angle)
        prompts = _remap_prompts(kwargs, self.prompt_key_map, strict=False)

        # === Biopsy branch ===
        # When load_biopsies=false, biopsy_roll_angle will be a list of
        # empty (0,) tensors.  In that case we produce empty hidden states.
        biopsy_image_feats_live = None  # (N_i, C, H, W) per case; only set when heatmap_decoder is active
        if biopsy_backbone_embeddings is not None:
            # Use pre-extracted features — project to embedding_dim
            biopsy_hidden = [self.input_proj(h) for h in biopsy_backbone_embeddings]
        else:
            if biopsy_image is not None:
                if self.heatmap_decoder is not None:
                    biopsy_hidden, biopsy_image_feats_live = self.biopsy_encoder(
                        biopsy_image, return_image_feats=True, **prompts
                    )
                else:
                    biopsy_hidden = self.biopsy_encoder(biopsy_image, **prompts)
                biopsy_hidden = [self.input_proj(h) for h in biopsy_hidden]
            else:
                # No biopsy data at all — produce empty hidden states per case
                # We need embedding_dim from the model; derive from type_embedding.
                emb_dim = self.type_embedding.embedding_dim
                device = biopsy_roll_angle[0].device
                biopsy_hidden = [
                    torch.empty(0, emb_dim, device=device)
                    for _ in range(batch_size)
                ]

        # Snapshot pre-BERT biopsy features for detached token classification
        pre_bert_biopsy = [h.detach() for h in biopsy_hidden] if self.detached_token_classifier is not None else None

        # === Sweep branch ===
        # For each case, flatten all sweeps' frames into one sequence.
        sweep_hidden_per_case = []   # list[(T_total_i, D)]
        sweep_rolls_per_case = []    # list[(T_total_i,)]

        if sweep_backbone_embeddings is not None:
            # Use pre-extracted sweep features
            for i in range(batch_size):
                case_embs = sweep_backbone_embeddings[i]  # list of (T_j, D_backbone)
                case_rolls = sweep_roll_series[i]          # list of (T_j,)
                if len(case_embs) == 0:
                    device = biopsy_hidden[0].device
                    sweep_hidden_per_case.append(
                        torch.empty(0, biopsy_hidden[0].shape[-1], device=device)
                    )
                    sweep_rolls_per_case.append(torch.empty(0, device=device))
                else:
                    sweep_hidden_per_case.append(
                        self.input_proj(torch.cat(case_embs, dim=0))
                    )
                    sweep_rolls_per_case.append(torch.cat(case_rolls, dim=0))
        else:
            if sweep is None:
                raise ValueError(
                    "Either sweep or sweep_backbone_embeddings must be provided."
                )
            # sweep[i] is a list of (T_j, C, H, W) tensors (one per sweep).
            # sweep_roll_series[i] is a list of (T_j,) tensors.
            for i in range(batch_size):
                case_sweeps = sweep[i]          # list of (T_j, C, H, W)
                case_rolls = sweep_roll_series[i]  # list of (T_j,)

                if len(case_sweeps) == 0:
                    device = biopsy_hidden[0].device
                    sweep_hidden_per_case.append(
                        torch.empty(0, biopsy_hidden[0].shape[-1], device=device)
                    )
                    sweep_rolls_per_case.append(torch.empty(0, device=device))
                    continue

                # Encode all frames of all sweeps for this case together.
                # No per-frame prompts for sweeps (prompts are biopsy-level).
                frames_per_case = self.sweep_encoder(case_sweeps)  # list[(T_j, D_backbone)]
                sweep_hidden_per_case.append(
                    self.input_proj(torch.cat(frames_per_case, dim=0))
                )
                sweep_rolls_per_case.append(torch.cat(case_rolls, dim=0))

        # === Merge sequences + add embeddings ===
        combined_hidden = []       # list[(N_i + T_total_i, D)]
        biopsy_token_counts = []   # track how many biopsy tokens per case
        for i in range(batch_size):
            bh = biopsy_hidden[i]          # (N_i, D)  — may be (0, D)
            sh = sweep_hidden_per_case[i]  # (T_i, D)
            n_biopsy = bh.shape[0]
            n_sweep = sh.shape[0]
            emb_dim = self.type_embedding.embedding_dim
            device = bh.device if n_biopsy > 0 else sh.device
            biopsy_token_counts.append(n_biopsy)

            # Type embeddings
            if n_biopsy > 0:
                biopsy_type = self.type_embedding(
                    torch.full((n_biopsy,), self.BIOPSY_TYPE, device=device, dtype=torch.long)
                )
                biopsy_roll_emb = self.roll_embedding(biopsy_roll_angle[i])
                bh = bh + biopsy_type + biopsy_roll_emb

            if n_sweep > 0:
                sweep_type = self.type_embedding(
                    torch.full((n_sweep,), self.SWEEP_TYPE, device=device, dtype=torch.long)
                )
                sweep_roll_emb = self.roll_embedding(sweep_rolls_per_case[i])
                sh = sh + sweep_type + sweep_roll_emb

            # Biopsies first, then sweeps (so we know where biopsy tokens are)
            if n_biopsy > 0 and n_sweep > 0:
                combined_hidden.append(torch.cat([bh, sh], dim=0))
            elif n_biopsy > 0:
                combined_hidden.append(bh)
            elif n_sweep > 0:
                combined_hidden.append(sh)
            else:
                combined_hidden.append(bh)

        # === BERT aggregator ===
        agg_out = self.agg(combined_hidden, output_attentions=output_attentions)

        # === Extract biopsy-only token outputs ===
        biopsy_hidden_out = []
        for i in range(batch_size):
            n_b = biopsy_token_counts[i]
            token_h = agg_out["token_hidden_states"][i][:n_b]
            biopsy_hidden_out.append(token_h)

        # === Biopsy logits — from separate agg or main agg ===
        if self.biopsy_agg is not None:
            biopsy_agg_out = self.biopsy_agg(combined_hidden)
            biopsy_logits = [
                biopsy_agg_out["token_logits"][i][:biopsy_token_counts[i]]
                for i in range(batch_size)
            ]
        else:
            biopsy_logits = []
            for i in range(batch_size):
                n_b = biopsy_token_counts[i]
                if agg_out["token_logits"][i] is not None and not self.ignore_biopsy_images_or_embeddings:
                    biopsy_logits.append(agg_out["token_logits"][i][:n_b])
                else:
                    biopsy_logits.append(None)

        # === Detached biopsy-level logits (pre-BERT, no grad to aggregator) ===
        detached_biopsy_logits = None
        if self.detached_token_classifier is not None and pre_bert_biopsy is not None:
            detached_biopsy_logits = [
                self.detached_token_classifier(h) for h in pre_bert_biopsy
            ]

        out = {
            "logits": agg_out["logits"],
            "biopsy_logits": biopsy_logits,
            "biopsy_hidden": biopsy_hidden_out,
            "cls_hidden": agg_out["clstoken_hidden_state"],          # (B, D) for UMAP
            "biopsy_token_counts": biopsy_token_counts,              # list[int]
            "sweep_token_counts": [sh.shape[0] for sh in sweep_hidden_per_case],  # list[int]
        }
        if detached_biopsy_logits is not None:
            out["detached_biopsy_logits"] = detached_biopsy_logits
        if output_attentions and "attentions" in agg_out:
            out["attentions"] = agg_out["attentions"]  # tuple of (B, H, S, S)

        # === Heatmap decoder ===
        if self.heatmap_decoder is not None:
            # Resolve canvas: prefer cached feats from batch, fall back to live feats
            resolved_feats = biopsy_image_feats   # may be list of per-case tensors from batch
            if resolved_feats is None:
                resolved_feats = biopsy_image_feats_live   # produced by live encoder branch

            if resolved_feats is not None:
                # Gather all biopsies across the batch for a single batched decode.
                img_list = []
                ctx_list = []
                for i in range(batch_size):
                    n_b = biopsy_token_counts[i]
                    if n_b > 0:
                        img_list.append(resolved_feats[i])               # (N_i, C, H, W)
                        if not self.detach_biopsy_embeddings_in_heatmap_decoder:
                            ctx = self.heatmap_context_proj(biopsy_hidden_out[i])  # (N_i, 256)
                        else: 
                            ctx = self.heatmap_context_proj(biopsy_hidden_out[i].detach())
                        ctx_list.append(ctx)

                if img_list:
                    all_imgs = torch.cat(img_list, dim=0)           # (N_total, C, H, W)
                    all_ctx  = torch.cat(ctx_list, dim=0).unsqueeze(1)  # (N_total, 1, 256)
                    N_total  = all_imgs.shape[0]

                    # Null prompts — use frozen backbone's prompt encoder
                    raw_pnf = self._raw_pnf
                    with torch.no_grad():
                        sparse, dense = raw_pnf.medsam_model.prompt_encoder(None, None, None)
                    sparse = sparse.expand(N_total, -1, -1)       # (N_total, 0, 256)
                    dense  = dense.expand(N_total, -1, -1, -1)    # (N_total, 256, H_f, W_f)
                    if dense.shape[-2:] != all_imgs.shape[-2:]:
                        dense = F.interpolate(dense, size=all_imgs.shape[-2:])

                    # Inject BERT context as an extra sparse prompt token
                    sparse = torch.cat([sparse, all_ctx], dim=1)  # (N_total, 1, 256)

                    # Positional encoding for image embeddings
                    pe = raw_pnf.medsam_model.prompt_encoder.get_dense_pe()
                    if pe.shape[-2:] != all_imgs.shape[-2:]:
                        pe = F.interpolate(pe, size=all_imgs.shape[-2:])

                    heatmap_logits, _ = self.heatmap_decoder(
                        all_imgs, pe, sparse, dense, multimask_output=False,
                    )  # (N_total, 1, H_out, W_out)

                    # Split back into per-case tensors
                    biopsy_heatmaps = []
                    offset = 0
                    for i in range(batch_size):
                        n_b = biopsy_token_counts[i]
                        if n_b > 0:
                            biopsy_heatmaps.append(heatmap_logits[offset : offset + n_b])
                            offset += n_b
                        else:
                            biopsy_heatmaps.append(None)
                else:
                    biopsy_heatmaps = [None] * batch_size
            else:
                # No image canvas available (neither cached nor live)
                biopsy_heatmaps = [None] * batch_size
            out["biopsy_heatmaps"] = biopsy_heatmaps

        return out


# =============================================================================
# Compass factory functions
# =============================================================================


@register_model
def compass(
    freeze_backbone=False,
    num_classes=2,
    num_token_classes=2,
    pnf_cfg={"name": "prostnfound_plus_final"},
    pooling_strategy: str = "cls",
    prompt_key_map: dict | None = None,
    roll_embedding_type = "mlp",
    roll_hidden_dim: int = 128,
    backbone_dim: int | None = None,
    embedding_dim: int = 256,
    detached_token_classes: int | None = None,
    force_ignore_biopsy_images_or_embeddings = False,
    separate_biopsy_bert: bool = False,
    biopsy_bert_kw: dict | None = None,
    heatmap_decoder: bool = False,
    detach_biopsy_embeddings_in_heatmap_decoder = False,
    **bert_kwargs,
):
    """Unified biopsy + sweep MIL model with type & roll-angle PE.

    Uses a shared ProstNFound backbone for both biopsy images and sweep frames.
    Sweep frames are encoded per-frame (spatial pooling by ProstNFound), then
    all tokens (biopsy + sweep) are fed to a BERT aggregator with additive
    type and MLP roll-angle positional embeddings.

    Args:
        freeze_backbone: Freeze the shared ProstNFound backbone.
        roll_hidden_dim: Hidden dim of the roll-angle MLP.
        backbone_dim: Output dimension of the backbone.  If ``None``, defaults
            to ``embedding_dim`` (no projection).  When set to a different value,
            a learned linear projection maps backbone outputs to
            ``embedding_dim`` before type/roll embeddings and the aggregator.
        embedding_dim: Transformer / aggregator width.
        detached_token_classes: If set, adds a detached (pre-BERT) token
            classifier head with this many output classes.  Biopsy features are
            ``.detach()``-ed before classification so gradients only train the
            head, not the aggregator.
        prompt_key_map: Mapping from data keys to ProstNFound prompt kwarg names.
        **bert_kwargs: Forwarded to BertBasedMILAggregatorForTokenAndSequenceClassification.
    """
    from src.modeling import create_model as _create_model

    pnf = _create_model(**pnf_cfg)

    # Deepcopy mask decoder BEFORE freezing so its weights are independent
    # and remain trainable even when the backbone is frozen.
    heatmap_dec = None
    if heatmap_decoder:
        heatmap_dec = copy.deepcopy(pnf.medsam_model.mask_decoder)

    if freeze_backbone:
        pnf = FrozenModuleWrapper(pnf)

    from src.modeling.multiple_instance_learning.bert_based_mil_aggregator import (
        BertBasedMILAggregatorForTokenAndSequenceClassification,
    )

    bert_kw_defaults = dict(
        num_attention_heads=8, num_hidden_layers=2, intermediate_size=1024
    )
    bert_kw_defaults.update(bert_kwargs)

    if separate_biopsy_bert:
        # Main agg: case logits only (no token head)
        agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_token_classes=None,
            pooling_strategy=pooling_strategy,
            **bert_kw_defaults,
        )
        # Biopsy agg: token logits only (no seq head)
        biopsy_bert_kw_resolved = dict(bert_kw_defaults)
        if biopsy_bert_kw:
            biopsy_bert_kw_resolved.update(biopsy_bert_kw)
        biopsy_agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
            embedding_dim=embedding_dim,
            num_classes=None,
            num_token_classes=num_token_classes,
            pooling_strategy=pooling_strategy,
            **biopsy_bert_kw_resolved,
        )
    else:
        agg = BertBasedMILAggregatorForTokenAndSequenceClassification(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_token_classes=num_token_classes,
            pooling_strategy=pooling_strategy,
            **bert_kw_defaults,
        )
        biopsy_agg = None

    if roll_embedding_type == "mlp":
        roll_emb = MLPRollAngleEmbedding(
            embedding_dim=embedding_dim, hidden_dim=roll_hidden_dim
        )
    elif roll_embedding_type == "sinusoidal":
        roll_emb = SinusoidalRollAngleEmbedding(
            embedding_dim=embedding_dim,
        )
    elif roll_embedding_type == "none":
        roll_emb = ZeroRollAngleEmbedding(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown roll embedding type {roll_embedding_type}")

    # Optional detached (pre-BERT) token classifier
    detached_tok_clf = None
    if detached_token_classes is not None:
        detached_tok_clf = nn.Linear(embedding_dim, detached_token_classes)

    return Compass(
        pnf_backbone=pnf,
        agg=agg,
        roll_embedding=roll_emb,
        embedding_dim=embedding_dim,
        backbone_dim=backbone_dim,
        prompt_key_map=prompt_key_map or dict(_DEFAULT_PROMPT_KEY_MAP),
        detached_token_classifier=detached_tok_clf,
        force_ignore_biopsy_images_or_embeddings=force_ignore_biopsy_images_or_embeddings,
        biopsy_agg=biopsy_agg,
        heatmap_decoder=heatmap_dec,
        detach_biopsy_embeddings_in_heatmap_decoder=detach_biopsy_embeddings_in_heatmap_decoder,
    )


