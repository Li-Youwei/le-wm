"""Visual encoder builders for LeWM LIBERO runs."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


class HFVisionBackbone(nn.Module):
    """Thin wrapper around HuggingFace vision models.

    The rest of the repo expects a ViT-like module whose forward returns an
    object with ``last_hidden_state`` and accepts ``interpolate_pos_encoding``.
    HuggingFace vision backbones are not uniform about that keyword, so this
    wrapper keeps the JEPA call site stable.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = True):
        try:
            out = self.model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
        except TypeError:
            out = self.model(pixel_values=pixel_values)
        if not hasattr(out, "last_hidden_state"):
            raise TypeError(
                f"{self.model.__class__.__name__} output has no last_hidden_state; "
                "choose a vision backbone that returns token features."
            )
        return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if cfg is not None else default


def _hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "embed_dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        return _hidden_size(vision_config)
    raise AttributeError(f"Could not infer hidden size from config: {config}")


def _offline_default() -> bool:
    return os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get(
        "HF_HUB_OFFLINE"
    ) == "1"


def build_visual_encoder(cfg: Any, spt_module: Any) -> tuple[nn.Module, int, bool]:
    """Build the visual encoder.

    Returns:
        encoder: nn.Module returning ``last_hidden_state``.
        hidden_dim: token feature width before the projector.
        freeze_encoder: whether JEPA should keep the encoder in eval/no-grad mode.
    """
    vision_cfg = _cfg_get(cfg, "vision_encoder", {}) or {}
    source = str(_cfg_get(vision_cfg, "source", "spt_vit")).lower()

    if source == "spt_vit":
        encoder = spt_module.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=bool(_cfg_get(vision_cfg, "pretrained", False)),
            use_mask_token=False,
        )
        return encoder, int(encoder.config.hidden_size), bool(
            _cfg_get(vision_cfg, "freeze", False)
        )

    if source != "hf":
        raise ValueError(
            f"vision_encoder.source must be 'spt_vit' or 'hf', got {source!r}"
        )

    from transformers import AutoModel

    model_name = _cfg_get(vision_cfg, "model_name_or_path", None)
    if not model_name:
        raise ValueError(
            "vision_encoder.source='hf' requires vision_encoder.model_name_or_path"
        )
    local_files_only = bool(_cfg_get(vision_cfg, "local_files_only", _offline_default()))
    trust_remote_code = bool(_cfg_get(vision_cfg, "trust_remote_code", False))
    print(
        "[vision_encoder] loading HF backbone "
        f"{model_name!r} local_files_only={local_files_only}"
    )
    model = AutoModel.from_pretrained(
        str(model_name),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    encoder = HFVisionBackbone(model)
    freeze = bool(_cfg_get(vision_cfg, "freeze", True))
    if freeze:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    hidden_dim = _hidden_size(encoder.config)
    print(f"[vision_encoder] hidden_dim={hidden_dim}, freeze={freeze}")
    return encoder, hidden_dim, freeze
