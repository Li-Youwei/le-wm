"""Action token codecs for LIBERO VLA training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


FAST_CODEC = "fast"
WORLDVLA_BINS_CODEC = "worldvla_bins"
FAST_VOCAB_SIZE = 1024
DEFAULT_WORLDVLA_BINS = 256


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, str):
        if key == "type":
            return cfg
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass(frozen=True)
class ActionCodec:
    """Small value object shared by train, preprocess, and eval."""

    name: str
    vocab_size: int
    token_min: float = -1.0
    token_max: float = 1.0

    @property
    def bos_token_id(self) -> int:
        return self.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.vocab_size + 1

    @property
    def pad_token_id(self) -> int:
        return self.vocab_size + 2

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size + 3

    @property
    def action_head_size(self) -> int:
        return self.vocab_size + 2

    def hdf5_attrs(self) -> dict[str, Any]:
        return {
            "action_codec_type": self.name,
            "action_vocab_size": int(self.vocab_size),
            "action_num_bins": 0,
            "action_token_min": float(self.token_min),
            "action_token_max": float(self.token_max),
        }


@dataclass(frozen=True)
class FastActionCodec(ActionCodec):
    def __init__(self) -> None:
        super().__init__(name=FAST_CODEC, vocab_size=FAST_VOCAB_SIZE)

    def decode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        processor: Any,
        time_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        from fast_utils import fast_decode

        return fast_decode(
            token_ids,
            lengths,
            processor,
            time_horizon=time_horizon,
            action_dim=action_dim,
        )


@dataclass(frozen=True)
class WorldVLAMeta(ActionCodec):
    num_bins: int = DEFAULT_WORLDVLA_BINS
    bin_min: float = -1.0
    bin_max: float = 1.0

    @property
    def bin_width(self) -> float:
        return (self.bin_max - self.bin_min) / self.num_bins

    def hdf5_attrs(self) -> dict[str, Any]:
        attrs = super().hdf5_attrs()
        attrs["action_num_bins"] = int(self.num_bins)
        return attrs


class WorldVLABinActionCodec(WorldVLAMeta):
    """WorldVLA-style scalar action binning over normalized actions."""

    def __init__(
        self,
        *,
        num_bins: int = DEFAULT_WORLDVLA_BINS,
        bin_min: float = -1.0,
        bin_max: float = 1.0,
    ) -> None:
        if num_bins <= 1:
            raise ValueError(f"num_bins must be > 1, got {num_bins}")
        if not bin_max > bin_min:
            raise ValueError(
                f"bin_max must be greater than bin_min, got {bin_min}..{bin_max}"
            )
        super().__init__(
            name=WORLDVLA_BINS_CODEC,
            vocab_size=int(num_bins),
            token_min=float(bin_min),
            token_max=float(bin_max),
            num_bins=int(num_bins),
            bin_min=float(bin_min),
            bin_max=float(bin_max),
        )

    def encode(self, action_chunks: np.ndarray) -> list[np.ndarray]:
        """Encode normalized ``(N,H,D)`` chunks into flattened scalar bin ids."""
        chunks = np.asarray(action_chunks, dtype=np.float32)
        if chunks.ndim != 3:
            raise ValueError(
                "WorldVLABinActionCodec.encode expected (N,H,D), got "
                f"{chunks.shape}"
            )
        clipped = np.clip(chunks, self.bin_min, self.bin_max)
        scaled = np.floor((clipped - self.bin_min) / self.bin_width)
        ids = np.clip(scaled, 0, self.num_bins - 1).astype(np.int32)
        return [row.reshape(-1).copy() for row in ids]

    def decode(
        self,
        token_ids: torch.Tensor | np.ndarray,
        lengths: torch.Tensor | np.ndarray,
        *,
        time_horizon: int,
        action_dim: int,
        processor: Any | None = None,
    ) -> np.ndarray:
        """Decode flattened scalar bin ids to bin centers in normalized space."""
        del processor
        if isinstance(token_ids, torch.Tensor):
            tokens_np = token_ids.detach().cpu().numpy()
        else:
            tokens_np = np.asarray(token_ids)
        if isinstance(lengths, torch.Tensor):
            lengths_np = lengths.detach().cpu().numpy()
        else:
            lengths_np = np.asarray(lengths)

        expected = int(time_horizon) * int(action_dim)
        decoded: list[np.ndarray] = []
        for i in range(tokens_np.shape[0]):
            k = int(lengths_np[i])
            if k != expected:
                raise ValueError(
                    "worldvla_bins decode expected exactly "
                    f"{expected} tokens (H={time_horizon}, D={action_dim}), "
                    f"got {k} for sample {i}."
                )
            ids = np.asarray(tokens_np[i, :k], dtype=np.int64)
            ids = np.clip(ids, 0, self.num_bins - 1)
            centers = self.bin_min + (ids.astype(np.float64) + 0.5) * self.bin_width
            decoded.append(centers.reshape(time_horizon, action_dim))
        return np.stack(decoded, axis=0).astype(np.float32)


def build_action_codec(cfg: Any = None) -> ActionCodec:
    codec_type_raw = _cfg_get(cfg, "type", FAST_CODEC)
    if isinstance(codec_type_raw, bytes):
        codec_type_raw = codec_type_raw.decode("utf-8")
    codec_type = str(codec_type_raw).lower()
    if codec_type == FAST_CODEC:
        return FastActionCodec()
    if codec_type == WORLDVLA_BINS_CODEC:
        return WorldVLABinActionCodec(
            num_bins=int(_cfg_get(cfg, "num_bins", DEFAULT_WORLDVLA_BINS)),
            bin_min=float(_cfg_get(cfg, "bin_min", -1.0)),
            bin_max=float(_cfg_get(cfg, "bin_max", 1.0)),
        )
    raise ValueError(
        f"Unknown action codec '{codec_type}'. Expected '{FAST_CODEC}' or "
        f"'{WORLDVLA_BINS_CODEC}'."
    )


def build_action_codec_from_attrs(attrs: Any, fallback: Any = None) -> ActionCodec:
    codec_type = attrs.get("action_codec_type", None) if attrs is not None else None
    if codec_type is None:
        return build_action_codec(fallback)
    return build_action_codec(
        {
            "type": codec_type,
            "num_bins": int(attrs.get("action_num_bins", DEFAULT_WORLDVLA_BINS)),
            "bin_min": float(attrs.get("action_token_min", -1.0)),
            "bin_max": float(attrs.get("action_token_max", 1.0)),
        }
    )


def action_codecs_compatible(left: ActionCodec, right: ActionCodec) -> bool:
    """Return whether two codec configs describe the same token semantics."""
    if left.name != right.name or int(left.vocab_size) != int(right.vocab_size):
        return False
    if not np.isclose(left.token_min, right.token_min):
        return False
    if not np.isclose(left.token_max, right.token_max):
        return False
    if left.name == WORLDVLA_BINS_CODEC:
        return (
            int(getattr(left, "num_bins", left.vocab_size))
            == int(getattr(right, "num_bins", right.vocab_size))
            and np.isclose(
                float(getattr(left, "bin_min", left.token_min)),
                float(getattr(right, "bin_min", right.token_min)),
            )
            and np.isclose(
                float(getattr(left, "bin_max", left.token_max)),
                float(getattr(right, "bin_max", right.token_max)),
            )
        )
    return True
