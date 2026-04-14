"""LIBERO dataset for VLA baseline training.

Loads preprocessed HDF5 files produced by preprocess_libero.py.

Expected HDF5 structure (one file per task):
    /image_agent:         (N, H_img, W_img, 3) uint8 HWC  — agentview at chunk start
    /image_hand:          (N, H_img, W_img, 3) uint8 HWC  — eye-in-hand at chunk start
    /proprio:             (N, 8) float64                   — ee_pos(3)+ee_quat(4)+gripper(1)
    /fast_tokens:         variable-length int32 (h5py vlen_dtype)
    /continuous_actions:  (N, H, action_dim) float32  [optional, for debug]
    attrs:
        language_instruction: str — task description for T5 encoding
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from module import PAD_TOKEN_ID

logger = logging.getLogger(__name__)

# ImageNet normalization stats (matches spt.data.dataset_stats.ImageNet / torchvision)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess_image(img_uint8: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """uint8 HWC (H,W,3) → float32 CHW (3, img_size, img_size), ImageNet-normalized.

    Pipeline: uint8→float32(/255) → HWC→CHW → resize → ImageNet normalize.
    """
    # uint8 HWC → float32 CHW
    img = torch.from_numpy(img_uint8).float().div_(255.0)  # (H, W, 3)
    img = img.permute(2, 0, 1)  # (3, H, W)

    # Resize to img_size (bilinear, same as torchvision default)
    if img.shape[1] != img_size or img.shape[2] != img_size:
        img = F.interpolate(
            img.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False
        ).squeeze(0)

    # ImageNet normalize
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img


class LiberoDataset(Dataset):
    """Loads preprocessed LIBERO HDF5 files for VLA baseline training.

    Supports multiple HDF5 files in a directory (one per task or per split).
    Each file contains chunk-aligned samples with dual-view images, proprioception,
    and FAST tokens. Language instruction is stored per-file and tokenized at init.

    Args:
        hdf5_dir: directory containing .hdf5/.h5 files.
        max_action_tokens: pad/truncate FAST tokens to this length.
        max_lang_tokens: pad/truncate language tokens to this length.
        img_size: resize images to this resolution.
    """

    def __init__(
        self,
        hdf5_dir: str,
        max_action_tokens: int = 80,
        max_lang_tokens: int = 25,
        img_size: int = 224,
        use_language: bool = True,
    ):
        self.max_action_tokens = max_action_tokens
        self.max_lang_tokens = max_lang_tokens
        self.img_size = img_size
        self.use_language = use_language

        # Discover all HDF5 files and build a global index
        hdf5_dir = Path(hdf5_dir)
        self.files = sorted(hdf5_dir.glob("*.hdf5"))
        if not self.files:
            self.files = sorted(hdf5_dir.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5/.h5 files found in {hdf5_dir}")

        # T5 tokenizer (only when language is enabled)
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small") if use_language else None

        # Pre-tokenize language instructions (one per file) and build global index
        self._index: list[tuple[Path, int]] = []
        self._lang_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}

        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                # Determine sample count — try new key first, fall back to old
                if "image_agent" in f:
                    n_samples = f["image_agent"].shape[0]
                elif "image_current" in f:
                    n_samples = f["image_current"].shape[0]
                else:
                    raise KeyError(f"No image_agent or image_current in {fpath}")

                # Read language instruction only if needed
                if use_language:
                    lang_str = f.attrs.get("language_instruction", "")
                    if not lang_str:
                        logger.warning(
                            "No language_instruction attr in %s, using empty string", fpath.name
                        )
                else:
                    lang_str = None

            # Tokenize language instruction only if needed
            if use_language:
                tok_out = self.t5_tokenizer(
                    lang_str,
                    max_length=self.max_lang_tokens,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self._lang_cache[fpath] = (
                    tok_out["input_ids"].squeeze(0),        # (max_lang_tokens,) long
                    tok_out["attention_mask"].squeeze(0),   # (max_lang_tokens,) long
                )

            self._index.extend([(fpath, i) for i in range(n_samples)])

        # Lazy file handles (opened per-worker in DataLoader)
        self._open_files: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _get_file(self, fpath: Path) -> h5py.File:
        """Lazy-open HDF5 file (one handle per worker process)."""
        if fpath not in self._open_files:
            self._open_files[fpath] = h5py.File(fpath, "r")
        return self._open_files[fpath]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fpath, local_idx = self._index[idx]
        f = self._get_file(fpath)

        # Images: uint8 HWC → float32 CHW (3, 224, 224) ImageNet-normalized
        # Handle both new and old HDF5 key names
        agent_key = "image_agent" if "image_agent" in f else "image_current"
        img_agent = _preprocess_image(f[agent_key][local_idx], self.img_size)

        hand_key = "image_hand"
        if hand_key not in f:
            raise KeyError(f"No '{hand_key}' dataset in {fpath}. "
                           f"Re-run preprocess_libero.py to generate new format.")
        img_hand = _preprocess_image(f[hand_key][local_idx], self.img_size)

        # Proprioception: (8,) float64 → float32 tensor
        proprio = torch.from_numpy(np.array(f["proprio"][local_idx], dtype=np.float32))

        # FAST tokens: variable-length → pad to max_action_tokens
        raw_tokens = f["fast_tokens"][local_idx]
        raw_tokens = np.array(raw_tokens, dtype=np.int64)
        if len(raw_tokens) > self.max_action_tokens:
            logger.warning(
                "Sample %d (file=%s, local=%d): FAST token length %d exceeds "
                "max_action_tokens=%d, truncating.",
                idx, fpath.name, local_idx, len(raw_tokens), self.max_action_tokens,
            )
        token_len = min(len(raw_tokens), self.max_action_tokens)

        fast_tokens = np.full(self.max_action_tokens, PAD_TOKEN_ID, dtype=np.int64)
        fast_tokens[:token_len] = raw_tokens[:token_len]

        item = {
            "pixels_agent": img_agent,                                  # (3, 224, 224)
            "pixels_hand": img_hand,                                    # (3, 224, 224)
            "proprio": proprio,                                         # (8,)
            "fast_tokens": torch.from_numpy(fast_tokens),               # (max_action_tokens,)
            "fast_lengths": torch.tensor(token_len, dtype=torch.long),  # scalar
        }

        # Language tokens (only when enabled)
        if self.use_language:
            lang_ids, lang_mask = self._lang_cache[fpath]
            item["lang_input_ids"] = lang_ids                           # (max_lang_tokens,)
            item["lang_attention_mask"] = lang_mask                     # (max_lang_tokens,)

        return item

    def __del__(self):
        for f in getattr(self, "_open_files", {}).values():
            f.close()
