"""LIBERO dataset for unified action prediction + world model training.

Loads preprocessed HDF5 files produced by preprocess_libero.py.

Expected HDF5 structure (one file per task):
    /image_current:       (N, 256, 256, 3) uint8 HWC
    /image_future:        (N, 256, 256, 3) uint8 HWC
    /fast_tokens:         variable-length int32 (h5py vlen_dtype)
    /continuous_actions:  (N, H, action_dim) float32  [optional, for debug]
"""

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from module import PAD_TOKEN_ID

# ImageNet normalization stats (matches spt.data.dataset_stats.ImageNet / torchvision)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _preprocess_image(img_uint8: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """uint8 HWC (256,256,3) → float32 CHW (3, img_size, img_size), ImageNet-normalized.

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
    """Loads preprocessed LIBERO HDF5 files for unified training.

    Supports multiple HDF5 files in a directory (one per task or per split).
    Each file contains chunk-aligned samples with image pairs and FAST tokens.

    Args:
        hdf5_dir: directory containing .hdf5 files.
        max_action_tokens: pad/truncate FAST tokens to this length. Default 35.
        img_size: resize images to this resolution. Default 224.
    """

    def __init__(
        self,
        hdf5_dir: str,
        max_action_tokens: int = 35,
        img_size: int = 224,
    ):
        self.max_action_tokens = max_action_tokens
        self.img_size = img_size

        # Discover all HDF5 files and build a global index
        hdf5_dir = Path(hdf5_dir)
        self.files = sorted(hdf5_dir.glob("*.hdf5"))
        if not self.files:
            self.files = sorted(hdf5_dir.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5/.h5 files found in {hdf5_dir}")

        # Build (file_idx, sample_idx) mapping for global indexing
        self._index = []  # list of (file_path, local_idx)
        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                n_samples = f["image_current"].shape[0]
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

        # Images: uint8 HWC (256, 256, 3) → float32 CHW (3, 224, 224) ImageNet-normalized
        img_current = _preprocess_image(f["image_current"][local_idx], self.img_size)
        img_future = _preprocess_image(f["image_future"][local_idx], self.img_size)

        # FAST tokens: variable-length → pad to max_action_tokens
        raw_tokens = f["fast_tokens"][local_idx]  # numpy array, variable length
        raw_tokens = np.array(raw_tokens, dtype=np.int64)
        token_len = min(len(raw_tokens), self.max_action_tokens)

        fast_tokens = np.full(self.max_action_tokens, PAD_TOKEN_ID, dtype=np.int64)
        fast_tokens[:token_len] = raw_tokens[:token_len]

        return {
            "pixels_current": img_current,                             # (3, 224, 224) float32
            "pixels_future": img_future,                               # (3, 224, 224) float32
            "fast_tokens": torch.from_numpy(fast_tokens),              # (max_action_tokens,) long
            "fast_lengths": torch.tensor(token_len, dtype=torch.long), # scalar
        }

    def __del__(self):
        for f in self._open_files.values():
            f.close()
