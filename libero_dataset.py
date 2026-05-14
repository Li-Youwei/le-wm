"""LIBERO dataset for VLA baseline training.

Loads preprocessed HDF5 files produced by preprocess_libero.py.

Expected HDF5 structure (one file per task):
    /image_agent:         (N, H_img, W_img, 3) uint8 HWC  — agentview at chunk start
    /image_hand:          (N, H_img, W_img, 3) uint8 HWC  — eye-in-hand at chunk start
    /proprio:             (N, 9) float64                   — ee_pos(3)+xyzw_quat(4)+gripper_raw(2)
    /fast_tokens:         variable-length int32 (h5py vlen_dtype)
    /continuous_actions:  (N, H, action_dim) float32  [optional, for debug]
    attrs:
        language_instruction: str — task description for T5 encoding
        chunk_size:  int          — H (raw steps per chunk)
        chunk_stride: int         — sliding-window stride (default 1)
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
            img.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
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
        use_state_prediction: bool = False,
    ):
        self.max_action_tokens = max_action_tokens
        self.max_lang_tokens = max_lang_tokens
        self.img_size = img_size
        self.use_language = use_language
        self.use_state_prediction = use_state_prediction

        # Discover all HDF5 files and build a global index
        hdf5_dir = Path(hdf5_dir)
        self.files = sorted(hdf5_dir.glob("*.hdf5"))
        if not self.files:
            self.files = sorted(hdf5_dir.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5/.h5 files found in {hdf5_dir}")

        # Stable task_id per file (used for per-task balanced sampling +
        # per-task validation CE logging). Order is `sorted(self.files)`.
        self._file_to_task_id: dict[Path, int] = {
            fpath: i for i, fpath in enumerate(self.files)
        }

        # T5 tokenizer (only when language is enabled)
        self.t5_tokenizer = (
            T5Tokenizer.from_pretrained("t5-small") if use_language else None
        )

        # Pre-tokenize language instructions (one per file) and build global index
        self._index: list[tuple[Path, int]] = []
        self._lang_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
        # Per-sample demo_idx cache — avoids re-opening every HDF5 in train.py
        # for the train/val split. Same length as self._index, aligned 1:1.
        self._demo_ids: list[int] = []
        # Counters for the 3-level balanced WeightedRandomSampler.
        self.n_demos_per_task: dict[Path, int] = {}
        self.n_chunks_per_demo: dict[tuple[Path, int], int] = {}

        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                # All preprocessed HDF5 produced by the current pipeline have
                # `image_agent` (agent view) and `image_hand` (eye-in-hand)
                # at the top level. The old single-view name `image_current`
                # had no `image_hand` companion, so any file with that legacy
                # name would already fail at the hand-image load below — the
                # fallback was effectively dead. Removed for clarity.
                if "image_agent" not in f:
                    raise KeyError(f"No 'image_agent' in {fpath}")
                n_samples = f["image_agent"].shape[0]

                # When state prediction is enabled, all three future fields
                # MUST exist in the HDF5 — fail fast with a clear message
                # rather than silently broadcasting current frame as future.
                # (Old preprocessed HDF5 from the frozen baseline does not
                # have these; users must re-run preprocess_libero.py.)
                if use_state_prediction:
                    missing = [
                        key
                        for key in (
                            "image_agent_future",
                            "image_hand_future",
                            "proprio_future",
                        )
                        if key not in f
                    ]
                    if missing:
                        raise KeyError(
                            f"State-prediction enabled but {fpath.name} lacks "
                            f"future fields: {missing}. Re-run preprocess_libero.py "
                            "to regenerate (it will add the t+H frames)."
                        )

                # demo_idx is always written by preprocess_libero.py. Read it
                # once per file here so train.py can split by (file, demo)
                # tuple and the sampler can balance per-(task, demo) without
                # re-opening files.
                if "demo_idx" not in f:
                    raise KeyError(
                        f"No 'demo_idx' in {fpath} — preprocess_libero.py output "
                        "is incomplete. Re-run preprocessing."
                    )
                demo_arr = f["demo_idx"][()]
                if demo_arr.shape[0] != n_samples:
                    raise ValueError(
                        f"{fpath.name}: demo_idx has {demo_arr.shape[0]} entries "
                        f"but image_agent has {n_samples}. Preprocessing produced "
                        "an inconsistent file."
                    )

                # Read language instruction only if needed
                if use_language:
                    lang_str = f.attrs.get("language_instruction", "")
                    if not lang_str:
                        raise ValueError(
                            f"Empty language_instruction in {fpath}. This makes "
                            "LIBERO multi-task training ambiguous, especially for "
                            "libero_object where the target object is language "
                            "conditioned. Re-run preprocess_libero.py with a "
                            "version that records task language, or set "
                            "data.dataset.use_language=False only for an explicit "
                            "no-language ablation."
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
                    tok_out["input_ids"].squeeze(0),  # (max_lang_tokens,) long
                    tok_out["attention_mask"].squeeze(0),  # (max_lang_tokens,) long
                )

            # Per-task / per-demo counters from demo_arr.
            unique_demos, counts = np.unique(demo_arr, return_counts=True)
            self.n_demos_per_task[fpath] = int(unique_demos.size)
            for d, c in zip(unique_demos.tolist(), counts.tolist()):
                self.n_chunks_per_demo[(fpath, int(d))] = int(c)
            self._demo_ids.extend(int(d) for d in demo_arr.tolist())

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

        # Images: uint8 HWC → float32 CHW (3, 224, 224) ImageNet-normalized.
        # Existence of both keys was checked in __init__.
        img_agent = _preprocess_image(f["image_agent"][local_idx], self.img_size)
        if "image_hand" not in f:
            raise KeyError(
                f"No 'image_hand' dataset in {fpath}. Re-run preprocess_libero.py."
            )
        img_hand = _preprocess_image(f["image_hand"][local_idx], self.img_size)

        # Proprioception: (9,) float64 → float32 tensor
        # [ee_pos(3) + xyzw_quat(4) + gripper_raw(2)] — see preprocess_libero.py
        proprio = torch.from_numpy(np.array(f["proprio"][local_idx], dtype=np.float32))

        # FAST tokens: variable-length → pad to max_action_tokens
        raw_tokens = f["fast_tokens"][local_idx]
        raw_tokens = np.array(raw_tokens, dtype=np.int64)
        if len(raw_tokens) > self.max_action_tokens:
            logger.warning(
                "Sample %d (file=%s, local=%d): FAST token length %d exceeds "
                "max_action_tokens=%d, truncating.",
                idx,
                fpath.name,
                local_idx,
                len(raw_tokens),
                self.max_action_tokens,
            )
        token_len = min(len(raw_tokens), self.max_action_tokens)

        fast_tokens = np.full(self.max_action_tokens, PAD_TOKEN_ID, dtype=np.int64)
        fast_tokens[:token_len] = raw_tokens[:token_len]

        # Direct gripper-command supervision target — bypasses FAST tokenization
        # to give the gripper dim a clean signal that doesn't get diluted by the
        # 6 spatial dims when FAST jointly BPE-encodes them.
        # `continuous_actions` is normalized to [-1, 1] per task. Eval maps the
        # aux prediction back through action_low/high before writing dim 6, so
        # this target must stay in the same normalized space as FAST.
        grip_seq = np.array(f["continuous_actions"][local_idx, :, 6], dtype=np.float32)

        item = {
            "pixels_agent": img_agent,  # (3, 224, 224)
            "pixels_hand": img_hand,  # (3, 224, 224)
            "proprio": proprio,  # (9,)
            "fast_tokens": torch.from_numpy(fast_tokens),  # (max_action_tokens,)
            "fast_lengths": torch.tensor(token_len, dtype=torch.long),  # scalar
            "task_id": torch.tensor(
                self._file_to_task_id[fpath],
                dtype=torch.long,
            ),  # scalar — index into sorted(self.files), stable across runs
            "gripper_seq": torch.from_numpy(grip_seq),  # (H,) [-1, 1]
        }

        # Language tokens (only when enabled)
        if self.use_language:
            lang_ids, lang_mask = self._lang_cache[fpath]
            item["lang_input_ids"] = lang_ids  # (max_lang_tokens,)
            item["lang_attention_mask"] = lang_mask  # (max_lang_tokens,)

        # Future-frame state-prediction targets (only when enabled).
        # Existence was asserted in __init__; here we just read them.
        if self.use_state_prediction:
            item["pixels_agent_future"] = _preprocess_image(
                f["image_agent_future"][local_idx],
                self.img_size,
            )
            item["pixels_hand_future"] = _preprocess_image(
                f["image_hand_future"][local_idx],
                self.img_size,
            )
            item["proprio_future"] = torch.from_numpy(
                np.array(f["proprio_future"][local_idx], dtype=np.float32),
            )

        return item

    def get_sampler_weights(self) -> torch.Tensor:
        """Per-sample weights for 3-level balanced WeightedRandomSampler.

        Each sample i gets weight
            w_i = 1 / (n_tasks * n_demos[task_i] * n_chunks[(task_i, demo_i)])

        so the marginal probability of each task is 1/n_tasks (uniform across
        the 4 LIBERO suites × 40 tasks), the conditional probability of each
        demo within a task is 1/n_demos[task] (uniform across demos), and
        within a demo every chunk is equally likely (uniform across time).

        Returns:
            weights: float64 tensor of shape (len(self),) — pass directly to
                torch.utils.data.WeightedRandomSampler(weights=..., ...).
        """
        n_tasks = len(self.files)
        weights = torch.empty(len(self._index), dtype=torch.float64)
        for i, (fpath, _local_idx) in enumerate(self._index):
            demo_id = self._demo_ids[i]
            n_d = self.n_demos_per_task[fpath]
            n_c = self.n_chunks_per_demo[(fpath, demo_id)]
            weights[i] = 1.0 / (n_tasks * n_d * n_c)
        return weights

    def __del__(self):
        for f in getattr(self, "_open_files", {}).values():
            f.close()
