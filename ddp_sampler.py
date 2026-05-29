"""DDP-aware samplers used by the LIBERO training loop."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


def _resolve_rank_info(
    num_replicas: int | None,
    rank: int | None,
) -> tuple[int, int]:
    if num_replicas is not None and rank is not None:
        return int(num_replicas), int(rank)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        resolved_replicas = torch.distributed.get_world_size()
        resolved_rank = torch.distributed.get_rank()
    else:
        resolved_replicas = int(os.environ.get("WORLD_SIZE", "1"))
        resolved_rank = int(os.environ.get("RANK", "0"))

    if num_replicas is not None:
        resolved_replicas = int(num_replicas)
    if rank is not None:
        resolved_rank = int(rank)
    return resolved_replicas, resolved_rank


def _partition_sizes(
    num_samples: int,
    num_replicas: int,
    drop_last: bool,
) -> tuple[int, int]:
    if drop_last:
        samples_per_rank = num_samples // num_replicas
    else:
        samples_per_rank = int(math.ceil(num_samples / num_replicas))
    total_size = samples_per_rank * num_replicas
    return samples_per_rank, total_size


class DistributedWeightedSampler(Sampler[int]):
    """Weighted replacement sampler that shards one shared draw across ranks.

    Each rank uses the same seed and epoch to generate a single global weighted
    index stream, then takes a rank-strided slice. This keeps the existing
    task/demo/time weighting while avoiding the "all ranks draw the same batch"
    failure mode of using independent WeightedRandomSampler instances in DDP.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        replacement: bool = True,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._iteration = 0

        if self.weights.dim() != 1:
            raise ValueError("weights must be a 1D tensor")
        if self.num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {self.num_samples}")
        if self.weights.numel() == 0 and self.num_samples > 0:
            raise ValueError("weights must be non-empty when num_samples > 0")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._iteration = 0

    def __iter__(self) -> Iterator[int]:
        num_replicas, rank = _resolve_rank_info(self.num_replicas, self.rank)
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        _samples_per_rank, total_size = _partition_sizes(
            self.num_samples,
            num_replicas,
            self.drop_last,
        )
        if not self.replacement and total_size > self.weights.numel():
            raise ValueError(
                "replacement=False cannot draw more samples than weights length: "
                f"total_size={total_size}, len(weights)={self.weights.numel()}"
            )

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003 + self._iteration)
        indices = torch.multinomial(
            self.weights,
            total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        self._iteration += 1
        return iter(indices[rank:total_size:num_replicas])

    def __len__(self) -> int:
        num_replicas, _rank = _resolve_rank_info(self.num_replicas, self.rank)
        samples_per_rank, _total_size = _partition_sizes(
            self.num_samples,
            num_replicas,
            self.drop_last,
        )
        return samples_per_rank


class DistributedIndexSampler(Sampler[int]):
    """DDP-safe index sampler for validation and overfit loaders."""

    def __init__(
        self,
        dataset_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self._iteration = 0
        if self.dataset_size < 0:
            raise ValueError(f"dataset_size must be >= 0, got {self.dataset_size}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._iteration = 0

    def __iter__(self) -> Iterator[int]:
        num_replicas, rank = _resolve_rank_info(self.num_replicas, self.rank)
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        _samples_per_rank, total_size = _partition_sizes(
            self.dataset_size,
            num_replicas,
            self.drop_last,
        )
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + self.epoch * 1_000_003 + self._iteration
            )
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        self._iteration += 1

        if total_size > len(indices):
            padding_size = total_size - len(indices)
            repeats = int(math.ceil(padding_size / len(indices)))
            padding = (indices * repeats)[:padding_size]
            indices.extend(padding)
        else:
            indices = indices[:total_size]
        return iter(indices[rank:total_size:num_replicas])

    def __len__(self) -> int:
        num_replicas, _rank = _resolve_rank_info(self.num_replicas, self.rank)
        samples_per_rank, _total_size = _partition_sizes(
            self.dataset_size,
            num_replicas,
            self.drop_last,
        )
        return samples_per_rank
