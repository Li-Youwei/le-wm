"""Tests for multi-token visual projection contracts."""

from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from jepa import JEPA


class FakeEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 4, n_patches: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = True):
        batch = x.size(0)
        n_tokens = 1 + self.n_patches
        hidden = torch.arange(
            batch * n_tokens * self.hidden_dim,
            dtype=x.dtype,
            device=x.device,
        ).reshape(batch, n_tokens, self.hidden_dim)
        return types.SimpleNamespace(last_hidden_state=hidden)


class RecordingProjector(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.calls: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(x.shape))
        return x + self.offset


class VisualTokenProjectionTest(unittest.TestCase):
    def test_cls_and_patch_projectors_are_not_mixed(self) -> None:
        cls_projector = RecordingProjector(offset=100.0)
        patch_projector = RecordingProjector(offset=1000.0)
        model = JEPA(
            encoder=FakeEncoder(hidden_dim=4, n_patches=16),
            predictor=nn.Identity(),
            projector=cls_projector,
            patch_projector=patch_projector,
            visual_pool_grid=2,
        )

        pixels_agent = torch.zeros(2, 3, 224, 224)
        pixels_hand = torch.zeros(2, 3, 224, 224)
        z_agent, z_hand, _, _ = model.encode(pixels_agent, pixels_hand)

        self.assertEqual(tuple(z_agent.shape), (2, 5, 4))
        self.assertEqual(tuple(z_hand.shape), (2, 5, 4))
        self.assertEqual(cls_projector.calls, [(4, 4)])
        self.assertEqual(patch_projector.calls, [(16, 4)])
        self.assertTrue(torch.all(z_agent[:, 0] < z_agent[:, 1:].amin(dim=1)))

        future_agent, future_hand = model.encode_future_visual(
            pixels_agent,
            pixels_hand,
        )
        self.assertEqual(tuple(future_agent.shape), (2, 4))
        self.assertEqual(tuple(future_hand.shape), (2, 4))
        self.assertEqual(cls_projector.calls, [(4, 4), (4, 4)])
        self.assertEqual(patch_projector.calls, [(16, 4)])


if __name__ == "__main__":
    unittest.main()
