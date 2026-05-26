from __future__ import annotations

import unittest

import torch

from loss_utils import (
    _equal_horizon_mse,
    _sigreg_loss_over_future_horizons,
    _weighted_stream_loss,
)


class LossHorizonWeightingTest(unittest.TestCase):
    def test_equal_horizon_mse_averages_per_horizon_losses(self) -> None:
        pred = torch.zeros(1, 2, 3)
        target = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]])

        loss = _equal_horizon_mse(pred, target)

        self.assertTrue(torch.allclose(loss, torch.tensor(5.0)))

    def test_weighted_stream_loss_normalizes_weights(self) -> None:
        loss = _weighted_stream_loss(
            {
                "ag": torch.tensor(1.0),
                "hd": torch.tensor(2.0),
                "pr": torch.tensor(10.0),
            },
            {"ag": 2.0, "hd": 1.0, "pr": 1.0},
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(3.5)))

    def test_weighted_stream_loss_rejects_non_positive_total_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            _weighted_stream_loss(
                {
                    "ag": torch.tensor(1.0),
                    "hd": torch.tensor(2.0),
                    "pr": torch.tensor(3.0),
                },
                {"ag": 0.0, "hd": 0.0, "pr": 0.0},
            )

    def test_sigreg_runs_once_per_future_horizon_and_averages(self) -> None:
        class RecordingSigReg(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.calls.append(x.detach().clone())
                return x[3].mean()

        sigreg = RecordingSigReg()
        z_ag = torch.zeros(2, 1)
        z_hd = torch.ones(2, 1)
        z_ag_future = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]]).expand(
            2, -1, -1
        )
        z_hd_future = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]]).expand(
            2, -1, -1
        )

        loss = _sigreg_loss_over_future_horizons(
            sigreg,
            z_ag,
            z_hd,
            z_ag_future,
            z_hd_future,
        )

        self.assertEqual(len(sigreg.calls), 4)
        self.assertTrue(torch.allclose(loss, torch.tensor(4.0)))
        for k, call in enumerate(sigreg.calls):
            self.assertEqual(tuple(call.shape), (4, 2, 1))
            self.assertTrue(torch.equal(call[0], z_ag))
            self.assertTrue(torch.equal(call[1], z_hd))
            self.assertTrue(torch.equal(call[2], z_ag_future[:, k]))
            self.assertTrue(torch.equal(call[3], z_hd_future[:, k]))

    def test_sigreg_uses_cls_when_future_targets_include_patch_tokens(self) -> None:
        class FirstFutureValueSigReg(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[2].mean()

        z_ag = torch.zeros(1, 1)
        z_hd = torch.zeros(1, 1)
        # (B, K, N, D); token 0 is CLS, token 1 is a patch token that should
        # not enter SIGReg.
        z_ag_future = torch.tensor([[[[2.0], [200.0]], [[4.0], [400.0]]]])
        z_hd_future = torch.tensor([[[[3.0], [300.0]], [[5.0], [500.0]]]])

        loss = _sigreg_loss_over_future_horizons(
            FirstFutureValueSigReg(),
            z_ag,
            z_hd,
            z_ag_future,
            z_hd_future,
        )

        self.assertTrue(torch.allclose(loss, torch.tensor(3.0)))


if __name__ == "__main__":
    unittest.main()
