from __future__ import annotations

import torch


def _equal_horizon_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with each future horizon weighted equally."""
    if pred.shape != target.shape:
        raise RuntimeError(
            "Prediction/target shapes must match exactly, got "
            f"pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
        )
    if pred.dim() < 3:
        raise RuntimeError(
            "Expected tensors with shape (B,K,...), got "
            f"pred={tuple(pred.shape)}"
        )
    per_horizon_dims = tuple(range(2, pred.dim()))
    return (pred - target).square().mean(dim=per_horizon_dims).mean()


def _weighted_stream_loss(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Combine stream losses with weights normalized to sum to one."""
    missing = sorted(set(losses) - set(weights))
    if missing:
        raise ValueError(f"Missing stream loss weights for: {missing}")
    total_weight = sum(float(weights[name]) for name in losses)
    if total_weight <= 0:
        raise ValueError("Prediction stream weights must sum to a positive value")
    out = None
    for name, loss in losses.items():
        scaled = loss * (float(weights[name]) / total_weight)
        out = scaled if out is None else out + scaled
    assert out is not None
    return out


def _future_cls_by_horizon(z_future: torch.Tensor) -> torch.Tensor:
    """Return future CLS latents as (B,K,D)."""
    if z_future.dim() == 4:
        return z_future[:, :, 0]
    if z_future.dim() == 3:
        return z_future
    if z_future.dim() == 2:
        return z_future.unsqueeze(1)
    raise RuntimeError(
        "Future visual latents must be (B,K,N,D), (B,K,D), or legacy (B,D), "
        f"got {tuple(z_future.shape)}"
    )


def _sigreg_loss_over_future_horizons(
    sigreg: torch.nn.Module,
    z_ag_cls: torch.Tensor,
    z_hd_cls: torch.Tensor,
    z_agent_future: torch.Tensor,
    z_hand_future: torch.Tensor,
) -> torch.Tensor:
    """Apply SIGReg to each future horizon and average horizons equally."""
    z_ag_future_cls = _future_cls_by_horizon(z_agent_future)
    z_hd_future_cls = _future_cls_by_horizon(z_hand_future)
    if z_ag_future_cls.shape != z_hd_future_cls.shape:
        raise RuntimeError(
            "Future agent/hand CLS shapes must match for SIGReg, got "
            f"{tuple(z_ag_future_cls.shape)} vs {tuple(z_hd_future_cls.shape)}"
        )
    if z_ag_future_cls.size(0) != z_ag_cls.size(0):
        raise RuntimeError(
            "Future/current batch sizes must match for SIGReg, got "
            f"future={z_ag_future_cls.size(0)} current={z_ag_cls.size(0)}"
        )
    losses = []
    for horizon_idx in range(z_ag_future_cls.size(1)):
        sigreg_input = torch.stack(
            [
                z_ag_cls,
                z_hd_cls,
                z_ag_future_cls[:, horizon_idx],
                z_hd_future_cls[:, horizon_idx],
            ],
            dim=0,
        )
        losses.append(sigreg(sigreg_input))
    return torch.stack(losses).mean()
