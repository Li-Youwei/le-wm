"""End-to-end smoke test for the unified action prediction + world model pipeline.

Validates the full pipeline with real preprocessed LIBERO data:
  LiberoDataset → DataLoader → JEPA(ViT encoder + ARPredictor) → losses → backward

Usage:
    python smoke_test.py --data data/libero_processed/test.h5

No Hydra, PyTorch Lightning, or WandB required. Runs entirely on CPU.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from jepa import JEPA
from libero_dataset import LiberoDataset
from module import (
    ACTION_HEAD_SIZE,
    ARPredictor,
    EOS_TOKEN_ID,
    MLP,
    SIGReg,
)


def create_model():
    """Create the full model stack matching train.py / lewm.yaml config."""

    # ViT-Tiny encoder (replaces spt.backbone.utils.vit_hf('tiny', ...))
    vit_config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=224,
        patch_size=14,
    )
    encoder = ViTModel(vit_config, add_pooling_layer=False)

    hidden_dim = encoder.config.hidden_size  # 192
    embed_dim = 192

    # ARPredictor (matches cfg.predictor from lewm.yaml)
    predictor = ARPredictor(
        embed_dim=embed_dim,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        max_action_tokens=45,
        dropout=0.1,
        emb_dropout=0.0,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        pred_proj=pred_proj,
    )

    return model


def training_step(model, sigreg, batch, step_idx):
    """Replicate lejepa_forward logic from train.py."""
    alpha = 1.0   # pred_weight
    lambd = 0.09  # sigreg weight

    # 1. Encode two frames
    pixels = torch.stack([batch["pixels_current"], batch["pixels_future"]], dim=1)
    info = {"pixels": pixels}
    output = model.encode(info)
    emb = output["emb"]       # (B, 2, D)
    z_t = emb[:, 0]           # (B, D)
    z_t1 = emb[:, 1]          # (B, D)

    print(f"  emb.shape           = {emb.shape}")

    # 2. Predict
    fast_tokens = batch["fast_tokens"]
    fast_lengths = batch["fast_lengths"]
    action_logits, state_pred = model.predict(z_t, fast_tokens, fast_lengths)

    print(f"  action_logits.shape = {action_logits.shape}")
    print(f"  state_pred.shape    = {state_pred.shape}")

    # 3. Build CE targets
    B = z_t.size(0)
    num_pos = action_logits.size(1)
    targets = torch.full((B, num_pos), -100, dtype=torch.long)
    for i in range(B):
        k = fast_lengths[i].item()
        targets[i, :k] = fast_tokens[i, :k]
        targets[i, k] = EOS_TOKEN_ID

    # 4. Losses
    ce_loss = F.cross_entropy(
        action_logits.reshape(-1, ACTION_HEAD_SIZE),
        targets.reshape(-1),
        ignore_index=-100,
    )
    pred_loss = F.mse_loss(state_pred, z_t1)
    sigreg_loss = sigreg(emb.transpose(0, 1))
    total_loss = ce_loss + alpha * pred_loss + lambd * sigreg_loss

    print(f"  ce_loss     = {ce_loss.item():.4f}")
    print(f"  pred_loss   = {pred_loss.item():.4f}")
    print(f"  sigreg_loss = {sigreg_loss.item():.4f}")
    print(f"  total_loss  = {total_loss.item():.4f}")

    assert torch.isfinite(total_loss), "Loss is not finite!"
    assert total_loss.requires_grad, "Loss does not require grad!"

    # 5. Backward
    total_loss.backward()

    # Verify gradients
    grad_checks = {
        "encoder": model.encoder.embeddings.patch_embeddings.projection.weight,
        "predictor.action_head": model.predictor.action_head.weight,
        "predictor.state_query": model.predictor.state_query,
        "projector": model.projector.net[0].weight,
        "pred_proj": model.pred_proj.net[0].weight,
    }
    for name, param in grad_checks.items():
        assert param.grad is not None, f"No gradient for {name}"
    print(f"  gradients: all {len(grad_checks)} components OK")

    return total_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Smoke test for unified LeWM pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed HDF5 file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    args = parser.parse_args()

    data_path = Path(args.data)
    assert data_path.exists(), f"Data file not found: {data_path}"

    # Dataset & DataLoader
    print(f"Loading dataset from {data_path}")
    dataset = LiberoDataset(
        hdf5_dir=str(data_path.parent),
        max_action_tokens=45,
        img_size=224,
    )
    print(f"Dataset size: {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Inspect first sample
    sample = dataset[0]
    print(f"Sample 0: pixels_current={sample['pixels_current'].shape}, "
          f"fast_tokens={sample['fast_tokens'].shape}, "
          f"fast_lengths={sample['fast_lengths'].item()}")

    # Model
    print("\nCreating model...")
    model = create_model()
    sigreg = SIGReg()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ==============================
    # Training forward + backward
    # ==============================
    print("\n" + "=" * 60)
    print("TRAINING SMOKE TEST")
    print("=" * 60)
    model.train()

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print(f"\n--- Batch {i} (B={batch['pixels_current'].size(0)}) ---")
        print(f"  fast_lengths = {batch['fast_lengths'].tolist()}")

        model.zero_grad()
        training_step(model, sigreg, batch, i)

    # ==============================
    # Inference (autoregressive)
    # ==============================
    print("\n" + "=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)
    model.eval()

    with torch.no_grad():
        # Encode a single batch for inference
        batch = next(iter(loader))
        pixels = batch["pixels_current"]
        info = {"pixels": pixels.unsqueeze(1)}  # (B, 1, C, H, W) — single frame
        output = model.encode(info)
        z_t = output["emb"][:, 0]  # (B, D)

        tokens, lengths = model.predict_actions(z_t, max_len=45, temperature=0.0)
        print(f"\n  z_t.shape     = {z_t.shape}")
        print(f"  tokens.shape  = {tokens.shape}")
        print(f"  lengths       = {lengths.tolist()}")
        print(f"  tokens[0,:10] = {tokens[0, :10].tolist()}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
