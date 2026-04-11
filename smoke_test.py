"""End-to-end smoke test for the VLA baseline pipeline.

Validates the full pipeline with real preprocessed LIBERO data:
  LiberoDataset → DataLoader → JEPA(ViT + T5 + ARPredictor) → L_CE → backward

Usage:
    python smoke_test.py --data data/libero_processed/test.h5

No Hydra, PyTorch Lightning, or WandB required. Runs entirely on CPU.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, ViTConfig, ViTModel

from jepa import JEPA
from libero_dataset import LiberoDataset
from module import (
    ACTION_HEAD_SIZE,
    ARPredictor,
    EOS_TOKEN_ID,
    MLP,
)


def create_model():
    """Create the full VLA model stack matching train.py config."""

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

    # ARPredictor with language + proprio support
    predictor = ARPredictor(
        embed_dim=embed_dim,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        max_action_tokens=100,
        max_lang_tokens=25,
        proprio_dim=8,
        dropout=0.1,
        emb_dropout=0.0,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    # T5-small (frozen)
    lang_encoder = T5EncoderModel.from_pretrained("t5-small")
    lang_encoder.eval()
    for p in lang_encoder.parameters():
        p.requires_grad_(False)

    lang_proj = nn.Linear(lang_encoder.config.d_model, embed_dim)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
    )

    return model


def training_step(model, batch, step_idx):
    """Replicate lejepa_forward logic from train.py — L_CE only."""

    pixels_agent = batch["pixels_agent"]
    pixels_hand = batch["pixels_hand"]
    proprio = batch["proprio"]
    lang_ids = batch["lang_input_ids"]
    lang_mask = batch["lang_attention_mask"]
    fast_tokens = batch["fast_tokens"]
    fast_lengths = batch["fast_lengths"]

    # 1. Encode
    z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
        pixels_agent, pixels_hand, lang_ids, lang_mask,
    )
    print(f"  z_agent.shape    = {z_agent.shape}")
    print(f"  z_hand.shape     = {z_hand.shape}")
    print(f"  lang_embeds.shape = {lang_embeds.shape}")
    print(f"  lang_lengths     = {lang_lengths.tolist()}")

    # 2. Predict
    action_logits = model.predict(
        z_agent, z_hand, proprio, lang_embeds, lang_lengths,
        fast_tokens, fast_lengths,
    )
    print(f"  action_logits.shape = {action_logits.shape}")

    # 3. Build CE targets
    B = z_agent.size(0)
    num_pos = action_logits.size(1)
    targets = torch.full((B, num_pos), -100, dtype=torch.long)
    for i in range(B):
        k = fast_lengths[i].item()
        targets[i, :k] = fast_tokens[i, :k]
        targets[i, k] = EOS_TOKEN_ID

    # 4. L_CE only
    ce_loss = F.cross_entropy(
        action_logits.reshape(-1, ACTION_HEAD_SIZE),
        targets.reshape(-1),
        ignore_index=-100,
    )
    total_loss = ce_loss

    print(f"  ce_loss     = {ce_loss.item():.4f}")
    print(f"  total_loss  = {total_loss.item():.4f}")

    assert torch.isfinite(total_loss), "Loss is not finite!"
    assert total_loss.requires_grad, "Loss does not require grad!"

    # 5. Backward
    total_loss.backward()

    # Verify gradients — trained components should have grads
    grad_checks = {
        "encoder.patch_embed": model.encoder.embeddings.patch_embeddings.projection.weight,
        "predictor.action_head": model.predictor.action_head.weight,
        "predictor.proprio_encoder": model.predictor.proprio_encoder.net[0].weight,
        "projector": model.projector.net[0].weight,
        "lang_proj": model.lang_proj.weight,
    }
    for name, param in grad_checks.items():
        assert param.grad is not None, f"No gradient for {name}"
    print(f"  gradients: all {len(grad_checks)} trained components OK")

    # Verify T5 is frozen — should NOT have gradients
    for name, param in model.lang_encoder.named_parameters():
        assert param.grad is None, f"T5 param {name} has gradient (should be frozen)!"
    print(f"  T5 encoder: confirmed frozen (no gradients)")

    return total_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Smoke test for VLA baseline pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed HDF5 file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    args = parser.parse_args()

    data_path = Path(args.data)
    assert data_path.exists(), f"Data file not found: {data_path}"

    # If user passed a file, create a temp symlink dir so LiberoDataset loads only that file.
    # LiberoDataset expects a directory and loads ALL .h5/.hdf5 inside it.
    if data_path.is_file():
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="smoke_test_")
        link_path = Path(tmp_dir) / data_path.name
        link_path.symlink_to(data_path.resolve())
        hdf5_dir = tmp_dir
        print(f"Loading single file: {data_path} (via temp dir)")
    else:
        hdf5_dir = str(data_path)
        print(f"Loading all files from directory: {data_path}")

    dataset = LiberoDataset(
        hdf5_dir=hdf5_dir,
        max_action_tokens=100,
        max_lang_tokens=25,
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
    print(f"Sample 0: pixels_agent={sample['pixels_agent'].shape}, "
          f"pixels_hand={sample['pixels_hand'].shape}, "
          f"proprio={sample['proprio'].shape}, "
          f"lang_input_ids={sample['lang_input_ids'].shape}, "
          f"fast_tokens={sample['fast_tokens'].shape}, "
          f"fast_lengths={sample['fast_lengths'].item()}")

    # Model
    print("\nCreating model...")
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen T5 parameters: {total_params - trainable_params:,}")

    # ==============================
    # Training forward + backward
    # ==============================
    print("\n" + "=" * 60)
    print("TRAINING SMOKE TEST")
    print("=" * 60)
    model.train()
    # Keep T5 in eval mode even during training
    model.lang_encoder.eval()

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print(f"\n--- Batch {i} (B={batch['pixels_agent'].size(0)}) ---")
        print(f"  fast_lengths = {batch['fast_lengths'].tolist()}")

        model.zero_grad()
        training_step(model, batch, i)

    # ==============================
    # Inference (autoregressive)
    # ==============================
    print("\n" + "=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader))

        # Encode
        z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
            batch["pixels_agent"], batch["pixels_hand"],
            batch["lang_input_ids"], batch["lang_attention_mask"],
        )

        # Generate actions
        tokens, lengths = model.predict_actions(
            z_agent, z_hand, batch["proprio"], lang_embeds, lang_lengths,
            max_len=45, temperature=0.0,
        )
        print(f"\n  z_agent.shape = {z_agent.shape}")
        print(f"  tokens.shape  = {tokens.shape}")
        print(f"  lengths       = {lengths.tolist()}")
        if tokens.size(1) > 0:
            print(f"  tokens[0,:10] = {tokens[0, :min(10, tokens.size(1))].tolist()}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
