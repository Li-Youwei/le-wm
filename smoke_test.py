"""End-to-end smoke test for the VLA baseline (v3).

Validates the full pipeline with real preprocessed LIBERO data on GPU:
  LiberoDataset → DataLoader → JEPA(frozen DINOv2 + T5-small + ARPredictor) → L_CE → backward

Tests:
  1. Model instantiation (frozen DINOv2 shared encoder, frozen T5, proprio MLP)
  2. Data loading (dual-view, proprio 9d, language tokens, FAST tokens)
  3. Forward pass shape correctness
  4. CE loss finite and nonzero
  5. Backward pass gradient flow (all trainable params have grads)
  6. T5 params frozen (no grads)
  7. Inference predict_actions() generates token sequences
  8. VRAM profiling across batch sizes (128 → 64 → 32)

Usage:
    python smoke_test.py --data /data/lyw/libero_processed/libero_spatial/some_task.h5
    python smoke_test.py --data /data/lyw/libero_processed/libero_spatial/  # whole dir
"""

from __future__ import annotations

import argparse
import atexit
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

from jepa import JEPA
from libero_dataset import LiberoDataset
from module import (
    ACTION_HEAD_SIZE,
    ARPredictor,
    EOS_TOKEN_ID,
    MLP,
)

# ---------------------------------------------------------------------------
# Config — must match CLAUDE.md / train.py / config/train/data/libero.yaml
# ---------------------------------------------------------------------------
EMBED_DIM = 192
MAX_ACTION_TOKENS = 80
MAX_LANG_TOKENS = 25
PROPRIO_DIM = 9  # ee_pos(3) + xyzw_quat(4) + gripper_raw(2)


def create_model(device: torch.device) -> JEPA:
    """Create the full VLA model stack matching train.py's default config.

    Default visual backbone = frozen DINOv2-base (via build_visual_encoder),
    matching config/train/lewm.yaml; the projector maps the encoder hidden_dim
    (768) → EMBED_DIM (192). local_files_only is omitted so it honors
    HF_HUB_OFFLINE. Pass source=spt_vit in enc_cfg to exercise the legacy
    trainable ViT-Tiny path instead.
    """
    import stable_pretraining as spt
    from omegaconf import OmegaConf

    from vision_backbone import build_visual_encoder

    enc_cfg = OmegaConf.create(
        {
            "encoder_scale": "tiny",
            "patch_size": 14,
            "img_size": 224,
            "vision_encoder": {
                "source": "hf",
                "model_name_or_path": "facebook/dinov2-base",
                "freeze": True,
                "trust_remote_code": False,
            },
        }
    )
    encoder, hidden_dim, freeze_encoder = build_visual_encoder(enc_cfg, spt)

    predictor = ARPredictor(
        embed_dim=EMBED_DIM,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        max_action_tokens=MAX_ACTION_TOKENS,
        max_lang_tokens=MAX_LANG_TOKENS,
        proprio_dim=PROPRIO_DIM,
        dropout=0.1,
        emb_dropout=0.0,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=EMBED_DIM,
        hidden_dim=2048,
        norm_fn=nn.LayerNorm,
    )

    # T5-small (frozen)
    lang_encoder = T5EncoderModel.from_pretrained("t5-small")
    lang_encoder.eval()
    for p in lang_encoder.parameters():
        p.requires_grad_(False)

    lang_proj = nn.Linear(lang_encoder.config.d_model, EMBED_DIM)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
        freeze_encoder=freeze_encoder,
    )
    return model.to(device)


def print_vram(tag: str) -> None:
    """Print current GPU VRAM usage."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        f"  [VRAM {tag}] allocated={alloc:.2f}GB, reserved={reserved:.2f}GB, total={total:.2f}GB"
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move all tensors in batch to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def training_step(model: JEPA, batch: dict, device: torch.device) -> float:
    """Replicate lejepa_forward logic from train.py — L_CE only."""

    batch = move_batch(batch, device)
    pixels_agent = batch["pixels_agent"]
    pixels_hand = batch["pixels_hand"]
    proprio = batch["proprio"]
    lang_ids = batch["lang_input_ids"]
    lang_mask = batch["lang_attention_mask"]
    fast_tokens = batch["fast_tokens"]
    fast_lengths = batch["fast_lengths"]

    B = pixels_agent.size(0)

    # 1. Encode (dual-view shared DINOv2 + frozen T5)
    z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
        pixels_agent,
        pixels_hand,
        lang_ids,
        lang_mask,
    )

    # encode() returns (B, N, D) per view — N=1 for CLS-only (default).
    assert z_agent.shape[0] == B and z_agent.shape[-1] == EMBED_DIM, (
        f"z_agent shape {tuple(z_agent.shape)}"
    )
    assert z_hand.shape[0] == B and z_hand.shape[-1] == EMBED_DIM, (
        f"z_hand shape {tuple(z_hand.shape)}"
    )
    assert lang_embeds.shape == (B, MAX_LANG_TOKENS, EMBED_DIM), (
        f"lang_embeds shape {lang_embeds.shape}"
    )
    print(
        f"  encode OK: z_agent={tuple(z_agent.shape)}, lang_embeds={tuple(lang_embeds.shape)}, lang_lengths={lang_lengths.tolist()}"
    )

    # 2. Predict (teacher forcing with hybrid attention mask)
    action_logits = model.predict(
        z_agent,
        z_hand,
        proprio,
        lang_embeds,
        lang_lengths,
        fast_tokens,
        fast_lengths,
    )

    expected_logits_shape = (B, 1 + MAX_ACTION_TOKENS, ACTION_HEAD_SIZE)
    assert action_logits.shape == expected_logits_shape, (
        f"action_logits shape {action_logits.shape}, expected {expected_logits_shape}"
    )
    print(f"  predict OK: action_logits={action_logits.shape}")

    # 3. Build CE targets
    num_pos = action_logits.size(1)
    targets = torch.full((B, num_pos), -100, dtype=torch.long, device=device)
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

    assert torch.isfinite(ce_loss), f"Loss is not finite: {ce_loss.item()}"
    assert ce_loss.item() > 0, f"Loss is zero: {ce_loss.item()}"
    assert ce_loss.requires_grad, "Loss does not require grad"
    print(f"  L_CE = {ce_loss.item():.4f}")

    # 5. Backward
    ce_loss.backward()
    print_vram("after backward")

    # 6. Verify gradients — all trainable components (the DINOv2 encoder is
    # frozen, so it is intentionally NOT here; see the frozen check below).
    grad_checks = {
        "predictor.action_head": model.predictor.action_head.weight,
        "predictor.proprio_encoder": model.predictor.proprio_encoder.net[0].weight,
        "predictor.type_embedding": model.predictor.type_embedding.weight,
        "predictor.pos_embedding": model.predictor.pos_embedding,
        "projector": model.projector.net[0].weight,
        "lang_proj": model.lang_proj.weight,
    }
    for name, param in grad_checks.items():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    print(f"  gradients: all {len(grad_checks)} trained components OK")

    # 7. Verify T5 + the frozen DINOv2 encoder have no gradients
    for name, param in model.lang_encoder.named_parameters():
        assert param.grad is None, f"T5 param {name} has gradient (should be frozen)"
    for name, param in model.encoder.named_parameters():
        assert param.grad is None, (
            f"encoder param {name} has gradient (should be frozen)"
        )
    print("  T5 + DINOv2 encoder frozen: confirmed (no gradients)")

    return ce_loss.item()


def inference_test(model: JEPA, batch: dict, device: torch.device) -> None:
    """Test autoregressive generation."""

    batch = move_batch(batch, device)
    B = batch["pixels_agent"].size(0)

    z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
        batch["pixels_agent"],
        batch["pixels_hand"],
        batch["lang_input_ids"],
        batch["lang_attention_mask"],
    )

    tokens, lengths = model.predict_actions(
        z_agent,
        z_hand,
        batch["proprio"],
        lang_embeds,
        lang_lengths,
        max_len=MAX_ACTION_TOKENS,
        temperature=0.0,
    )

    assert tokens.dim() == 2, f"tokens should be 2D, got {tokens.dim()}D"
    assert lengths.shape == (B,), f"lengths shape {lengths.shape}"
    # All generated tokens should be in FAST vocab range [0, 1023]
    real_mask = torch.arange(tokens.size(1), device=device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if real_mask.any():
        real_tokens = tokens[real_mask]
        assert (real_tokens >= 0).all() and (real_tokens < 1024).all(), (
            f"Generated tokens outside FAST vocab range: min={real_tokens.min()}, max={real_tokens.max()}"
        )

    print(f"  tokens shape = {tokens.shape}")
    print(f"  lengths      = {lengths.tolist()}")
    if tokens.size(1) > 0:
        print(f"  tokens[0,:10]= {tokens[0, : min(10, tokens.size(1))].tolist()}")
    print_vram("after inference")


def vram_profile(model: JEPA, dataset: LiberoDataset, device: torch.device) -> None:
    """Test batch sizes to find max for 24GB 4090."""

    print("\n" + "=" * 60)
    print("VRAM PROFILING")
    print("=" * 60)

    for bs in [128, 64, 32, 16]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )
        batch = move_batch(next(iter(loader)), device)

        try:
            model.train()
            model.lang_encoder.eval()
            model.zero_grad()

            z_agent, z_hand, lang_embeds, lang_lengths = model.encode(
                batch["pixels_agent"],
                batch["pixels_hand"],
                batch["lang_input_ids"],
                batch["lang_attention_mask"],
            )
            action_logits = model.predict(
                z_agent,
                z_hand,
                batch["proprio"],
                lang_embeds,
                lang_lengths,
                batch["fast_tokens"],
                batch["fast_lengths"],
            )

            B = z_agent.size(0)
            num_pos = action_logits.size(1)
            targets = torch.full((B, num_pos), -100, dtype=torch.long, device=device)
            for i in range(B):
                k = batch["fast_lengths"][i].item()
                targets[i, :k] = batch["fast_tokens"][i, :k]
                targets[i, k] = EOS_TOKEN_ID

            loss = F.cross_entropy(
                action_logits.reshape(-1, ACTION_HEAD_SIZE),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()

            peak = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            status = "OK" if peak < total * 0.95 else "TIGHT"
            print(
                f"  batch_size={bs:>4d}: peak VRAM = {peak:.2f}GB / {total:.2f}GB  [{status}]"
            )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch_size={bs:>4d}: OOM")

        model.zero_grad()
        del batch, loader
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for VLA baseline (v3)")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed HDF5 file or directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for train/inference tests (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-vram", action="store_true", help="Skip VRAM profiling")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({total_gb:.1f}GB)")

    data_path = Path(args.data)
    assert data_path.exists(), f"Data not found: {data_path}"

    # If user passed a file, create a temp symlink dir
    if data_path.is_file():
        tmp_dir = tempfile.mkdtemp(prefix="smoke_test_")
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
        (Path(tmp_dir) / data_path.name).symlink_to(data_path.resolve())
        hdf5_dir = tmp_dir
        print(f"Loading single file: {data_path.name}")
    else:
        hdf5_dir = str(data_path)
        print(f"Loading directory: {data_path}")

    # ==============================
    # Data loading
    # ==============================
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    dataset = LiberoDataset(
        hdf5_dir=hdf5_dir,
        max_action_tokens=MAX_ACTION_TOKENS,
        max_lang_tokens=MAX_LANG_TOKENS,
        img_size=224,
    )
    print(f"Dataset size: {len(dataset)} samples")

    sample = dataset[0]
    checks = {
        "pixels_agent": (3, 224, 224),
        "pixels_hand": (3, 224, 224),
        "proprio": (PROPRIO_DIM,),
        "lang_input_ids": (MAX_LANG_TOKENS,),
        "lang_attention_mask": (MAX_LANG_TOKENS,),
        "fast_tokens": (MAX_ACTION_TOKENS,),
    }
    for key, expected_shape in checks.items():
        actual = sample[key].shape
        assert actual == expected_shape, (
            f"{key}: expected {expected_shape}, got {actual}"
        )
        print(f"  {key}: {actual} OK")
    print(f"  fast_lengths: {sample['fast_lengths'].item()}")
    print(f"  proprio values: {sample['proprio'].tolist()}")

    # ==============================
    # Model creation
    # ==============================
    print("\n" + "=" * 60)
    print("MODEL CREATION")
    print("=" * 60)

    model = create_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters:     {total_params:>12,}")
    print(f"Trainable parameters: {trainable_params:>12,}")
    print(f"Frozen T5 parameters: {frozen_params:>12,}")
    print_vram("after model creation")

    # ==============================
    # Training smoke test
    # ==============================
    print("\n" + "=" * 60)
    print("TRAINING SMOKE TEST")
    print("=" * 60)

    model.train()
    model.lang_encoder.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    for i, batch in enumerate(loader):
        if i >= 2:
            break
        B = batch["pixels_agent"].size(0)
        print(f"\n--- Batch {i} (B={B}) ---")
        print(
            f"  fast_lengths = {batch['fast_lengths'].tolist()[:8]}{'...' if B > 8 else ''}"
        )

        model.zero_grad()
        training_step(model, batch, device)

    # ==============================
    # Inference smoke test
    # ==============================
    print("\n" + "=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inference_test(model, batch, device)

    # ==============================
    # VRAM profiling (GPU only)
    # ==============================
    if device.type == "cuda" and not args.skip_vram:
        model.zero_grad()
        torch.cuda.empty_cache()
        vram_profile(model, dataset, device)

    # ==============================
    # Summary
    # ==============================
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    print("  Model: frozen DINOv2-base(shared) + T5-small(frozen) + ARPredictor")
    print(f"  Params: {trainable_params:,} trainable + {frozen_params:,} frozen")
    print(
        f"  Sequence: [lang({MAX_LANG_TOKENS}) + z_ag + z_hd + z_pr + BOS + tokens({MAX_ACTION_TOKENS})]"
    )
    print("  Loss: L_CE only")
    print(f"  H=20, max_action_tokens={MAX_ACTION_TOKENS}, proprio_dim={PROPRIO_DIM}")
    if device.type == "cuda":
        print_vram("final")


if __name__ == "__main__":
    main()
