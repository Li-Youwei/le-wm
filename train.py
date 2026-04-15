from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from transformers import T5EncoderModel

from jepa import JEPA
from module import ARPredictor, MLP, ACTION_HEAD_SIZE, EOS_TOKEN_ID
from utils import ModelObjectCallBack, PeriodicPrintCallback


def lejepa_forward(self, batch, stage, cfg):
    """VLA baseline training step: L_CE only."""

    # 1. Unpack batch
    pixels_agent = batch["pixels_agent"]          # (B, 3, H, W)
    pixels_hand = batch["pixels_hand"]            # (B, 3, H, W)
    proprio = batch["proprio"]                    # (B, 8)
    fast_tokens = batch["fast_tokens"]            # (B, max_action_tokens)
    fast_lengths = batch["fast_lengths"]          # (B,)

    # Language batch fields may be absent in no-language ablation runs
    lang_ids = batch.get("lang_input_ids", None)   # (B, max_lang_tokens) or None
    lang_mask = batch.get("lang_attention_mask", None)  # (B, max_lang_tokens) or None

    # 2. Encode visual + language (language skipped when lang_ids is None)
    z_agent, z_hand, lang_embeds, lang_lengths = self.model.encode(
        pixels_agent, pixels_hand, lang_ids, lang_mask,
    )

    # 3. Predict action logits (teacher forcing)
    action_logits = self.model.predict(
        z_agent, z_hand, proprio, lang_embeds, lang_lengths,
        fast_tokens, fast_lengths,
    )

    # 4. Build CE targets (shifted by 1: position j predicts token j+1)
    B = z_agent.size(0)
    num_action_positions = action_logits.size(1)  # 1 (BOS) + max_action_tokens
    targets = torch.full((B, num_action_positions), -100, dtype=torch.long, device=z_agent.device)
    for i in range(B):
        k = fast_lengths[i].item()
        targets[i, :k] = fast_tokens[i, :k]    # T_1, T_2, ..., T_k
        targets[i, k] = EOS_TOKEN_ID           # EOS after last real token

    # 5. L_CE only
    output = {}
    output["ce_loss"] = F.cross_entropy(
        action_logits.reshape(-1, ACTION_HEAD_SIZE),
        targets.reshape(-1),
        ignore_index=-100,
    )
    output["loss"] = output["ce_loss"]

    # 6. Token accuracy (diagnostic)
    with torch.no_grad():
        preds = action_logits.argmax(dim=-1)
        valid = targets != -100
        correct = (preds == targets) & valid
        n_valid = valid.sum().float()
        output["token_accuracy"] = correct.sum().float() / n_valid.clamp(min=1)

    # 7. Logging
    log_dict = {
        f"{stage}/ce_loss": output["ce_loss"].detach(),
        f"{stage}/total_loss": output["loss"].detach(),
        f"{stage}/token_accuracy": output["token_accuracy"],
    }
    # Log learning rate if available
    if hasattr(self, "trainer") and self.trainer is not None:
        opts = self.trainer.optimizers
        if opts:
            lr = opts[0].param_groups[0]["lr"]
            log_dict[f"{stage}/lr"] = lr
    self.log_dict(log_dict, on_step=True, on_epoch=True, sync_dist=True)

    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    # Single source of truth for max_action_tokens and max_lang_tokens
    max_action_tokens = cfg.data.dataset.get("max_action_tokens", 80)
    max_lang_tokens = cfg.data.dataset.get("max_lang_tokens", 25)
    proprio_dim = cfg.data.dataset.get("proprio_dim", 9)
    use_language = cfg.data.dataset.get("use_language", True)

    from libero_dataset import LiberoDataset

    dataset = LiberoDataset(
        hdf5_dir=cfg.data.dataset.hdf5_dir,
        max_action_tokens=max_action_tokens,
        max_lang_tokens=max_lang_tokens,
        img_size=cfg.data.dataset.get("img_size", cfg.img_size),
        use_language=use_language,
    )

    # Demo-level split: avoid leaking chunks from the same demo into both train and val.
    # Read demo_idx per file (not per sample) for efficiency.
    import h5py as _h5py
    demo_ids = []
    prev_file = None
    _demo_arr = None
    for fpath, local_idx in dataset._index:
        if fpath != prev_file:
            with _h5py.File(fpath, "r") as _f:
                _demo_arr = _f["demo_idx"][()]
            prev_file = fpath
        demo_ids.append(int(_demo_arr[local_idx]))

    overfit_demo = cfg.get("overfit_demo", None)
    if overfit_demo is not None:
        # Pipeline sanity check: take one demo and use the same samples for
        # both train and val. If the model can't drive loss to ~0 on this,
        # there is a bug in the pipeline (data, forward, or loss).
        target = int(overfit_demo)
        overfit_indices = [i for i, d in enumerate(demo_ids) if d == target]
        if not overfit_indices:
            available = sorted(set(demo_ids))
            raise ValueError(
                f"overfit_demo={target} produced 0 samples. "
                f"Available demo_ids in this dataset: {available[:20]}"
            )
        print(
            f"[Overfit mode] demo_idx={target}: {len(overfit_indices)} chunks "
            f"(train == val, demo-level split disabled)."
        )
        train_indices = overfit_indices
        val_indices = list(overfit_indices)
    else:
        unique_demos = sorted(set(demo_ids))
        n_train_demos = int(len(unique_demos) * cfg.train_split)
        perm = torch.randperm(len(unique_demos), generator=rnd_gen).tolist()
        train_demo_set = set(unique_demos[perm[i]] for i in range(n_train_demos))

        train_indices = [i for i, d in enumerate(demo_ids) if d in train_demo_set]
        val_indices = [i for i, d in enumerate(demo_ids) if d not in train_demo_set]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    # In overfit mode, keep every sample each epoch — drop_last=True could
    # discard the only batch when the sample count is smaller than batch_size.
    train_drop_last = overfit_demo is None
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=train_drop_last, generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    # ARPredictor with language + proprio support.
    # max_lang_tokens is still passed in so pos_embedding has a large-enough max_seq_len —
    # when use_language=False, language positions are simply never populated at runtime.
    predictor = ARPredictor(
        embed_dim=embed_dim,
        max_action_tokens=max_action_tokens,
        max_lang_tokens=max_lang_tokens,
        proprio_dim=proprio_dim,
        **cfg.predictor,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    if use_language:
        # T5-small encoder (frozen)
        lang_encoder = T5EncoderModel.from_pretrained("t5-small")
        lang_encoder.eval()
        for p in lang_encoder.parameters():
            p.requires_grad_(False)

        # Language projection: T5 d_model (512) → embed_dim
        lang_proj = torch.nn.Linear(lang_encoder.config.d_model, embed_dim)
    else:
        print("[Ablation] use_language=False — skipping T5 encoder and language projection.")
        lang_encoder = None
        lang_proj = None

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    # TensorBoard logger
    logger = TensorBoardLogger(str(run_dir / "tb_logs"), name="vla_baseline")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    callbacks = [object_dump_callback]
    if overfit_demo is not None:
        # Terminal progress line every N epochs — easier to eyeball than
        # scrolling Lightning progress bars over 2k epochs.
        print_every = int(cfg.get("overfit_print_every", 100))
        callbacks.append(PeriodicPrintCallback(every_n_epochs=print_every))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
