import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, MLP, SIGReg, ACTION_HEAD_SIZE
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """MODIFIED: unified action prediction + world model training step."""

    lambd = cfg.loss.sigreg.weight                 # 0.09
    alpha = cfg.loss.get("pred_weight", 1.0)       # NEW: L_pred weight, default 1.0

    # 1. Encode two frames: current (o_t) and future (o_{t+1}, one chunk later)
    #    Stack into (B, 2, C, H, W) for efficient batched encoding
    pixels = torch.stack([batch["pixels_current"], batch["pixels_future"]], dim=1)
    info = {"pixels": pixels}
    output = self.model.encode(info)
    emb = output["emb"]          # (B, 2, D)
    z_t = emb[:, 0]              # (B, D) current frame
    z_t1 = emb[:, 1]             # (B, D) future frame (H=10 raw steps later)

    # 2. Run unified predictor
    fast_tokens = batch["fast_tokens"]      # (B, max_action_tokens), padded with PAD=1026
    fast_lengths = batch["fast_lengths"]    # (B,)
    action_logits, state_pred = self.model.predict(z_t, fast_tokens, fast_lengths)

    # 3. Build CE targets (shifted by 1: position j predicts token j+1)
    B = z_t.size(0)
    num_action_positions = action_logits.size(1)  # 1 + max_action_tokens = 36
    targets = torch.full((B, num_action_positions), -100, dtype=torch.long, device=z_t.device)
    for i in range(B):
        k = fast_lengths[i].item()
        targets[i, :k] = fast_tokens[i, :k]    # T_1, T_2, ..., T_k
        targets[i, k] = 1025                    # EOS after last real token

    # 4. Losses
    output["ce_loss"] = F.cross_entropy(
        action_logits.reshape(-1, ACTION_HEAD_SIZE),
        targets.reshape(-1),
        ignore_index=-100,
    )
    output["pred_loss"] = F.mse_loss(state_pred, z_t1)  # NO detach — end-to-end through encoder
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))  # (2, B, D)
    output["loss"] = output["ce_loss"] + alpha * output["pred_loss"] + lambd * output["sigreg_loss"]

    # 5. Logging
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    is_libero = cfg.data.dataset.get("name", "") == "libero"

    # Single source of truth for max_action_tokens — used by both dataset and model.
    # Avoids silent shape mismatch if only one side is updated.
    max_action_tokens = cfg.data.dataset.get("max_action_tokens", 40)

    if is_libero:
        # MODIFIED: LIBERO uses standalone dataset with pre-computed FAST tokens
        from libero_dataset import LiberoDataset

        dataset = LiberoDataset(
            hdf5_dir=cfg.data.dataset.hdf5_dir,
            max_action_tokens=max_action_tokens,
            img_size=cfg.data.dataset.get("img_size", cfg.img_size),
        )

        n_train = int(len(dataset) * cfg.train_split)
        n_val = len(dataset) - n_train
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=rnd_gen,
        )

    else:
        # Original path for PushT, OGBench, etc.
        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue

                normalizer = get_column_normalizer(dataset, col, col)
                transforms.append(normalizer)

                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
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

    # MODIFIED: ARPredictor now handles unified sequence (action tokens + state prediction)
    predictor = ARPredictor(
        embed_dim=embed_dim,
        max_action_tokens=max_action_tokens,
        **cfg.predictor,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # MODIFIED: no action_encoder parameter
    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        pred_proj=predictor_proj,
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
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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
