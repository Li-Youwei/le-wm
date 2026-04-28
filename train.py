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
from module import ARPredictor, MLP, SIGReg, ACTION_HEAD_SIZE, EOS_TOKEN_ID
from utils import ModelObjectCallBack, PeriodicPrintCallback


def lejepa_forward(self, batch, stage, cfg):
    """VLA training step.

    Always computes ``L_CE`` (action token cross-entropy).

    When ``cfg.loss.pred_weight > 0``: also computes ``L_pred`` — three MSE
    losses against the future-frame encoder outputs (visual) and the raw
    9-d future proprio. Per LeWM paper Section 3, the target encoder
    branch does NOT use stop-gradient; SIGReg is what prevents collapse.

    When ``cfg.loss.sigreg_weight > 0``: also computes ``L_sigreg`` on the
    encoder outputs only — Algorithm 1 strict (4 streams in our setup:
    z_ag_t, z_hd_t, z_ag_{t+H}, z_hd_{t+H}). Predictor outputs (ẑ) are
    NOT included; that matches the paper's Algorithm 1 listing rather
    than Figure 1's looser visual.
    """
    # `cfg.loss` may be absent in overfit.yaml (which doesn't define a
    # loss: section); guard with cfg.get(...) so both configs work.
    loss_cfg = cfg.get("loss", {}) or {}
    pred_weight = float(loss_cfg.get("pred_weight", 0.0))
    sigreg_weight = float(loss_cfg.get("sigreg_weight", 0.0))
    use_state_pred = pred_weight > 0

    # Pre-initialize future-frame latents so the SIGReg block can reference
    # them unconditionally without static-analysis warnings, and so the
    # legitimate "SIGReg-only, no SP" ablation (pred_weight=0,
    # sigreg_weight>0) reads cleanly. They are populated in step 6 only
    # when use_state_pred is True.
    z_agent_future: torch.Tensor | None = None
    z_hand_future: torch.Tensor | None = None

    # 1. Unpack batch
    pixels_agent = batch["pixels_agent"]          # (B, 3, H, W)
    pixels_hand = batch["pixels_hand"]            # (B, 3, H, W)
    proprio = batch["proprio"]                    # (B, 9) ee_pos(3)+xyzw_quat(4)+grip_raw(2)
    fast_tokens = batch["fast_tokens"]            # (B, max_action_tokens)
    fast_lengths = batch["fast_lengths"]          # (B,)

    # Language batch fields may be absent in no-language ablation runs
    lang_ids = batch.get("lang_input_ids", None)   # (B, max_lang_tokens) or None
    lang_mask = batch.get("lang_attention_mask", None)  # (B, max_lang_tokens) or None

    # 2. Encode visual + language (language skipped when lang_ids is None)
    z_agent, z_hand, lang_embeds, lang_lengths = self.model.encode(
        pixels_agent, pixels_hand, lang_ids, lang_mask,
    )

    # 3. Predict action logits (teacher forcing) and optional state predictions
    pred_out = self.model.predict(
        z_agent, z_hand, proprio, lang_embeds, lang_lengths,
        fast_tokens, fast_lengths,
    )
    if use_state_pred:
        action_logits, pred_ag, pred_hd, pred_pr = pred_out
    else:
        action_logits = pred_out

    # 4. Build CE targets (shifted by 1: position j predicts token j+1)
    B = z_agent.size(0)
    num_action_positions = action_logits.size(1)  # 1 (BOS) + max_action_tokens
    targets = torch.full((B, num_action_positions), -100, dtype=torch.long, device=z_agent.device)
    for i in range(B):
        k = fast_lengths[i].item()
        targets[i, :k] = fast_tokens[i, :k]    # T_1, T_2, ..., T_k
        targets[i, k] = EOS_TOKEN_ID           # EOS after last real token

    # 5. L_CE (always)
    # label_smoothing defaults to 0.1 for backward compat; override via
    # `label_smoothing=0` on the CLI to run a no-regularization ablation.
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    output = {}
    output["ce_loss"] = F.cross_entropy(
        action_logits.reshape(-1, ACTION_HEAD_SIZE),
        targets.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )

    total_loss = output["ce_loss"]

    # 6. State-prediction loss (LeWM-style, NO stop-gradient on target).
    if use_state_pred:
        pixels_agent_future = batch["pixels_agent_future"]
        pixels_hand_future = batch["pixels_hand_future"]
        proprio_future = batch["proprio_future"]   # (B, 9) raw target

        # Encode future visual through the SAME encoder + projector. Both
        # the source path (z_agent/z_hand of the current frame, computed
        # in step 2) and the target path (here) get gradients — that's
        # the LeWM "no heuristics" recipe; SIGReg is the only thing
        # holding off collapse.
        z_agent_future, z_hand_future = self.model.encode_future_visual(
            pixels_agent_future, pixels_hand_future,
        )

        loss_pred_ag = F.mse_loss(pred_ag, z_agent_future)
        loss_pred_hd = F.mse_loss(pred_hd, z_hand_future)
        loss_pred_pr = F.mse_loss(pred_pr, proprio_future)

        output["pred_loss"] = loss_pred_ag + loss_pred_hd + loss_pred_pr
        output["pred_loss_ag"] = loss_pred_ag
        output["pred_loss_hd"] = loss_pred_hd
        output["pred_loss_pr"] = loss_pred_pr

        total_loss = total_loss + pred_weight * output["pred_loss"]

    # 7. SIGReg loss (Algorithm 1 strict — encoder outputs only).
    # The 4-stream stack maps to LeWM's `emb` over 2 timesteps × 2 views.
    # Predictor outputs (pred_ag/pred_hd) are intentionally NOT included
    # — see paper Algorithm 1 + upstream `train.py::lejepa_forward`.
    if sigreg_weight > 0:
        if not use_state_pred:
            # Without state prediction we don't have z_*_future encoded —
            # SIGReg degenerates to 2 streams (current views only).
            sigreg_input = torch.stack([z_agent, z_hand], dim=0)
        else:
            sigreg_input = torch.stack(
                [z_agent, z_hand, z_agent_future, z_hand_future], dim=0,
            )
        output["sigreg_loss"] = self.sigreg(sigreg_input)
        total_loss = total_loss + sigreg_weight * output["sigreg_loss"]

    output["loss"] = total_loss

    # 8. Token accuracy (diagnostic)
    with torch.no_grad():
        preds = action_logits.argmax(dim=-1)
        valid = targets != -100
        correct = (preds == targets) & valid
        n_valid = valid.sum().float()
        output["token_accuracy"] = correct.sum().float() / n_valid.clamp(min=1)

    # 9. Logging
    log_dict = {
        f"{stage}/ce_loss": output["ce_loss"].detach(),
        f"{stage}/total_loss": output["loss"].detach(),
        f"{stage}/token_accuracy": output["token_accuracy"],
    }
    if "pred_loss" in output:
        log_dict[f"{stage}/pred_loss"] = output["pred_loss"].detach()
        log_dict[f"{stage}/pred_loss_ag"] = output["pred_loss_ag"].detach()
        log_dict[f"{stage}/pred_loss_hd"] = output["pred_loss_hd"].detach()
        log_dict[f"{stage}/pred_loss_pr"] = output["pred_loss_pr"].detach()
    if "sigreg_loss" in output:
        log_dict[f"{stage}/sigreg_loss"] = output["sigreg_loss"].detach()
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

    # Loss-weight-driven feature toggles. State prediction needs the
    # extra HDF5 fields (image_*_future, proprio_future) and the 3
    # STATE_QUERY tokens in the predictor; both are auto-enabled when
    # cfg.loss.pred_weight > 0 so users only have to set one knob.
    loss_cfg_run = cfg.get("loss", {}) or {}
    pred_weight = float(loss_cfg_run.get("pred_weight", 0.0))
    sigreg_weight = float(loss_cfg_run.get("sigreg_weight", 0.0))
    use_state_prediction = pred_weight > 0

    # Projector normalization: must be 'batch' when SIGReg is enabled
    # (LeWM paper Section 3 — LayerNorm prevents the anti-collapse
    # objective from being optimized). Default 'layer' reproduces the
    # frozen baseline exactly.
    projector_norm = cfg.get("projector", {}).get("norm_type", "layer")
    if projector_norm not in ("layer", "batch"):
        raise ValueError(
            f"projector.norm_type must be 'layer' or 'batch', got '{projector_norm}'"
        )
    if sigreg_weight > 0 and projector_norm != "batch":
        raise ValueError(
            f"sigreg_weight={sigreg_weight} > 0 requires projector.norm_type='batch'. "
            f"Got '{projector_norm}'. LeWM paper Section 3: LayerNorm projector "
            "prevents SIGReg from optimizing the latent distribution toward "
            "isotropic Gaussian. See CLAUDE.md 'Known caveats of switching to "
            "BatchNorm projector' before flipping this on."
        )

    # Multi-GPU + plain BatchNorm = silent divergence. nn.BatchNorm1d computes
    # statistics per-GPU, so without SyncBatchNorm each rank sees a different
    # normalization → encoder representations drift apart with no error
    # signal. Hard-fail rather than warn — debugging silent BN drift later
    # costs more than this guard. Setting devices=1 (or [N]) is the safe
    # path; SyncBatchNorm conversion is not yet auto-applied.
    if projector_norm == "batch":
        devices_cfg = cfg.trainer.get("devices", "auto")
        is_explicit_single_gpu = (
            devices_cfg == 1
            or (isinstance(devices_cfg, list) and len(devices_cfg) == 1)
        )
        if not is_explicit_single_gpu:
            raise ValueError(
                f"projector.norm_type='batch' is unsafe with cfg.trainer.devices="
                f"{devices_cfg!r}. BatchNorm without "
                "torch.nn.SyncBatchNorm.convert_sync_batchnorm gives per-GPU "
                "statistics → silent divergence across ranks. Either: "
                "(a) set trainer.devices=1, or (b) wrap the model in "
                "SyncBatchNorm before training (not yet auto-applied). "
                "'auto' is also rejected because it can resolve to >1 GPUs "
                "without warning."
            )

    from libero_dataset import LiberoDataset

    dataset = LiberoDataset(
        hdf5_dir=cfg.data.dataset.hdf5_dir,
        max_action_tokens=max_action_tokens,
        max_lang_tokens=max_lang_tokens,
        img_size=cfg.data.dataset.get("img_size", cfg.img_size),
        use_language=use_language,
        use_state_prediction=use_state_prediction,
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
        use_state_prediction=use_state_prediction,
        # Mirror the encoder-side projector's norm choice — paper Section 3
        # says the predictor projector has the "same implementation as the
        # one used for the encoder", so when the encoder projector is BN
        # (sigreg-on path) the state heads must also be BN.
        state_head_norm_type=projector_norm,
        **cfg.predictor,
    )

    norm_fn = torch.nn.BatchNorm1d if projector_norm == "batch" else torch.nn.LayerNorm
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=norm_fn,
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

    # SIGReg module — only constructed when enabled. spt.Module accepts
    # arbitrary kwargs and exposes them as attributes, so passing
    # `sigreg=...` makes `self.sigreg` available inside `lejepa_forward`.
    spt_module_kwargs = dict(
        model=world_model,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    if sigreg_weight > 0:
        sigreg_kwargs = loss_cfg_run.get("sigreg", {}).get("kwargs", {})
        spt_module_kwargs["sigreg"] = SIGReg(**sigreg_kwargs)
    world_model = spt.Module(**spt_module_kwargs)

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
