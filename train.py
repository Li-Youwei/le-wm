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
    gripper_aux_weight = float(loss_cfg.get("gripper_aux_weight", 0.0))
    use_state_pred = pred_weight > 0
    use_gripper_aux = gripper_aux_weight > 0

    # Pre-initialize future-frame latents so the SIGReg block can reference
    # them unconditionally without static-analysis warnings, and so the
    # legitimate "SIGReg-only, no SP" ablation (pred_weight=0,
    # sigreg_weight>0) reads cleanly. They are populated in step 6 only
    # when use_state_pred is True.
    z_agent_future: torch.Tensor | None = None
    z_hand_future: torch.Tensor | None = None

    # 1. Unpack batch
    pixels_agent = batch["pixels_agent"]  # (B, 3, H, W)
    pixels_hand = batch["pixels_hand"]  # (B, 3, H, W)
    proprio = batch["proprio"]  # (B, 9) ee_pos(3)+xyzw_quat(4)+grip_raw(2)
    fast_tokens = batch["fast_tokens"]  # (B, max_action_tokens)
    fast_lengths = batch["fast_lengths"]  # (B,)

    # Language batch fields may be absent in no-language ablation runs
    lang_ids = batch.get("lang_input_ids", None)  # (B, max_lang_tokens) or None
    lang_mask = batch.get("lang_attention_mask", None)  # (B, max_lang_tokens) or None

    # 2. Encode visual + language (language skipped when lang_ids is None)
    z_agent, z_hand, lang_embeds, lang_lengths = self.model.encode(
        pixels_agent,
        pixels_hand,
        lang_ids,
        lang_mask,
    )

    # 3. Predict action logits (teacher forcing) and optional state predictions
    pred_out = self.model.predict(
        z_agent,
        z_hand,
        proprio,
        lang_embeds,
        lang_lengths,
        fast_tokens,
        fast_lengths,
    )
    # Unpack predictor output. Shape depends on (use_state_pred, use_gripper_aux):
    #   (F, F) → action_logits
    #   (F, T) → (action_logits, pred_grip)
    #   (T, F) → (action_logits, pred_ag, pred_hd, pred_pr)
    #   (T, T) → (action_logits, pred_ag, pred_hd, pred_pr, pred_grip)
    pred_grip = None
    if use_state_pred and use_gripper_aux:
        action_logits, pred_ag, pred_hd, pred_pr, pred_grip = pred_out
    elif use_state_pred:
        action_logits, pred_ag, pred_hd, pred_pr = pred_out
    elif use_gripper_aux:
        action_logits, pred_grip = pred_out
    else:
        action_logits = pred_out

    # 4. Build CE targets (shifted by 1: position j predicts token j+1)
    B = z_agent.size(0)
    num_action_positions = action_logits.size(1)  # 1 (BOS) + max_action_tokens
    targets = torch.full(
        (B, num_action_positions), -100, dtype=torch.long, device=z_agent.device
    )
    for i in range(B):
        k = fast_lengths[i].item()
        targets[i, :k] = fast_tokens[i, :k]  # T_1, T_2, ..., T_k
        targets[i, k] = EOS_TOKEN_ID  # EOS after last real token

    # 5. L_CE (always). Computed with reduction="none" so we can aggregate
    # two ways from a single forward:
    #   - output["ce_loss"]: token-level mean (sum-over-tokens / count-of-valid),
    #     preserves the original training gradient semantics.
    #   - output["ce_per_sample"]: per-sample mean — used by validation to bucket
    #     by task_id and log a per-task / task-balanced CE metric (the
    #     ckpt-selection signal under 4-suite joint training where suites have
    #     unequal val counts).
    # label_smoothing defaults to 0.1 for backward compat; override via
    # `label_smoothing=0` on the CLI to run a no-regularization ablation.
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    output = {}
    ce_per_token = F.cross_entropy(
        action_logits.reshape(-1, ACTION_HEAD_SIZE),
        targets.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction="none",
    ).reshape(B, num_action_positions)
    valid_mask = (targets != -100).float()
    n_valid_total = valid_mask.sum().clamp(min=1)
    output["ce_loss"] = (ce_per_token * valid_mask).sum() / n_valid_total
    # Per-sample mean (each sample weighted equally regardless of token count).
    n_valid_per_sample = valid_mask.sum(dim=1).clamp(min=1)
    output["ce_per_sample"] = (
        (ce_per_token * valid_mask).sum(dim=1) / n_valid_per_sample
    ).detach()

    total_loss = output["ce_loss"]

    # 6. State-prediction loss (LeWM-style, NO stop-gradient on target).
    if use_state_pred:
        pixels_agent_future = batch["pixels_agent_future"]
        pixels_hand_future = batch["pixels_hand_future"]
        proprio_future = batch["proprio_future"]  # (B, 9) raw target

        # Encode future visual through the SAME encoder + projector. Both
        # the source path (z_agent/z_hand of the current frame, computed
        # in step 2) and the target path (here) get gradients — that's
        # the LeWM "no heuristics" recipe; SIGReg is the only thing
        # holding off collapse.
        z_agent_future, z_hand_future = self.model.encode_future_visual(
            pixels_agent_future,
            pixels_hand_future,
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
    # `encode()` returns (B, N, D); for SIGReg we use the CLS slice only
    # (see encode_future_visual docstring — SP heads + SIGReg both operate
    # on a single per-view embedding, multi-token prefix is a predictor
    # concern). encode_future_visual already returns (B, D).
    if sigreg_weight > 0:
        z_ag_cls = z_agent[:, 0] if z_agent.dim() == 3 else z_agent
        z_hd_cls = z_hand[:, 0] if z_hand.dim() == 3 else z_hand
        if not use_state_pred:
            # Without state prediction we don't have z_*_future encoded —
            # SIGReg degenerates to 2 streams (current views only).
            sigreg_input = torch.stack([z_ag_cls, z_hd_cls], dim=0)
        else:
            sigreg_input = torch.stack(
                [z_ag_cls, z_hd_cls, z_agent_future, z_hand_future],
                dim=0,
            )
        output["sigreg_loss"] = self.sigreg(sigreg_input)
        total_loss = total_loss + sigreg_weight * output["sigreg_loss"]

    # 7b. Gripper auxiliary loss. Direct (H,) regression against the
    # already-normalized gripper command sequence from the dataset. The
    # gripper dim in continuous_actions is in [-1, 1] (low=-1, high=+1 for
    # that channel), matching the Tanh output range of the aux head, so the
    # MSE is well-scaled with no extra normalization.
    if use_gripper_aux:
        gripper_seq = batch["gripper_seq"]  # (B, H) float32 in [-1, 1]
        output["gripper_aux_loss"] = F.mse_loss(pred_grip, gripper_seq)
        total_loss = total_loss + gripper_aux_weight * output["gripper_aux_loss"]

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
    if "gripper_aux_loss" in output:
        log_dict[f"{stage}/gripper_aux_loss"] = output["gripper_aux_loss"].detach()
    # Log learning rate if available
    if hasattr(self, "trainer") and self.trainer is not None:
        opts = self.trainer.optimizers
        if opts:
            lr = opts[0].param_groups[0]["lr"]
            log_dict[f"{stage}/lr"] = lr
    self.log_dict(log_dict, on_step=True, on_epoch=True, sync_dist=True)

    # 9b. Per-task validation CE for task-balanced ckpt selection.
    # During training the WeightedRandomSampler already balances per-task,
    # so per-task train CE is uninformative; only log during validate to
    # keep TB scalar count manageable. The TaskBalancedCEMetric callback
    # aggregates these into validate/ce_loss_taskbal at epoch end.
    if stage == "validate" and "task_id" in batch:
        task_ids = batch["task_id"]  # (B,)
        ce_per_sample = output["ce_per_sample"]
        for t in task_ids.unique().tolist():
            mask = task_ids == t
            if mask.any():
                self.log(
                    f"validate/ce_loss/task_{int(t)}",
                    ce_per_sample[mask].mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

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
    gripper_aux_weight = float(loss_cfg_run.get("gripper_aux_weight", 0.0))
    use_state_prediction = pred_weight > 0
    use_gripper_aux = gripper_aux_weight > 0

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

    # Multi-token visual prefix (CLS + grid-pooled patches). Off by default
    # (visual_pool_grid=0 → CLS only, same as legacy single-token prefix).
    # When > 0, the encoder output is fed as (B, 1 + G*G, D) per view through
    # the projector. JEPA.encode now reshapes (B, N, D) → (B*N, D) before the
    # projector so BatchNorm1d sees a flat (Batch, Channels) input — the
    # combo multi-token + BN is therefore unblocked.
    visual_cfg = cfg.get("visual_tokens", {}) or {}
    visual_pool_grid = int(visual_cfg.get("pool_grid", 0))
    n_visual_per_view = (
        1 + visual_pool_grid * visual_pool_grid if visual_pool_grid > 0 else 1
    )
    print(
        f"[visual_tokens] pool_grid={visual_pool_grid}, "
        f"n_visual_per_view={n_visual_per_view}"
    )

    # Multi-GPU + plain BatchNorm = silent divergence. nn.BatchNorm1d computes
    # statistics per-GPU, so without SyncBatchNorm each rank sees a different
    # normalization → encoder representations drift apart with no error
    # signal. Hard-fail rather than warn — debugging silent BN drift later
    # costs more than this guard. Setting devices=1 (or [N]) is the safe
    # path; SyncBatchNorm conversion is not yet auto-applied.
    if projector_norm == "batch":
        devices_cfg = cfg.trainer.get("devices", "auto")
        is_explicit_single_gpu = devices_cfg == 1 or (
            isinstance(devices_cfg, list) and len(devices_cfg) == 1
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
    # Use (fpath, demo_idx) as the unique demo key — under joint 4-suite training,
    # demo_0 from different tasks must be split independently, not lumped together.
    # dataset._demo_ids was already cached during LiberoDataset.__init__ (one read
    # per file there, no re-opens here).
    demo_keys: list[tuple] = [
        (fpath, dataset._demo_ids[i])
        for i, (fpath, _local_idx) in enumerate(dataset._index)
    ]
    # Keep a flat int view for the overfit_demo path (which selects by demo_idx alone).
    demo_ids = [k[1] for k in demo_keys]

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
        unique_demo_keys = sorted(set(demo_keys))
        n_train_demos = int(len(unique_demo_keys) * cfg.train_split)
        perm = torch.randperm(len(unique_demo_keys), generator=rnd_gen).tolist()
        train_demo_set = {unique_demo_keys[perm[i]] for i in range(n_train_demos)}

        train_indices = [i for i, k in enumerate(demo_keys) if k in train_demo_set]
        val_indices = [i for i, k in enumerate(demo_keys) if k not in train_demo_set]
        print(
            f"[Split] {len(unique_demo_keys)} unique (task, demo) pairs → "
            f"{n_train_demos} train / {len(unique_demo_keys) - n_train_demos} val. "
            f"Train chunks: {len(train_indices)}, val chunks: {len(val_indices)}."
        )

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    # In overfit mode, keep every sample each epoch — drop_last=True could
    # discard the only batch when the sample count is smaller than batch_size.
    train_drop_last = overfit_demo is None

    # 3-level balanced sampling (task → demo → time uniform) for the train loader.
    # Skipped in overfit mode (we want the same chunks every step).
    if overfit_demo is None:
        from torch.utils.data import WeightedRandomSampler

        all_weights = dataset.get_sampler_weights()
        train_weights = all_weights[train_indices]
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
            generator=rnd_gen,
        )
        # cfg.loader has no `shuffle` key (verified in lewm.yaml), so passing
        # both **cfg.loader and sampler=... is safe. WeightedRandomSampler is
        # incompatible with shuffle=True.
        train = torch.utils.data.DataLoader(
            train_set,
            **cfg.loader,
            sampler=train_sampler,
            drop_last=train_drop_last,
        )
    else:
        train = torch.utils.data.DataLoader(
            train_set,
            **cfg.loader,
            shuffle=True,
            drop_last=train_drop_last,
            generator=rnd_gen,
        )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

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
    # Gripper aux head reads chunk_size from the dataset config so the head's
    # output dim matches batch["gripper_seq"].shape[1]. We treat it as
    # immutable (H=20 across the project); pull from data config to surface
    # mismatches loudly.
    gripper_chunk_size = int(cfg.data.dataset.get("chunk_size", 20))
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
        # Multi-token visual prefix. 0 = CLS-only legacy layout, matches
        # spatial-65% / sp_sigreg-67.8% checkpoints exactly.
        visual_pool_grid=visual_pool_grid,
        use_gripper_aux=use_gripper_aux,
        gripper_chunk_size=gripper_chunk_size,
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
        print(
            "[Ablation] use_language=False — skipping T5 encoder and language projection."
        )
        lang_encoder = None
        lang_proj = None

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
        visual_pool_grid=visual_pool_grid,
    )

    # Cosine-annealing scheduler with explicit warmup_steps / max_steps.
    # spt's smart-defaults factory pulls these from `trainer.estimated_stepping_batches`,
    # but at scheduler-construction time (inside configure_optimizers) the
    # dataloader hasn't been bound yet and the property can return None — which
    # caused a `missing required arg max_steps` TypeError on the first run.
    # Resolve the budget here:
    #   - If trainer.max_steps is set → use it directly.
    #   - Else → estimate from max_epochs × len(train_loader).
    # Warmup is 2% of total steps, capped at 2K (rough rule for VLA-scale runs).
    trainer_cfg_sched = cfg.get("trainer", {}) or {}
    cfg_max_steps = int(trainer_cfg_sched.get("max_steps", -1))
    if cfg_max_steps > 0:
        sched_max_steps = cfg_max_steps
    else:
        # Fall back to epoch-budgeted estimate; len(train) is exact at this point.
        steps_per_epoch = max(1, len(train))
        epochs_budget = int(trainer_cfg_sched.get("max_epochs", 1))
        sched_max_steps = max(1, steps_per_epoch * epochs_budget)
    sched_warmup = max(1, min(2000, int(0.02 * sched_max_steps)))
    print(
        f"[scheduler] LinearWarmupCosineAnnealingLR: max_steps={sched_max_steps}, "
        f"warmup_steps={sched_warmup}, interval=step"
    )
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "max_steps": sched_max_steps,
                "warmup_steps": sched_warmup,
            },
            "interval": "step",
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

    # Checkpointing cadence: step-based when trainer.max_steps is set
    # (4-suite joint run), per-epoch fall-back for legacy runs.
    trainer_cfg = cfg.get("trainer", {})
    max_steps_cfg = int(trainer_cfg.get("max_steps", -1))
    val_check_int = trainer_cfg.get("val_check_interval", None)
    step_save_interval = (
        int(val_check_int)
        if (max_steps_cfg > 0 and val_check_int is not None)
        else None
    )
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
        step_interval=step_save_interval,
        top_k=int(cfg.get("ckpt_top_k", 3)),
    )

    callbacks = [object_dump_callback]

    # Task-balanced val CE metric — aggregates per-task scalars logged in
    # lejepa_forward and writes validate/ce_loss_taskbal at val end. Used by
    # pick_best_ckpt.py for ckpt selection under joint 4-suite training.
    from utils import TaskBalancedCEMetric

    callbacks.append(TaskBalancedCEMetric())

    # Online health probe at step 20K (Stage A: full 40-rollout breadth sweep).
    # Disabled by default; opt in via cfg.probe.enabled=true on the CLI.
    probe_cfg = cfg.get("probe", {}) or {}
    if probe_cfg.get("enabled", False):
        from utils import EarlyProbeCallback

        callbacks.append(
            EarlyProbeCallback(
                trigger_steps=tuple(probe_cfg.get("trigger_steps", (20000,))),
                tokenizer_path=str(probe_cfg["tokenizer_path"]),
                processed_root=str(probe_cfg["processed_root"]),
                ckpt_dir=str(run_dir),
                script_path=str(probe_cfg.get("script", "quick_probe_eval.py")),
            )
        )

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
