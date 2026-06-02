import os
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
    pixels_agent = batch["pixels_agent"]  # (B, 3, H, W)
    pixels_hand = batch["pixels_hand"]  # (B, 3, H, W)
    proprio = batch["proprio"]  # (B, 9) ee_pos(3)+xyzw_quat(4)+grip_raw(2)
    fast_tokens = batch["fast_tokens"]  # (B, max_action_tokens)
    fast_lengths = batch["fast_lengths"]  # (B,)

    lang_ids = batch["lang_input_ids"]  # (B, max_lang_tokens)
    lang_mask = batch["lang_attention_mask"]  # (B, max_lang_tokens)

    # 2. Encode visual + language
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
    # Unpack predictor output. Shape depends on use_state_pred:
    #   F → action_logits
    #   T → (action_logits, pred_ag, pred_hd, pred_pr)
    if use_state_pred:
        action_logits, pred_ag, pred_hd, pred_pr = pred_out
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
    n_valid_local = valid_mask.sum()
    ce_sum_local = (ce_per_token * valid_mask).sum()
    # Token-mean CE that is EXACT across GPU counts (for accumulate_grad_batches
    # == 1). The naive per-rank sum/local-count then DDP-averaged is NOT equal to
    # a single big-batch token-mean when ranks have unequal valid-token counts
    # (FAST seqs vary in length) — each rank's tokens get weighted by 1/n_local
    # instead of the global 1/n_global. Fix: normalize by the GLOBAL token count
    # (SUM-reduced across ranks, non-differentiable) and multiply by world_size
    # to cancel DDP's gradient averaging (÷W). Single-GPU (W=1, n_global=n_local)
    # is bit-identical to the original sum/count. Stage="fit" only — val keeps
    # the plain per-batch mean (no backward, comparable per-task numbers).
    # CAVEAT: under gradient accumulation (accum>1) each micro-batch is
    # normalized by ITS OWN n_global (the streaming accumulator can't know the
    # window's total token count up front), so the token-mean is APPROXIMATE
    # along the accumulation axis — same ~1-4% variance as the cross-GPU case,
    # and no worse than the old code (which was approximate on both axes). Exact
    # only at accum=1. The *world factor assumes DDP syncs gradients every
    # micro-batch (spt's manual-opt training_step does manual_backward without a
    # no_sync() wrapper, so this holds).
    import torch.distributed as _dist

    if (
        stage == "fit"
        and _dist.is_available()
        and _dist.is_initialized()
        and _dist.get_world_size() > 1
    ):
        world = _dist.get_world_size()
        n_global = n_valid_local.detach().clone()
        _dist.all_reduce(n_global, op=_dist.ReduceOp.SUM)
        output["ce_loss"] = ce_sum_local * world / n_global.clamp(min=1)
    else:
        output["ce_loss"] = ce_sum_local / n_valid_local.clamp(min=1)
    # Per-sample mean (each sample weighted equally regardless of token count).
    n_valid_per_sample = valid_mask.sum(dim=1).clamp(min=1)
    output["ce_per_sample"] = (
        (ce_per_token * valid_mask).sum(dim=1) / n_valid_per_sample
    ).detach()

    total_loss = output["ce_loss"]

    # 6. Multi-horizon state-prediction loss (LeWM-style, NO stop-gradient).
    if use_state_pred:
        # Futures now carry a horizon axis K: (B, K, 3, H, W) / (B, K, 9).
        pixels_agent_future = batch["pixels_agent_future"]  # (B, K, 3, H, W)
        pixels_hand_future = batch["pixels_hand_future"]  # (B, K, 3, H, W)
        proprio_future = batch["proprio_future"]  # (B, K, 9) raw target

        # Encode all K future horizons through the SAME (now finetuned) encoder
        # + projector → (B, K, nv, D) per view. No stop-grad (LeWM recipe);
        # SIGReg is the anti-collapse guard, which matters more now the encoder
        # is trainable. SP heads emit matching (B, K, nv, D) so the MSE is
        # token- and horizon-aligned, routing SP gradient to every source patch.
        z_agent_future, z_hand_future = self.model.encode_future_visual(
            pixels_agent_future,
            pixels_hand_future,
        )  # each (B, K, nv, D)

        # The model emits one prediction per STATE_QUERY horizon; the data
        # carries one future per preprocess --sp-horizons offset. These K must
        # match, else the per-horizon MSE would broadcast silently (e.g. model
        # K=1 vs data K=4). Fail loudly instead.
        if pred_ag.size(1) != z_agent_future.size(1):
            raise ValueError(
                f"Horizon mismatch: predictor emits K={pred_ag.size(1)} "
                f"(len(cfg.loss.state_pred_horizons)) but the data has K="
                f"{z_agent_future.size(1)} future frames (preprocess "
                "--sp-horizons). Align cfg.loss.state_pred_horizons with the "
                "preprocessed *_future axis."
            )

        # Per-horizon weights (near-heavy), normalized to sum=1 so total SP
        # magnitude matches the single-horizon scale → pred_weight unchanged.
        K = z_agent_future.size(1)
        w = torch.tensor(
            list(loss_cfg.get("state_pred_horizon_weights", [1.0] * K)),
            device=z_agent_future.device,
            dtype=z_agent_future.dtype,
        )
        if w.numel() != K:
            raise ValueError(
                f"state_pred_horizon_weights length {w.numel()} != number of "
                f"horizons K={K} (from the data's *_future axis). Align "
                "cfg.loss.state_pred_horizon_weights / state_pred_horizons with "
                "preprocess_libero.py --sp-horizons."
            )
        w = w / w.sum().clamp(min=1e-8)  # (K,)

        def _w_mse(pred, target):
            # mean over all dims except (batch, horizon) → (B,K) → mean over
            # batch → (K,); then horizon-weighted sum.
            per_h = ((pred - target) ** 2).flatten(2).mean(dim=-1).mean(dim=0)
            return (w * per_h).sum()

        loss_pred_ag = _w_mse(pred_ag, z_agent_future)
        loss_pred_hd = _w_mse(pred_hd, z_hand_future)
        loss_pred_pr = _w_mse(pred_pr, proprio_future)

        output["pred_loss"] = loss_pred_ag + loss_pred_hd + loss_pred_pr
        output["pred_loss_ag"] = loss_pred_ag
        output["pred_loss_hd"] = loss_pred_hd
        output["pred_loss_pr"] = loss_pred_pr
        # Per-horizon agent loss (unweighted) — watch for far-horizon
        # free-riding / collapse; drives the 12tok / causal fallbacks.
        with torch.no_grad():
            per_h_ag = (
                ((pred_ag - z_agent_future) ** 2).flatten(2).mean(dim=-1).mean(dim=0)
            )
            for k in range(K):
                output[f"pred_loss_ag_h{k}"] = per_h_ag[k].detach()

        total_loss = total_loss + pred_weight * output["pred_loss"]

    # 7. SIGReg loss (Algorithm 1 strict — encoder CLS outputs only).
    # Stack the 2 current-view CLS + ALL K future-horizon CLS (both views) so
    # anti-collapse covers every predicted horizon, not just one — important
    # now the encoder is trainable (each future encode is a collapse channel).
    # Predictor outputs (pred_ag/pred_hd) stay excluded (paper Algorithm 1).
    if sigreg_weight > 0:
        z_ag_cls = z_agent[:, 0] if z_agent.dim() == 3 else z_agent
        z_hd_cls = z_hand[:, 0] if z_hand.dim() == 3 else z_hand
        streams = [z_ag_cls, z_hd_cls]
        if use_state_pred:
            # z_*_future: (B, K, nv, D) → per-horizon CLS (B, D) for all K.
            Kf = z_agent_future.size(1)
            for k in range(Kf):
                streams.append(z_agent_future[:, k, 0])
                streams.append(z_hand_future[:, k, 0])
        sigreg_input = torch.stack(streams, dim=0)  # (2 + 2K, B, D)
        output["sigreg_loss"] = self.sigreg(sigreg_input)
        total_loss = total_loss + sigreg_weight * output["sigreg_loss"]

    # Gradient accumulation: spt accumulates grads over `frequency` micro-batches
    # but does NOT rescale the loss, so the summed gradient would be N x too
    # large. Divide the loss handed to backward (output["loss"]) by N so the
    # effective update matches a single batch of (micro_batch * N). Only in the
    # fit stage (val never backwards). N=1 is a no-op. Logging below uses the
    # UNDIVIDED total_loss so curves stay comparable across accum settings.
    accum = max(1, int(cfg.get("accumulate_grad_batches", 1)))
    output["loss"] = (
        total_loss / accum if (stage == "fit" and accum > 1) else total_loss
    )

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
        # Undivided total (output["loss"] may be /accum for backward) so curves
        # stay comparable across accumulate_grad_batches settings.
        f"{stage}/total_loss": total_loss.detach(),
        f"{stage}/token_accuracy": output["token_accuracy"],
    }
    if "pred_loss" in output:
        log_dict[f"{stage}/pred_loss"] = output["pred_loss"].detach()
        log_dict[f"{stage}/pred_loss_ag"] = output["pred_loss_ag"].detach()
        log_dict[f"{stage}/pred_loss_hd"] = output["pred_loss_hd"].detach()
        log_dict[f"{stage}/pred_loss_pr"] = output["pred_loss_pr"].detach()
        # Per-horizon agent SP loss (h0=nearest .. hK-1=farthest).
        for key in output:
            if key.startswith("pred_loss_ag_h"):
                log_dict[f"{stage}/{key}"] = output[key].detach()
    if "sigreg_loss" in output:
        log_dict[f"{stage}/sigreg_loss"] = output["sigreg_loss"].detach()
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
                    # sync_dist=False is REQUIRED under DDP: the set of task_ids
                    # present differs per rank/batch, so sync_dist=True would emit
                    # a per-rank-VARIABLE number of all_reduce collectives → the
                    # ranks desync (mismatched collective fingerprints) and NCCL
                    # hangs at the next collective. The val loader is unsharded
                    # (use_distributed_sampler=False) so every rank already sees
                    # the FULL val set; rank-0's local per-task CE is the global
                    # value, and TaskBalancedCEMetric reads it from callback_metrics
                    # (then logs ce_loss_taskbal once with sync_dist=True — a single
                    # fixed collective, safe).
                    sync_dist=False,
                )

    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # Seed the global torch / numpy / random / hash RNG BEFORE constructing
    # encoder, predictor, projector, lang_proj, etc. — otherwise their
    # parameter init draws from whatever state numpy/torch happen to be in,
    # which makes runs non-reproducible across reboots. workers=True propagates
    # the seed into DataLoader worker processes for deterministic shuffling.
    # spt.Manager re-seeds again inside its __call__, so we ALSO pass seed
    # there (below) to make that second seeding deterministic and aligned.
    pl.seed_everything(cfg.seed, workers=True)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    # --- Single source of truth for DDP topology (used by the BN guard, the
    # sampler shard, and the DDP/accum wiring — these MUST agree). ---
    # World size is derived from cfg.trainer.devices, NOT os.environ['WORLD_SIZE']:
    # under Lightning's subprocess DDP launcher, WORLD_SIZE is set in the CHILD
    # ranks but NOT in the rank-0 PARENT at the time run() builds the sampler
    # (the launcher only exports it from inside trainer.fit()). Reading env there
    # would make the parent build a full-size sampler while children build
    # sharded ones → mismatched epoch lengths → DDP hang. cfg.trainer.devices is
    # known identically on every process at run() time (run_all4 sets it to
    # $NUM_GPU). 'auto'/-1 are treated as potentially-multi and REJECTED for the
    # SIGReg/BN path (can't know the count up front); pass an explicit int.
    _dev = cfg.trainer.get("devices", "auto")
    if isinstance(_dev, int) and _dev >= 1:
        ddp_world = _dev
        ddp_devices_known = True
    elif isinstance(_dev, (list, tuple)):
        ddp_world = len(_dev)
        ddp_devices_known = True
    else:  # "auto" | -1 | None → count not statically known
        ddp_world = 1
        ddp_devices_known = False
    ddp_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    multi_gpu = ddp_world > 1

    # Single source of truth for max_action_tokens and max_lang_tokens
    max_action_tokens = cfg.data.dataset.get("max_action_tokens", 80)
    max_lang_tokens = cfg.data.dataset.get("max_lang_tokens", 25)
    proprio_dim = cfg.data.dataset.get("proprio_dim", 9)

    # Loss-weight-driven feature toggles. State prediction needs the
    # extra HDF5 fields (image_*_future, proprio_future) and the 3
    # STATE_QUERY tokens in the predictor; both are auto-enabled when
    # cfg.loss.pred_weight > 0 so users only have to set one knob.
    loss_cfg_run = cfg.get("loss", {}) or {}
    pred_weight = float(loss_cfg_run.get("pred_weight", 0.0))
    sigreg_weight = float(loss_cfg_run.get("sigreg_weight", 0.0))
    use_state_prediction = pred_weight > 0
    # Multi-horizon state prediction: one STATE_QUERY token per horizon. The
    # predictor only needs the COUNT (K); the actual offsets live in the data
    # (preprocess --sp-horizons) and the loss weights. K=1 = single-horizon SP.
    sp_horizons = list(loss_cfg_run.get("state_pred_horizons", [20]))
    n_sp_horizons = len(sp_horizons) if use_state_prediction else 1
    if use_state_prediction:
        print(f"[state_pred] horizons={sp_horizons} (K={n_sp_horizons})")

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
        sync_bn = bool(cfg.trainer.get("sync_batchnorm", False))
        # Reject a non-statically-known device count ('auto'/-1): it can silently
        # resolve to >1 GPU with plain per-rank BatchNorm. Require an explicit int.
        if not ddp_devices_known:
            raise ValueError(
                f"projector.norm_type='batch' with cfg.trainer.devices="
                f"{cfg.trainer.get('devices', 'auto')!r}: the device count must be "
                "an explicit int (or list) so the BN/SIGReg multi-GPU safety can "
                "be checked. Set trainer.devices=1 (single-GPU) or =N (multi-GPU "
                "with trainer.sync_batchnorm=true). 'auto'/-1 are rejected."
            )
        # Multi-GPU is allowed IFF sync_batchnorm=true: Lightning then converts
        # every BatchNorm to SyncBatchNorm at setup (works under manual
        # optimization), and our SIGReg is DDP-aware (broadcasts A + all-reduces
        # its moments).
        if multi_gpu and not sync_bn:
            raise ValueError(
                f"projector.norm_type='batch' is unsafe with trainer.devices="
                f"{ddp_world} and sync_batchnorm=False. Plain BatchNorm gives "
                "per-GPU statistics → silent divergence across ranks. Set "
                "trainer.sync_batchnorm=true (Lightning converts BN→SyncBatchNorm; "
                "SIGReg all-reduces its moments) for multi-GPU."
            )

    from libero_dataset import LiberoDataset

    dataset = LiberoDataset(
        hdf5_dir=cfg.data.dataset.hdf5_dir,
        max_action_tokens=max_action_tokens,
        max_lang_tokens=max_lang_tokens,
        img_size=cfg.data.dataset.get("img_size", cfg.img_size),
        use_state_prediction=use_state_prediction,
    )

    # Multi-horizon SP: the data's stored offsets (preprocess --sp-horizons) must
    # match cfg.loss.state_pred_horizons in BOTH count AND order/value — the
    # per-horizon query token k is MSE'd against future frame k with weight k, so
    # a reordering would silently bind each horizon to the wrong target. Counts
    # alone are checked at loss time; assert the full list here, up front.
    if use_state_prediction and dataset.sp_horizons is not None:
        if list(dataset.sp_horizons) != list(sp_horizons):
            raise ValueError(
                f"SP horizon mismatch: data was preprocessed with "
                f"--sp-horizons={dataset.sp_horizons} but cfg.loss."
                f"state_pred_horizons={sp_horizons}. They must match exactly "
                "(count + order + values); re-preprocess or fix the config."
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
        if n_train_demos >= len(unique_demo_keys):
            # 100% training (cfg.train_split >= 1.0): every demo trains, nothing
            # is held out. There is no true val set; instead reuse a small random
            # subset of TRAIN chunks as a "monitoring val". It only drives the
            # periodic _object.ckpt dump cadence (on_validation_end every
            # trainer.val_check_interval steps) and a train-CE signal — the
            # checkpoint is selected by downstream eval rollouts, NOT this CE, so
            # the train/val overlap is intentional and harmless.
            train_indices = list(range(len(demo_keys)))
            perm = torch.randperm(len(train_indices), generator=rnd_gen).tolist()
            n_mon = min(2048, len(train_indices))
            val_indices = [train_indices[i] for i in perm[:n_mon]]
            print(
                f"[Split] train_split={cfg.train_split} >= 1.0 → 100% TRAIN "
                f"({len(train_indices)} chunks, all {len(unique_demo_keys)} demos). "
                f"Monitoring val = {len(val_indices)} train chunks (overlaps train; "
                f"ckpt selected by eval, not val CE)."
            )
        else:
            perm = torch.randperm(len(unique_demo_keys), generator=rnd_gen).tolist()
            train_demo_set = {unique_demo_keys[perm[i]] for i in range(n_train_demos)}

            train_indices = [i for i, k in enumerate(demo_keys) if k in train_demo_set]
            val_indices = [
                i for i, k in enumerate(demo_keys) if k not in train_demo_set
            ]
            print(
                f"[Split] {len(unique_demo_keys)} unique (task, demo) pairs → "
                f"{n_train_demos} train / {len(unique_demo_keys) - n_train_demos} val. "
                f"Train chunks: {len(train_indices)}, val chunks: {len(val_indices)}."
            )

    # Train-only image augmentation. Build a SEPARATE augmenting dataset for the
    # train Subset (identical file glob + index as `dataset`, so train_indices /
    # sampler weights stay valid); val keeps the clean (augment=False) dataset.
    # Disabled in overfit mode (we want exact memorization, no aug noise).
    use_aug = bool(cfg.get("augment", False)) and overfit_demo is None
    if use_aug:
        print("[augment] train-only image augmentation ON (RRC + color jitter)")
        train_dataset = LiberoDataset(
            hdf5_dir=cfg.data.dataset.hdf5_dir,
            max_action_tokens=max_action_tokens,
            max_lang_tokens=max_lang_tokens,
            img_size=cfg.data.dataset.get("img_size", cfg.img_size),
            use_state_prediction=use_state_prediction,
            augment=True,
        )
    else:
        train_dataset = dataset
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
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
        # DDP-correctness: Lightning's default use_distributed_sampler=True would
        # SILENTLY replace this WeightedRandomSampler with a DistributedSampler,
        # destroying the 3-level balancing (no error). So under multi-GPU we (a)
        # build a PER-RANK weighted sampler over a 1/world_size shard and (b)
        # pass use_distributed_sampler=False to the Trainer (set below). Each
        # rank with replacement=True + a rank-distinct seed draws an independent
        # balanced sample; together they cover ~world_size× data per step.
        # ddp_world/ddp_rank are the resolved-topology values from the top of
        # run() (from cfg.trainer.devices, NOT env — see note there).
        if multi_gpu:
            sampler_num = max(1, len(train_weights) // ddp_world)
            sampler_gen = torch.Generator().manual_seed(int(cfg.seed) + ddp_rank)
            print(
                f"[DDP] rank {ddp_rank}/{ddp_world}: per-rank WeightedRandomSampler "
                f"num_samples={sampler_num} "
                "(Trainer use_distributed_sampler=False keeps it unreplaced)."
            )
        else:
            sampler_num = len(train_weights)
            sampler_gen = rnd_gen
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=sampler_num,
            replacement=True,
            generator=sampler_gen,
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

    # Visual encoder. Default (vision_encoder.source=spt_vit, pretrained=false,
    # freeze=false) is bit-identical to the frozen baseline's random-init
    # trainable ViT-Tiny. Set source=hf + model_name_or_path=facebook/dinov2-small
    # + freeze=true to use a frozen DINOv2 backbone instead (hidden_dim then
    # comes from the HF config; the projector adapts hidden_dim -> embed_dim).
    from vision_backbone import build_visual_encoder

    encoder, hidden_dim, freeze_encoder = build_visual_encoder(cfg, spt)
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    # ARPredictor with language + proprio support.
    predictor = ARPredictor(
        embed_dim=embed_dim,
        max_action_tokens=max_action_tokens,
        max_lang_tokens=max_lang_tokens,
        proprio_dim=proprio_dim,
        use_state_prediction=use_state_prediction,
        # One STATE_QUERY token per future horizon (multi-horizon SP).
        state_pred_horizons=n_sp_horizons,
        # Mirror the encoder-side projector's norm choice — paper Section 3
        # says the predictor projector has the "same implementation as the
        # one used for the encoder", so when the encoder projector is BN
        # (sigreg-on path) the state heads must also be BN.
        state_head_norm_type=projector_norm,
        # Multi-token visual prefix. 0 = CLS-only legacy layout, matches
        # spatial-65% / sp_sigreg-67.8% checkpoints exactly.
        visual_pool_grid=visual_pool_grid,
        **cfg.predictor,
    )

    norm_fn = torch.nn.BatchNorm1d if projector_norm == "batch" else torch.nn.LayerNorm
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=norm_fn,
    )

    # T5-small encoder (frozen)
    lang_encoder = T5EncoderModel.from_pretrained("t5-small")
    lang_encoder.eval()
    for p in lang_encoder.parameters():
        p.requires_grad_(False)

    # Language projection: T5 d_model (512) → embed_dim
    lang_proj = torch.nn.Linear(lang_encoder.config.d_model, embed_dim)

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        lang_encoder=lang_encoder,
        lang_proj=lang_proj,
        visual_pool_grid=visual_pool_grid,
        freeze_encoder=freeze_encoder,
        # Recompute encoder activations in backward (memory lever for batch>=32
        # on one GPU). No-op when the encoder is frozen.
        gradient_checkpointing=bool(
            cfg.vision_encoder.get("gradient_checkpointing", False)
        ),
    )

    # Cosine-annealing scheduler with explicit warmup_steps / max_steps.
    # spt's smart-defaults factory pulls these from `trainer.estimated_stepping_batches`,
    # but at scheduler-construction time (inside configure_optimizers) the
    # dataloader hasn't been bound yet and the property can return None — which
    # caused a `missing required arg max_steps` TypeError on the first run.
    # Resolve the budget here:
    #   - If trainer.max_steps is set → use it directly.
    #   - Else → estimate from max_epochs × len(train_loader).
    # Warmup is 4% of total steps, capped at 4K. Raised from 2%/2K alongside the
    # depth 6->10 bump: a deeper + longer-sequence (pool_grid=16) stack has more
    # brittle early attention/residual dynamics, so warmup should scale with it.
    trainer_cfg_sched = cfg.get("trainer", {}) or {}
    cfg_max_steps = int(trainer_cfg_sched.get("max_steps", -1))
    if cfg_max_steps > 0:
        sched_max_steps = cfg_max_steps
    else:
        # Fall back to epoch-budgeted estimate; len(train) is exact at this point.
        steps_per_epoch = max(1, len(train))
        epochs_budget = int(trainer_cfg_sched.get("max_epochs", 1))
        sched_max_steps = max(1, steps_per_epoch * epochs_budget)
    sched_warmup = max(1, min(4000, int(0.04 * sched_max_steps)))
    print(
        f"[scheduler] LinearWarmupCosineAnnealingLR: max_steps={sched_max_steps}, "
        f"warmup_steps={sched_warmup}, interval=step"
    )
    # spt's "optimizer" dict forwards EVERY key to torch.optim.<type>(**kwargs),
    # so it must NOT contain custom keys (encoder_lr would crash AdamW). "modules"
    # is a START-ANCHORED REGEX over named_modules() qualified names; the JEPA is
    # stored as self.model, so all params are prefixed "model." (encoder =
    # model.encoder.*). When the visual encoder is finetuned (freeze_encoder=False)
    # AND a distinct lower encoder_lr is requested, split into TWO optimizer
    # entries (encoder at encoder_lr, everything else at lr). spt uses manual
    # optimization and steps EVERY optimizer on the one joint loss, so this is
    # just two AdamW param groups descending the same backward, each with its own
    # warmup-cosine schedule annealing to its own peak.
    sched_cfg = {
        "type": "LinearWarmupCosineAnnealingLR",
        "max_steps": sched_max_steps,
        "warmup_steps": sched_warmup,
    }
    # Gradient accumulation = per-optimizer step "frequency": spt's manual-opt
    # training_step skips optimizer.step()+zero_grad() until (batch_idx+1)%freq==0
    # (grads accumulate). We set it as the PER-ENTRY "frequency" key, which spt's
    # configure_optimizers reads (it populates _optimizer_frequencies[name] =
    # entry.get("frequency", 1) for EVERY optimizer BEFORE on_train_start). This
    # is the robust path: poking trainer.accumulate_grad_batches_ would be IGNORED
    # for these named optimizers (on_train_start only fills a freq for names NOT
    # already set, and configure_optimizers already set them to 1). With accum>1,
    # both encoder_opt and rest_opt step on the SAME boundary, and global_step
    # ticks once per boundary → max_steps/warmup are in OPTIMIZER steps. The loss
    # is divided by N in lejepa_forward so the effective LR matches a big batch.
    accum = max(1, int(cfg.get("accumulate_grad_batches", 1)))
    opt_base = {k: v for k, v in dict(cfg.optimizer).items() if k != "encoder_lr"}
    encoder_lr = cfg.optimizer.get("encoder_lr", None)
    base_lr = float(opt_base.get("lr", 0.0))
    use_disc_lr = (
        (not freeze_encoder) and encoder_lr is not None and float(encoder_lr) != base_lr
    )
    if use_disc_lr:
        print(
            f"[optimizer] discriminative LR: encoder={float(encoder_lr)} "
            f"(model.encoder.*) / rest={base_lr}. Check the spt param-split log "
            "table on the first run to confirm the encoder/rest split is correct."
        )
        optimizers = {
            "encoder_opt": {
                "modules": r"^model\.encoder($|\.)",
                "optimizer": {**opt_base, "lr": float(encoder_lr)},
                "scheduler": dict(sched_cfg),
                "interval": "step",
                "frequency": accum,
            },
            "rest_opt": {
                "modules": r"^model\.(?!encoder($|\.)).*",
                "optimizer": dict(opt_base),
                "scheduler": dict(sched_cfg),
                "interval": "step",
                "frequency": accum,
            },
        }
    else:
        optimizers = {
            "model_opt": {
                "modules": "model",
                "optimizer": dict(opt_base),
                "scheduler": dict(sched_cfg),
                "interval": "step",
                "frequency": accum,
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

    trainer_kwargs = dict(cfg.trainer)
    # Multi-GPU (DDP): when >1 device, (a) keep our custom weighted sampler
    # unreplaced (use_distributed_sampler=False; we sharded it per-rank above)
    # and (b) use DDP. find_unused_parameters=True is needed because the SP
    # heads / MoT experts can be absent from a given step's autograd graph
    # (training-only heads, zero-weighted terms) → DDP would otherwise error.
    # multi_gpu/ddp_world are the resolved-topology values from the top of run()
    # (from cfg.trainer.devices — same value the BN guard and sampler used).
    if multi_gpu:
        from lightning.pytorch.strategies import DDPStrategy

        trainer_kwargs["use_distributed_sampler"] = False
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
        print(
            f"[DDP] multi-GPU run (devices={ddp_world}): "
            f"use_distributed_sampler=False, sync_batchnorm="
            f"{trainer_kwargs.get('sync_batchnorm', False)}, "
            "DDPStrategy(find_unused_parameters=True)."
        )

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        # 0 (not 1): under DDP the pre-train sanity val runs the custom
        # lejepa_forward (manual SIGReg/CE collectives) interleaved with
        # Lightning's own setup collectives (dataloader-length reduce, etc.),
        # and the two ranks scramble collective order → NCCL desync/hang before
        # the first train step. Skipping the sanity check avoids it; periodic
        # validation is fine (per-task CE log is sync_dist=False — see above).
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=True,
    )

    # Gradient accumulation is wired via the per-optimizer "frequency"=accum keys
    # set in the optim dict above (spt's configure_optimizers reads them). NOTE
    # max_steps/warmup are in OPTIMIZER steps (global_step ticks once per accum
    # boundary), so an accum=N run consumes N× the micro-batches/wall-clock for
    # the same max_steps — the # of weight updates is unchanged.
    if accum > 1:
        eff = int(cfg.loader.batch_size) * accum * max(1, ddp_world)
        print(
            f"[accum] gradient accumulation = {accum} micro-batches/optimizer step "
            f"(effective optimizer batch = {cfg.loader.batch_size} * {accum} * "
            f"{max(1, ddp_world)} = {eff}). Loss divided by N in lejepa_forward; "
            "BN/SIGReg still see one micro-batch per forward (accumulation does "
            "NOT raise their batch). max_steps/warmup are in OPTIMIZER steps."
        )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
