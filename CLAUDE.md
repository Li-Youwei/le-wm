# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## Project Overview

VLA (Vision-Language-Action) on the LeWorldModel (LeWM) codebase. Input: dual-view images + 9d proprio + language instruction. Output: autoregressively predicted FAST action tokens.

**Frozen baseline** = all toggles below off: loss = `L_CE` only (FAST-token cross-entropy); no SP / SIGReg / MoT / V17. Commit `7008f15` is the bit-identical reference for this loss/architecture (the default visual backbone has since changed — see the caveat below).

All toggles below collapse to the frozen-baseline _loss/architecture_ at their default, and any combination is supported in one run. **Default-changed caveat (LIBERO / RynnVLA-002 push — params/compute are NOT a constraint, optimize for eval success rate):** many defaults have moved far off `7008f15`:

- Visual backbone = **DINOv2-base, now FINETUNED** (`vision_encoder.freeze=false`, the #1 LIBERO lever) at a low discriminative `optimizer.encoder_lr=1e-5` while the rest trains at `lr=5e-5` (was a _frozen_ DINOv2-base; before that, random-init trainable ViT-Tiny).
- Predictor **`wm.embed_dim=384`** (was 192 → halves the 768→D projector squeeze, fixes the 5.3× attention over-projection), **depth 12** (was 6), **dropout 0.15** (was 0.2), **`residual_scale_init=true`**.
- **Multi-token visual default `pool_grid=16`** (full 16×16 DINOv2 patch grid) via `run_all4.sh`.
- **Multi-horizon state-prediction**: when SP is on, one STATE_QUERY token PER horizon (`loss.state_pred_horizons=[5,10,15,20]`, near-heavy weights), each read by 3 shared heads — replaces the old single-horizon 3-token (Q_ag/Q_hd/Q_pr) design. **This changed the data format** (`*_future` gained a horizon axis K) → re-preprocess required (see preprocess).
- **Train-only image augmentation** (`augment=true`: RRC + color jitter, per-view-consistent across current+future frames) and optional **receding-horizon eval** (`--exec-steps`).

Reproducing `7008f15` exactly therefore needs: `wm.embed_dim=192 vision_encoder.source=spt_vit vision_encoder.freeze=false predictor.depth=6 predictor.dropout=0.2 predictor.residual_scale_init=false loss.pred_weight=0 augment=false` (and `pool_grid=0`, the bare-`train.py` default; see Input encoders + Hyperparameters).

| Toggle                  | Config knob (`cfg.`)      | Default | Effect when on                                                                                                                                                                                                                                    |
| ----------------------- | ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State-prediction (SP)   | `loss.pred_weight`        | `0.0`   | +`Q_ag/Q_hd/Q_pr` STATE_QUERY tokens + 3 MSE losses vs future visual latents & future proprio                                                                                                                                                     |
| SIGReg anti-collapse    | `loss.sigreg_weight`      | `0.0`   | Algorithm-1 SIGReg on encoder CLS (4 streams w/ SP, 2 without). Forces `projector.norm_type='batch'`                                                                                                                                              |
| Multi-token visual      | `visual_tokens.pool_grid` | `0`\*   | Per-view encoder out → `1 (CLS) + G*G` pooled-patch tokens + per-view view-embedding & 2D pos grid. \*No key in `lewm.yaml` → bare `train.py` is CLS-only; **`run_all4.sh` defaults `POOL_GRID=16`** = full 16×16 patch grid (legacy "V17" = `4`) |
| Mixture-of-Transformers | `predictor.use_mot`       | `false` | Per-modality attn proj (QKV/O+norm) + FFN, partitioned by **true modality** (text/image/state/action[+query]); only SDPA mixing global; predictor ~3.9–4.2× params                                                                                |
| Step-based training     | `trainer.max_steps`       | unset   | Scheduler `interval=step`, `warmup=min(4000, 0.04·max_steps)`; else `max_epochs × len(train_loader)`                                                                                                                                              |
| 4-suite joint           | flat HDF5 dir             | n/a     | `hdf5_dir` → 40 symlinked task `.h5`; 3-level balanced sampler + per-task val CE + `ce_loss_taskbal` selection                                                                                                                                    |

**Inference is identical for all variants** — STATE_QUERY tokens and MoT routing are training-time only. `generate()` decodes FAST from `[lang, z_ag, z_hd, z_pr, BOS]` (+ multi-token visual when on).

**Not done (do not re-add without instruction):** `criterion()` planning cost, `rollout()`, `CEM` search, video co-training (Fast-WAM), multi-frame history (single timestep per view).

## Architecture

### Input encoders

| Input           | Source                            | Encoder                          | Output           | Frozen                          |
| --------------- | --------------------------------- | -------------------------------- | ---------------- | ------------------------------- |
| Agentview img   | `agentview_rgb` 128×128           | DINOv2-base (patch14, finetuned) | `z_agent` (D,)   | **No** (finetuned @ encoder_lr) |
| Eye-in-hand img | `eye_in_hand_rgb` 128×128         | same encoder (shared)            | `z_hand` (D,)    | **No**                          |
| Proprio (9d)    | EE pos(3)+xyzw quat(4)+gripper(2) | MLP→D                            | `z_proprio` (D,) | No                              |
| Language        | task string                       | T5-small                         | `l_1..l_n` (n,D) | **Yes**                         |

`D=384` (predictor/embed dim; was 192). Default visual encoder = **DINOv2-base, FINETUNED** (HF `facebook/dinov2-base`, hidden 768, `vision_encoder.freeze=false`) at the low discriminative `optimizer.encoder_lr=1e-5`; the trainable projector maps 768→`D`. Two views share the encoder. Set `vision_encoder.freeze=true` to keep it frozen; legacy random-init **trainable ViT-Tiny** (hidden=192) via `vision_encoder.source=spt_vit`. T5-small (512d, frozen) projected to `D` via trainable `lang_proj`.

**Resolution:** raw 128×128 must be resized **128→224** (224/14 = 16 patches/side). Never feed 128 to the ViT — not divisible by 14.

### Proprioception (9d, verified)

- **Keys:** `obs/ee_pos` (3, m, base frame) + `robot_states[5:9]` (4, xyzw quat, verified vs `scipy.from_rotvec(obs/ee_ori)` @ 2e-16) + `obs/gripper_states[0:2]` (2, raw fingers).
- `obs/ee_ori` is a 3d axis-angle vector (same format as `action[3:6]`), NOT Euler. We use `robot_states[5:9]` quat for the orientation component for historical consistency.
- Re-normalize quat to unit: `q/(||q||+1e-8)` (defensive).
- **Gripper:** use both raw dims as-is. **Never** `mean`/`abs`/reduce — fingers are symmetric around 0, so `mean` collapses to ~0 (no info).
- `normalize_proprio(raw)` is the single source of truth (re-normalizes quat at `[3:7]`, passes rest through); used by both `preprocess_libero.py` and `eval_libero.py::preprocess_obs`.

### Transformer sequence

Training (frozen baseline, CLS-only, no SP):

```
[l_1..l_n, z_agent, z_hand, z_proprio, BOS, T_1..T_k, PAD...]
 ←─ perception prefix (bidirectional) ─→  ←─ action (causal) + pad ─→
```

- **V17:** each `z_agent`/`z_hand` slot expands to `1+G*G` prefix positions (each with its own 1D pos enc).
- **SP (multi-horizon):** one STATE_QUERY token per horizon (`Q_t+5, Q_t+10, Q_t+15, Q_t+20`) appended at the very end; each is read by the 3 **shared** heads (ag/hd/pr) to predict that horizon's future agent-image + hand-image + proprio. `loss.state_pred_horizons` sets the offsets, `state_pred_horizon_weights` the near-heavy weights. K=1 = single-horizon.

Inference:

```
[l_1..l_n, z_agent, z_hand, z_proprio, BOS] → AR-generate T_1..T_k,<EOS>
→ FAST decode → chunk (H=20 × 7 dims, anchor-relative) → closed-loop execution
```

STATE_QUERY not appended at inference; multi-token visual preserved as in training.

**Decoding rules (exact):**

1. Stop on `<EOS>` (1025) or after `max_action_tokens`, whichever first.
2. Before FAST decode: strip leading `<BOS>`, truncate at `<EOS>` (drop it + after), drop tokens beyond `max_action_tokens`.
3. Only IDs 0..1023 go to the FAST decoder — never BOS/EOS/PAD.
4. `action_head` outputs 1026 logits (0..1025) so PAD (1026) can't be sampled — no extra PAD masking.
5. `generate()` forces `logits[:,BOS]=-inf` every step (BOS is start-only).

`z_agent`/`z_hand` = ViT CLS of each view; `z_proprio` = MLP of 9d. `T_i` variable-length, observed max 75 across 4 suites; `max_action_tokens=80`, preprocessing asserts `max(fast_length) ≤ 80`.

### Attention mask (training)

Prefix-bidirectional + action-causal hybrid:

```
        lang z_ag z_hd z_pr BOS T_i PAD
lang  [  Y    Y    Y    Y   -   -   - ]
z_*   [  Y    Y    Y    Y   -   -   - ]
BOS   [  Y    Y    Y    Y   Y   -   - ]
T_i   [  Y    Y    Y    Y   Y  csl  - ]
PAD   [  -    -    -    -   -   -   - ]
```

- **Prefix** (lang+visual+proprio): bidirectional among themselves; can't see action or PAD.
- **Action** (BOS+T_i): see all real prefix, causal within group (csl), can't see PAD.
- **PAD** (action & language): attend to nothing; nothing attends to them. Use `lang_lengths` for real language positions.
- **SP queries:** each `Q_x` sees all real prefix + entire real action zone + itself; NOT PAD, NOT any other `Q_y` (prediction independence); nothing outside the query block attends to `Q_*`.

Build an explicit `attn_mask=(B,1,L,L)` bool — NOT `is_causal=True` (pure-causal only). Teacher forcing within action zone (target shifted +1).

### Targets & loss

`action_logits` come **only from action-zone positions** (BOS+T_i+PAD); prefix positions don't go through `action_head`.

```
action_logits: (B, 1+max_action_tokens, 1026)   # BOS + T_1..T_k + PAD
targets:       (B, 1+max_action_tokens)
  BOS→T_1, T_i→T_{i+1}, T_k→<EOS>(1025), PAD→-100 (ignored)
```

```python
L_CE = F.cross_entropy(logits.reshape(-1, 1026), targets.reshape(-1),
                       ignore_index=-100, label_smoothing=0.1)  # cfg.label_smoothing
```

Label smoothing 0.1 puts a CE floor ≈ 1.02 on a 1026-way vocab — read curves against that, not 0. `token_accuracy` (logged in `train.py`) is the cleaner diagnostic vs smoothed/non-smoothed.

### Special tokens

Vocab 0..1023 = FAST; 1024=`<BOS>`, 1025=`<EOS>`, 1026=`<PAD>`. `nn.Embedding`=1027; `action_head`=1026 (PAD never a target).

### Temporal indexing (CRITICAL)

Raw-step indices (not chunk-level). Sliding window **stride=1** (standard VLA): consecutive samples share H-1 actions but get a fresh obs anchor.

- `H=20` raw steps (1s @ 20Hz). `t` = raw step (valid anchors 0..T-H-1).
- Obs `o_t` = frame at step `t` (not `t*H`). Chunk at `t` needs obs `[t..t+H]` (H+1 frames) and actions `[t..t+H-1]`.
- Per-demo samples (stride=1): `max(0, T-H)`.

### Anchor-relative action chunk

Element `k` (k=0..H-1) = cumulative displacement from anchor `(p_t, R_t)`, not a step-to-step delta (removes the implicit world model the model would otherwise learn):

```
chunk[k,0:3] = ee_pos[t+k+1] - ee_pos[t]            # base frame, m
chunk[k,3:6] = rotvec(R_{t+k+1} · R_t^{-1})         # base frame, axis-angle rad
               R_x = Rotation.from_rotvec(obs/ee_ori[x])
chunk[k,6]   = actions[t+k, 6]                      # gripper command, unchanged
```

Computed from **observed HDF5 poses** (OSC tracking is imperfect — observed poses are the true ground truth), not by accumulating commands. Stored physical (m/rad/cmd), then 1st/99th-pct normalized to [-1,1] before FAST. Dim 6 is NOT anchor-relative (raw command, already [-1,1]).

### Action format (verified — re-verify before changing)

- `action[0:3]`: pos delta, base frame, [-1,1], ×`pos_scale=0.05m`.
- `action[3:6]`: axis-angle rotation delta (NOT Euler), base frame, [-1,1], ×`rot_scale=0.5rad`. Same rep as `obs/ee_ori`.
- `action[6]`: gripper command [-1,1], direct (not a delta, not OSC-scaled).
- `robot_states[5:9]`: unit xyzw quat. `obs/ee_ori`: 3d axis-angle, continuous on libero_spatial (no ±π wrap).

### Closed-loop execution (inference)

Chunk is anchor-relative; execute step-by-step so the OSC controller self-corrects drift:

```python
anchor_pos, anchor_quat = obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat"].copy()  # xyzw
R_anchor = Rotation.from_quat(anchor_quat)
for k in range(H):
    target_pos = anchor_pos + chunk[k, 0:3]
    target_R   = Rotation.from_rotvec(chunk[k, 3:6]) * R_anchor
    cur_pos, cur_R = obs["robot0_eef_pos"], Rotation.from_quat(obs["robot0_eef_quat"])
    action_input = np.concatenate([
        np.clip((target_pos - cur_pos) / POS_SCALE, -1, 1),
        np.clip((target_R * cur_R.inv()).as_rotvec() / ROT_SCALE, -1, 1),
        [chunk[k, 6]]])
    obs, r, done, _ = env.step(action_input)
    if done: break
```

(`np.clip` is cosmetic — robosuite's `scale_action` clips internally.)

### Robosuite OSC constants (verified via `inspect.getsource`)

`pos_scale=0.05m`, `rot_scale=0.5rad`, `input_max/min=±1`, `control_delta=True` (deltas from current EE pose at `env.step`), `uncouple_pos_ori=True`. LIBERO forwards `OSC_POSE` unchanged (no override, `env_wrapper.py:47`). `eval_libero.py` reads `POS_SCALE`/`ROT_SCALE` at runtime from `env.env.robots[0].controller.output_max[0]`/`[3]` with uniformity asserts.

### Multi-token visual prefix (V17)

`pool_grid=G>0` → `JEPA._pool_visual_tokens` turns encoder `(B,1+P,D)` into `cat([CLS, adaptive_avg_pool2d(patches→(B,D,side,side),(G,G))→(B,G*G,D)])` = `(B,1+G*G,D)`. With 224/14, side=16, P=256 (raises `ValueError` if P not a perfect square → rules out DINOv2-**with-registers**; the default plain `facebook/dinov2-base` has no register tokens → 256 patches, V17-compatible). **G=16 (the default) → `n_visual_per_view=257` = the full DINOv2 patch grid (`adaptive_avg_pool2d(16→16)` is identity, i.e. no pooling at all)**; G=4 → `n_visual_per_view=17` (legacy "V17"). Both views use the same pooler.

In `ARPredictor.forward` each visual token gets: type-emb[1]; view-emb[0/1] (only if `nv>1`); for patch (r,c) a per-view 2D pos `agent/hand_patch_2d_pos[r,c]` (never shared); plus a 1D pos. CLS gets view-emb + 1D pos only. `pos_embedding` size = `max_lang + 2·nv + 1 + 1 + max_action_tokens + n_state_query` (pool_grid=16 +SP = 624; pool_grid=4/V17 +SP = 144; V17 no-SP = 141). All `1+G*G` tokens are prefix (bidirectional); action zone sees the full real prefix, causal within itself.

**Projector reshape** (`jepa.py`): `encode()` and `encode_future_visual()` both reshape `(B,N,D)→(B*N,D)→projector→(B,N,D)` so source & target feed the BN1d projector the same distribution (matters under V17, where N>1).

### Mixture-of-Transformers (MoT)

`use_mot=true` = full Meta MoT: each `Block` routes **both** attention (`Attention`→`MoTAttention`: per-modality pre-norm + QKV + output proj) **and** FFN (`FeedForward`→`MoTFeedForward`) per modality; only the scaled-dot-product attention (token mixing) stays global, so modalities still attend to each other. Partition is by **true modality** (Meta MoT "decouple by modality", _not_ by sequence-role), assigned by absolute position in `_build_modality_ids`: M=0 text (lang), M=1 image (both visual views), M=2 state (proprio), M=3 action (BOS + tokens + PAD), M=4 state-query (SP only); `n_modalities` = 5 w/ SP, 4 without. Mirrors RynnVLA-002's four input modalities (image/text/state/action). Both `MoT*` modules run every expert on the full sequence + merge by modality mask (~M× the projection/FFN FLOPs, attention op stays 1×; chosen for code simplicity over gather/scatter). All forward paths build their own `_build_modality_ids (B,L)`. `use_mot=false` → original shared `Attention` + MLP, `modality_ids=None`, **bit-identical to `7008f15`**. Predictor params at the current defaults (D=384, depth 12, heads 16, MoT+SP, `pool_grid=16`, 4 SP horizons): **~597M** total — but the two visual SP heads (`MLP(D→2048→D·nv)`, nv=257) are **~407M of that and are training-only** (discarded at inference), so the **inference-relevant predictor is ~190M**. SP-head cost scales with both D and nv; lower `pool_grid` or set `loss.pred_weight=0` to shrink it. (Plus the now-trainable DINOv2-base ~86M finetuned at `encoder_lr`.)

**Checkpoint loading:** `eval_libero.py::load_checkpoint` rejects a `_weights.ckpt` if it sees `state_pred_head_*`/`state_query_embeddings` or `projector.net.1.running_mean` (BN) keys — those must load from the `_object.ckpt` (pickled JEPA from `ModelObjectCallBack`). `build_model()` (the `_weights.ckpt` path) now reconstructs the **default DINOv2-base at D=384, depth 12** via `vision_backbone.build_visual_encoder`, so a `_weights.ckpt` trained with a _different_ backbone/shape (e.g. legacy `spt_vit` ViT-Tiny, or a finetuned encoder, or SP/MoT) won't match — eval it via `_object.ckpt` instead.

### State-prediction + SIGReg (SP)

`use_state_prediction` auto-set by `pred_weight>0`. Three learnable queries appended at sequence end (`pos_embedding` grows by 3). Heads read the post-transformer hidden at each `Q_x`; head norm type mirrors `cfg.projector.norm_type` (per LeWM §3, predictor projector = encoder projector):

| Head                 | Output                          | Target                        |
| -------------------- | ------------------------------- | ----------------------------- |
| `state_pred_head_ag` | `MLP(D→2048→D·nv)` → `(B,nv,D)` | `z_agent_future` `(B,nv,D)`   |
| `state_pred_head_hd` | same                            | `z_hand_future` `(B,nv,D)`    |
| `state_pred_head_pr` | `MLP(D→2048→9)`                 | re-normalized `proprio_{t+H}` |

`nv=n_visual_per_view` (1 CLS-only, 17 under V17 `pool_grid=4`, 257 under the default `pool_grid=16`). **Multi-horizon:** `state_pred_horizons` query tokens (default 4: t+5/10/15/20), each read by the SAME 3 heads (`state_pred_head_ag/hd/pr`) — so the head COUNT does not grow with horizons (only +K query embeddings + K pos slots + K future encodes). Each visual head outputs `D·nv` (input D), so its size scales with both: at the default **D=384, nv=257 each visual head is ~203M → ~407M for ag+hd**, all training-only (discarded at inference). `state_pred_head_ag/hd/pr` are applied per horizon (`forward` reshapes to `(B,K,nv,D)`/`(B,K,9)`). Shrink via lower `pool_grid` / smaller D / `loss.pred_weight=0`. **SIGReg** now stacks the 2 current-view CLS + all 2K future-horizon CLS (matters since the encoder is trainable). Loss = near-heavy horizon-weighted sum, normalized to sum 1 so `pred_weight` keeps its single-horizon scale.

```
L_pred  = MSE(pred_ag, z_ag_future) + MSE(pred_hd, z_hd_future) + MSE(pred_pr, proprio_future)  # per-token for visual
L_total = L_CE + pred_weight·L_pred + sigreg_weight·L_sigreg
```

- Targets from `JEPA.encode_future_visual` (shared ViT + projector, **no `.detach()`** per LeWM §3 — no stop-grad/EMA; SIGReg prevents collapse). `train.py` normalizes both sides to 3D so nv=1 == legacy `(B,D)` MSE.
- **SIGReg** (LeWM Algorithm 1): on the **encoder CLS slice only** — stack `[z_ag, z_hd, z_ag_future, z_hd_future][:,0]` = `(T=4,B,D)` (degenerates to T=2 current-only when `pred_weight=0`). Predictor outputs intentionally excluded (anchored indirectly via `L_pred`). Patch tokens are constrained by per-token SP MSE, not SIGReg.
- **Patch SP gradient:** SP heads output `D·nv` so every pooled patch token (not just CLS) gets direct SP gradient; collapses to CLS-only — bit-identical to the legacy arch — at nv=1.
- **Defaults (paper):** `pred_weight=1.0`, `sigreg_weight=0.1`; baseline both 0. `sigreg.kwargs`: `knots=17, num_proj=1024` (barely sensitive, leave alone).

**BatchNorm projector caveats** (forced when `sigreg_weight>0`, per LeWM §3 — LayerNorm blocks anti-collapse):

1. Train/eval stat mismatch — BN uses running stats at eval; OOD/single-traj eval can drift (LayerNorm baseline is per-sample, immune).
2. Needs batch ≥ 32 **per forward** (SIGReg moments + BN stats). **`overfit.yaml` defaults batch=8 — too small for SP+SIGReg; override `loader.batch_size=32`.** Don't go below 32. **Gradient accumulation does NOT help this** — BN/SIGReg compute stats per forward pass, so accumulation only grows the optimizer batch, not the per-forward statistical batch (see Batch / memory / multi-GPU).
3. Multi-GPU: set `trainer.sync_batchnorm=true` → Lightning converts BN→`SyncBatchNorm` at setup (works under spt manual optimization), and `module.py::SIGReg` is **DDP-aware** (all-reduces its cos/sin moments + **broadcasts rank-0's projection matrix A** so all ranks measure the same slices → global-batch normality test). The `train.py` guard allows `devices>1` **iff** `sync_batchnorm=true`, and requires `devices` to be an explicit int/list — `auto`/-1 are rejected (the world size must be known at sampler-build time; see DDP note). DDP world size is derived from `cfg.trainer.devices`, NOT env `WORLD_SIZE` (unset in the rank-0 parent when `run()` builds the sampler).

### Batch / memory / multi-GPU (finetuned-encoder retrain)

The finetuned DINOv2 + D384/depth12 model OOMs above ~16/GPU, but BN/SIGReg need ≥32 **per forward**. Three levers (`accumulate_grad_batches` only grows the optimizer batch, NOT the per-forward batch):

| GPUs | recipe                                                       | per-forward batch (BN/SIGReg) | effective optimizer batch |
| ---- | ------------------------------------------------------------ | ----------------------------- | ------------------------- |
| 1    | `GRAD_CKPT=true` raises micro-batch to ≥32; `ACCUM` for more | `BATCH_SIZE` (≥32 via ckpt)   | `BATCH_SIZE·ACCUM`        |
| 2    | `NUM_GPU=2 SYNC_BN=true` (auto); `ACCUM×2` → 128             | `BATCH_SIZE·2`                | `BATCH_SIZE·2·ACCUM`      |
| 4    | `NUM_GPU=4` → 64 directly                                    | `BATCH_SIZE·4`                | `BATCH_SIZE·4·ACCUM`      |

- **`vision_encoder.gradient_checkpointing=true`** (default): recomputes encoder activations in backward (~30% slower, ~½ encoder activation memory) — the single-GPU lever to fit batch≥32. No-op when `freeze=true`. Uses HF `gradient_checkpointing_enable(use_reentrant=False)` (DINOv2 supports it; legacy `spt_vit` warns + runs without).
- **`accumulate_grad_batches=N`**: spt's manual-opt `training_step` accumulates via per-optimizer `frequency` (skips step+zero*grad until the boundary). Lightning's `Trainer(accumulate_grad_batches=)` is ignored under manual opt, so `train.py` sets `trainer.accumulate_grad_batches*`(the attr spt reads first) and **divides the loss by N** (spt does not).`max_steps`/warmup are in OPTIMIZER steps (post-accum) → N× wall-clock, same # of weight updates.
- **DDP sampler fix**: Lightning's default `use_distributed_sampler=True` would silently replace the 3-level `WeightedRandomSampler` with a `DistributedSampler` (balancing lost, no error). `train.py` builds a **per-rank** weighted sampler (rank/world from `WORLD_SIZE`/`LOCAL_RANK` env, `num_samples=len//world`, rank-distinct seed) and sets `use_distributed_sampler=False` + `DDPStrategy(find_unused_parameters=true)` (SP heads / MoT experts can be absent from a step's graph). **Multi-GPU paths are untested locally — smoke-test on the server.**
- `run_all4.sh` knobs: `NUM_GPU` / `ACCUM` / `GRAD_CKPT` / `SYNC_BN` (auto-on when `NUM_GPU>1`). `BATCH_SIZE` is the **per-GPU micro-batch**.

**Known concern (not fixed):** quaternion sign ambiguity — `MSE(q,-q)=4` though same rotation. If `loss_pred_pr` spikes, add hemisphere canonicalization (`if q_w<0: q=-q` on `[3:7]`) to `normalize_proprio` and re-preprocess — but keep the frozen baseline ckpt off the canonicalized data (eval-drift risk).

### 4-suite joint training

Primary mode: joint over all 4 LIBERO suites (spatial+object+goal+10 = 40 tasks). Pure config: point `hdf5_dir` at a flat dir of 40 symlinked `.h5`; `LiberoDataset.__init__` globs non-recursively (`*.h5`), so the flat dir is required.

- Emits `task_id` (stable, sorted-file order) + standard fields (+ `lang_*` if language, + `*_future` if SP).
- **3-level balanced sampler:** `w_i = 1/(n_tasks · n_demos[task] · n_chunks[(task,demo)])` → `WeightedRandomSampler(replacement=True)`; uniform across task / demo / time.
- **Per-task val CE:** `lejepa_forward` logs `validate/ce_loss/task_{t}` (validate only); `TaskBalancedCEMetric` (utils.py) aggregates → `validate/ce_loss_taskbal` — the **ckpt-selection metric**. `pick_best_ckpt.py` reads it from TB events for the top-K `_object.ckpt`.
- `ModelObjectCallBack` keys its top-K heap on `ce_loss_taskbal` (else `validate/ce_loss_epoch`): epoch mode dumps each epoch; step mode (when `step_interval` set) dumps each val pass + maintains a `lewm_latest_object.ckpt` symlink (refreshed on eviction, never dangles).

## Files

| File                            | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `jepa.py`                       | `JEPA`: `encode`, `predict`, `encode_future_visual`, `predict_actions`, `_pool_visual_tokens`. `train()` pins `lang_encoder` (and a frozen encoder) in `.eval()`. Encoder built by `vision_backbone.build_visual_encoder`; `freeze_encoder` keeps a frozen backbone out of train mode + optimizer.                                                                                                                                                                                           |
| `vision_backbone.py`            | `build_visual_encoder(cfg, spt)` → `(encoder, hidden_dim, freeze_encoder)`. `vision_encoder.source=hf` (DEFAULT) = `AutoModel.from_pretrained` DINOv2-base wrapped in `HFVisionBackbone`, **finetuned by default** (`freeze=false`; `freeze=true` to freeze); `source=spt_vit` = legacy random-init ViT-Tiny via `vit_hf`. Projector adapts hidden_dim→`embed_dim`.                                                                                                                          |
| `module.py`                     | `ARPredictor`, `Block`, `Attention`, `MoTAttention`, `MLP`, `FeedForward`, `MoTFeedForward`, `SIGReg`; vocab consts (BOS 1024 / EOS 1025 / PAD 1026 / TOTAL 1027 / HEAD 1026). All toggles default off.                                                                                                                                                                                                                                                                                      |
| `train.py`                      | Hydra+Lightning+TB. `lejepa_forward` = L_CE (+ optional L_pred / L_sigreg). Demo-level train/val split via `(fpath,demo_idx)` keys (no chunk leakage). `seed_everything(cfg.seed)` + `seed=cfg.seed` to `spt.Manager`. Builds encoder via `build_visual_encoder`.                                                                                                                                                                                                                            |
| `libero_dataset.py`             | Dual-view + 9d proprio + T5 + FAST. ImageNet-normalized 224×224 (`_preprocess_image`). Lazy per-worker HDF5. Stable `task_id`, `get_sampler_weights()`.                                                                                                                                                                                                                                                                                                                                      |
| `preprocess_libero.py`          | Anchor-relative chunks + stride=1 + FAST BPE. `normalize_proprio` single source (reused by eval). Always stores `image_*_future`+`proprio_future` (no toggle).                                                                                                                                                                                                                                                                                                                               |
| `eval_libero.py`                | Closed-loop exec. Runtime OSC scales. Per-suite eval horizon `LIBERO_MAX_STEPS` (spatial 220 / object 280 / goal 300 / 10 520 / 90 400) auto-by-`--suite`; 50 rollouts/task; `--seed` default 3072; optional receding-horizon `--exec-steps N`. `build_model()` (the `_weights.ckpt` path) reconstructs DINOv2-base at D=384/depth 12; use `_object.ckpt` for all toggles / finetuned-or-non-default backbones / SP.                                                                         |
| `fast_utils.py`                 | `fast_decode` with pad/truncate fallback (built-in decoder zeros on length mismatch → freezes the robot) + `denormalize_actions`.                                                                                                                                                                                                                                                                                                                                                            |
| `utils.py`                      | `ModelObjectCallBack` (top-K + latest-symlink), `TaskBalancedCEMetric`, `EarlyProbeCallback` (online probe via `quick_probe_eval.py`), `PeriodicPrintCallback`.                                                                                                                                                                                                                                                                                                                              |
| `pick_best_ckpt.py`             | Top-K `_object.ckpt` by `ce_loss_taskbal` (fallback `ce_loss_epoch`) from TB events.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `quick_probe_eval.py`           | In-training probe: 10 tasks × `num_episodes` × 4 suites = 40 rollouts; reuses eval loader/rollout.                                                                                                                                                                                                                                                                                                                                                                                           |
| `fit_tokenizer_all4.py`         | One FAST tokenizer over all 40 tasks.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `smoke_test.py`                 | Full-pipeline forward/backward/generate + VRAM at batch 128/64/32/16.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `test_attn_mask.py`             | Unit check on `_build_attn_mask` (not V17/MoT).                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `check_fast_roundtrip.py`       | FAST encode→decode on stored GT chunks; per-dim normalized + physical L1.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `config/train/lewm.yaml`        | Primary config: `wm.embed_dim=384`, predictor depth 12 / dropout 0.15 / `residual_scale_init=true`; `vision_encoder.freeze=false` (finetuned DINOv2-base) + `optimizer.encoder_lr=1e-5`; `augment=true`; multi-horizon SP via `loss.state_pred_horizons` + `state_pred_horizon_weights` (SP/SIGReg/MoT still gated by `loss.pred_weight`/`sigreg_weight`/`predictor.use_mot`). Multi-token visual via CLI `+visual_tokens.pool_grid=16` (no default key; `run_all4.sh` sets `POOL_GRID=16`). |
| `config/train/overfit.yaml`     | 1-demo sanity (batch=8, lr=1e-3, no reg, max_epochs=2000); also pins the frozen DINOv2-base backbone.                                                                                                                                                                                                                                                                                                                                                                                        |
| `config/train/data/libero.yaml` | Dataset config + proprio layout.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `run_all4.sh`                   | 4-suite driver: train→pick best→per-suite eval (50 rollouts). Env knobs: `ARM`/`SEED`/`MAX_STEPS`/`VAL_INTERVAL`/`BATCH_SIZE`(per-GPU)/`NUM_WORKERS`/`USE_MOT`/`POOL_GRID`/`VISION_{SOURCE,MODEL,FREEZE,LOCAL_ONLY}`/`NUM_GPU`/`ACCUM`/`GRAD_CKPT`/`SYNC_BN`(auto when NUM_GPU>1). `launch_all4.sh` = tmux wrapper.                                                                                                                                                                          |
| `preprocess_all4.sh`            | Preprocess all 40 raw `.hdf5` → `libero_processed_v5/libero_{spatial,object,goal,10}/`.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `requirements.txt`              | `transformers>=4.48,<5` critical (earlier misses `TimmWrapperModel`; v5 breaks FAST).                                                                                                                                                                                                                                                                                                                                                                                                        |

Upstream LeWM `eval.py` (CEM/Adam planning) + `config/eval/` removed (`d3b172a`) — VLA uses only `eval_libero.py`.

## Hyperparameters (defaults)

| Param                                            | Value                                                                                                                                                                                                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunk `H` / stride                               | 20 raw steps (1s @ 20Hz) / 1                                                                                                                                                                                                                          |
| FAST vocab / `max_action_tokens`                 | 1024 / 80 (observed max 75)                                                                                                                                                                                                                           |
| `max_lang_tokens`                                | 25                                                                                                                                                                                                                                                    |
| `D` / predictor (depth / heads / dim_head / mlp) | 384 / 12 / 16 / 64 / 2048 (embed_dim 192→384, depth 6→12; inner attn = heads·dim_head = 1024 fixed, so D=384 → 2.67× over-projection); `predictor.residual_scale_init=true` = 1/√(2·depth) residual-output scaling (`false` → bit-identical baseline) |
| dropout / emb_dropout                            | predictor 0.15 / 0.0 (was 0.2)                                                                                                                                                                                                                        |
| proprio dim                                      | 9 (`ee_pos3 + xyzw_quat4 + gripper_raw2`)                                                                                                                                                                                                             |
| vision backbone                                  | **finetuned DINOv2-base** (`facebook/dinov2-base`, hidden 768, `vision_encoder.freeze=false`) default → projector 768→`D`; `freeze=true` to freeze; legacy ViT-Tiny via `source=spt_vit`                                                              |
| projector                                        | `MLP(hidden→2048→D)` LayerNorm (BatchNorm forced when SIGReg on)                                                                                                                                                                                      |
| T5                                               | `t5-small` frozen, 512d → `lang_proj`                                                                                                                                                                                                                 |
| optimizer                                        | AdamW lr=5e-5 wd=0.05; `LinearWarmupCosineAnnealingLR` interval=step                                                                                                                                                                                  |
| batch / precision / grad-clip                    | 128 / bf16 / 1.0                                                                                                                                                                                                                                      |
| label smoothing                                  | 0.1 (CE floor ≈ 1.02)                                                                                                                                                                                                                                 |
| train/val split                                  | 0.9 / 0.1 **demo-level** via `(fpath,demo_idx)` keys                                                                                                                                                                                                  |
| max epochs                                       | 100 (epoch mode). Step mode (4-suite): `max_steps=100000`, `max_epochs=999`, `val_check_interval=4000`, `warmup=4000`, `+check_val_every_n_epoch=null`                                                                                                |
| seed                                             | `cfg.seed=3072` (train); `eval_libero.py --seed` default also 3072 (was 42)                                                                                                                                                                           |

**Ckpt selection:** lowest `validate/ce_loss_taskbal` (4-suite) / `validate/ce_loss_epoch` (single-suite). **Never the last step/epoch** — the baseline overfits before the budget.

## Commands

```bash
conda activate vla                         # miniforge, py3.10 (GPU server; paths in run_all4.sh are server-specific)
pip install -r requirements.txt

# Train (default backbone is now a FINETUNED DINOv2-base, freeze=false). FIRST RUN must populate
# the HF cache for facebook/dinov2-base: either `huggingface-cli download
# facebook/dinov2-base`, or pass vision_encoder.local_files_only=false once
# (with HF_HUB_OFFLINE unset). Subsequent runs hit the cache offline.
python train.py data=libero

# Reproduce the exact 7008f15 frozen baseline (random-init trainable ViT-Tiny):
python train.py data=libero vision_encoder.source=spt_vit vision_encoder.freeze=false \
    wm.embed_dim=192 predictor.depth=6 predictor.dropout=0.2 \
    predictor.residual_scale_init=false loss.pred_weight=0 augment=false

# Toggles (any combination; SIGReg requires projector.norm_type=batch)
python train.py data=libero loss.pred_weight=1.0 loss.sigreg_weight=0.1 projector.norm_type=batch
python train.py data=libero +visual_tokens.pool_grid=16  # 1 CLS + 16x16 full patch grid (4 = legacy V17)
python train.py data=libero predictor.use_mot=true
python train.py data=libero label_smoothing=0 predictor.dropout=0 optimizer.weight_decay=0

# 1-demo sanity (SP needs loader.batch_size=32 — overfit default 8 too small for BN)
python train.py --config-name=overfit data.dataset.hdf5_dir=/path/single_task/ subdir=overfit_sanity

# 4-suite joint (point hdf5_dir at the flat 40-task dir)
bash run_all4.sh                           # ARM=baseline|sp_sigreg, SEED=...

# Eval — _object.ckpt for DINOv2 / SP / SIGReg / MoT / V17 (the default path).
# --max-steps auto-selected per --suite (220/280/300/520/400); 50 rollouts; seed 3072.
python eval_libero.py --checkpoint /path/lewm_step_{N}_object.ckpt \
    --tokenizer /path/fast_tokenizer --processed-dir /path/libero_processed/<suite>/ --suite libero_spatial
```

**Validation scripts (no pytest/CI):** `smoke_test.py`, `test_attn_mask.py`, `check_fast_roundtrip.py`.

**Local sanity without GPU/data:** runtime checks need the `vla` conda env — bare `python3`/`python` on PATH lacks torch. Fastest predictor check: build `ARPredictor(..., depth=2)` + dummy tensors, assert `forward`/`generate` return arity (baseline → tensor, SP → 4-tuple) and `sum(p.numel())` param counts on CPU (no ViT/T5/HDF5). `test_attn_mask.py` runs standalone too.

`overfit_demo=N`: train==val on one demo; a healthy pipeline drives `token_accuracy>0.95` within a few hundred epochs (else the bug is in data→forward→loss→backward, not capacity). `ce_loss` plateaus at the label-smoothing floor ≈1.02 (not 0) — read `token_accuracy`, not raw CE.

## External libraries

- **`stable-worldmodel` (swm):** env wrappers, `swm.data.utils.get_cache_dir()`.
- **`stable-pretraining` (spt):** `spt.Module`/`Manager`, `spt.data.DataModule`, `spt.backbone.utils.vit_hf`.
- **`transformers`:** `AutoProcessor` (FAST, `trust_remote_code=True`), `T5EncoderModel` + `T5Tokenizer`.
- **`lightning.pytorch.loggers.TensorBoardLogger`** (view: `tensorboard --logdir <run_dir>/tb_logs`).

## Key details

- **Data:** preprocessed HDF5 is one dir per suite — the **dir** (not individual files) is what `LiberoDataset` consumes; 4-suite training points `hdf5_dir` at a flat dir of 40 symlinked task `.h5`.
- **Raw LIBERO source (read-only — never write here):** `/data/lyw/libero_{spatial,object,goal,10}/*.hdf5` (`RAW_ROOT`/`--raw-root`, default `/data/lyw`). `preprocess_all4.sh` writes to a separate `OUT_ROOT` (default `/data/lyw/libero_processed_v5`) and aborts if `OUT_ROOT==RAW_ROOT` (would pollute the raw suite dirs). All run-time data/ckpt/tokenizer paths now live under `/data/lyw`; conda env `vla` at `/data/lyw/miniconda3`.
- **FAST tokens:** vlen int32 per sample; each HDF5 stores `action_low`/`action_high` (inverse-norm), `chunk_size`, `chunk_stride`, `language_instruction` attrs. Unified 4-suite tokenizer loaded via `--load-tokenizer`.
- **Device:** no hardcoded `cuda`; inferred from inputs (caller moves `JEPA` to device).
- **Checkpoints:** `lewm_weights.ckpt` (Lightning state*dict, strips `model.`, skips `lang_encoder.*`; rejected for SP/BN; `build_model()` rebuilds the DINOv2-base backbone at D=384/depth 12).`lewm\*{step,epoch}\_{N}\_object.ckpt` (`torch.save(model)`, load `weights_only=False`); step mode keeps the `lewm_latest_object.ckpt` symlink.
- **Seed:** enforced 3 places — `seed_everything(cfg.seed, workers=True)`, `seed=cfg.seed` to `spt.Manager`, `Generator().manual_seed(cfg.seed)` for split + sampler. `eval_libero.py --seed` defaults to 3072 too (was 42), matching train. Without all three, `spt.Manager` falls back to seed 0 and init drifts.
- **Vision backbone / HF cache:** default `facebook/dinov2-base` is fetched via `AutoModel.from_pretrained` and must be in the HF cache before an offline run (`run_all4.sh` forces `HF_HUB_OFFLINE=1` only when `VISION_LOCAL_ONLY=true`). First time: `huggingface-cli download facebook/dinov2-base`, or `VISION_LOCAL_ONLY=false bash run_all4.sh`. Inputs are already ImageNet-normalized 224×224 (`_preprocess_image`), so DINOv2 needs no preprocessing change. Frozen-encoder caveat: SIGReg then regularizes the trainable projector output, not the (frozen) encoder features.
