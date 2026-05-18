# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLA (Vision-Language-Action) on top of the LeWorldModel (LeWM) codebase.
The model takes dual-view images, proprioceptive state, and a language
instruction as input, and autoregressively predicts FAST action tokens.

The frozen baseline (`pred_weight = sigreg_weight = 0` in
`config/train/lewm.yaml`) is the reproducibility target locked at commit
`7008f15`: loss is `L_CE` only and the model is joint-trained on the 10
LIBERO-Spatial tasks into a single checkpoint that disambiguates tasks
via the language instruction. It achieved 65 % average success across
the 10 spatial tasks at evaluation.

On top of that frozen point several additive toggles have been wired in.
All of them collapse to the frozen baseline when their controlling
config knob is at its default value, and any combination of the toggles
is supported in one training run:

| Toggle | Config knob (under `cfg.`) | Default | When > / true |
|--------|----------------------------|---------|---------------|
| State-prediction (SP heads + L_pred) | `loss.pred_weight` | `0.0` | Adds `Q_ag/Q_hd/Q_pr` STATE_QUERY tokens + 3 MSE losses against future-frame visual latents and future proprio. |
| SIGReg anti-collapse | `loss.sigreg_weight` | `0.0` | Adds Algorithm-1 SIGReg loss on encoder CLS outputs (4 streams when SP is on, 2 when SP is off). **Forces `projector.norm_type='batch'`** per LeWM paper Section 3. |
| Gripper-auxiliary regression head | `loss.gripper_aux_weight` | `0.0` | Predictor grows a small MLP that reads the BOS hidden state and regresses the full (H,) gripper-command sequence directly, bypassing FAST tokenization for dim 6. `eval_libero.py` auto-detects the head and overrides `actions_phys[..., 6]` at inference. |
| Multi-token visual prefix (V17) | `visual_tokens.pool_grid` | `0` | Encoder output per view becomes `1 (CLS) + G*G (spatially-pooled patches)` tokens instead of CLS only. Each view also gets a learnable view embedding and per-view 2D positional grid in the predictor input. |
| Mixture-of-Transformers (MoT) | `predictor.use_mot` | `false` | Each transformer block's FFN is split into separate experts per modality (prefix / action / state-query); attention is still shared. Predictor parameter count grows ~1.6–2x. |
| Language ablation | `data.dataset.use_language` | `true` | Drops T5 entirely (`lang_encoder`, `lang_proj` not constructed); prefix becomes `[z_ag, z_hd, z_pr, BOS, ...]`. |
| Step-based training | `trainer.max_steps` | unset | When set, the LinearWarmupCosineAnnealingLR scheduler is built with `interval=step`, `max_steps=cfg.trainer.max_steps`, and `warmup_steps = min(2000, 0.02 * max_steps)`. Otherwise the scheduler falls back to `max_epochs × len(train_loader)`. |
| 4-suite joint training | per-script flat HDF5 dir | n/a | When `data.dataset.hdf5_dir` points at `/Data/lyw/libero_processed_v5/all4_flat/` (40 symlinked task `.h5`), `LiberoDataset` globs all 40, applies a 3-level balanced `WeightedRandomSampler`, and logs per-task validation CE plus the `validate/ce_loss_taskbal` ckpt-selection metric. See "4-suite joint training" below. |

**Inference is identical for all variants.** STATE_QUERY tokens, MoT
modality routing, and the gripper-aux read-out are training-time
constructs. `generate()` autoregressively decodes FAST tokens from
`[lang, z_ag, z_hd, z_pr, BOS]` (or `[..., multi-token visual, ...]`)
exactly as in the frozen baseline; the only delta is the optional
`predict_gripper_aux()` call inside `eval_libero.py` to overwrite the
gripper dim.

Things this repo still does NOT do (do not re-add without explicit
instruction):
- `criterion()` planning-time latent cost — original LeWM CEM/MPC, never
  used by the VLA inference path
- `rollout()` — multi-step latent rollouts for planning
- `CEM` — Cross-Entropy Method action search
- Video co-training (Fast-WAM)
- Multi-frame history per view — each observation is a single timestep

## Current Architecture

### Input Encoders

| Input | Source | Encoder | Output | Frozen? |
|-------|--------|---------|--------|---------|
| Agentview image | `agentview_rgb` stored at 128×128 | ViT (LeWM encoder, patch_size=14) | `z_agent` (D,) | No, end-to-end |
| Eye-in-hand image | `eye_in_hand_rgb` stored at 128×128 | Same ViT (shared weights) | `z_hand` (D,) | No, end-to-end |
| Proprioception | Base-frame EE position(3) + base-frame EE orientation quaternion(4) + gripper state(2) = 9d | MLP → D | `z_proprio` (D,) | No |
| Language instruction | Task description string | T5-small encoder | `l_1...l_n` (n, D) | **Yes, frozen** |

`D` = embed_dim = 192 (LeWM default for ViT-Tiny). Two image views share the same ViT encoder (weight sharing). T5-small outputs are projected to D via a learned linear layer.

**Image resolution chain**: Raw LIBERO images are 128×128. The ViT uses `patch_size=14`, which requires the input to be divisible by 14. The dataset transform must **resize 128→224** (matching LeWM's original training resolution of 224×224, where 224/14=16 patches per side). Do NOT feed 128×128 directly to the ViT — it won't divide evenly.

**Proprioception rules (9d, verified):**
- **Source keys**: `obs/ee_pos` (3d position, meters, base frame) + `robot_states[5:9]` (4d xyzw quaternion, verified against `scipy.from_rotvec(obs/ee_ori)` at 2e-16 precision) + `obs/gripper_states[0:2]` (2d raw finger positions) → 9d total.
- **Note on `obs/ee_ori`**: this is a **3d axis-angle rotation vector** in base frame — the **same format as `action[3:6]`**. We use `robot_states[5:9]` (xyzw quaternion) as the proprio orientation component for historical consistency with the first iteration of this code. Do not confuse `obs/ee_ori` with Euler angles — an earlier version of this doc erroneously said so.
- **Quaternion re-normalization**: Re-normalize `robot_states[5:9]` to unit length before feeding to the MLP: `q = q / (||q|| + 1e-8)`. LIBERO simulator outputs should already be unit, but this is a defensive safeguard against numerical drift.
- **Gripper**: `obs/gripper_states` is (T, 2) — two nearly-symmetric finger joint positions. **Use both raw dims as-is.** **Never** apply `mean` / `abs` / any per-sample reduction. The two fingers are symmetric around 0, so `mean(grip)` silently collapses to ≈0 (no information) and `mean(abs(grip))` halves the dynamic range. Both `preprocess_libero.py` and `eval_libero.py` must produce matching 2-element gripper vectors.
- **Consistency rule**: `normalize_proprio(raw)` is the single source of truth for proprio normalization (takes a 9d vector, re-normalizes the quat at `[3:7]`, returns unchanged everywhere else). Both `preprocess_libero.py` and `eval_libero.py::preprocess_obs` call it on the same 9d layout.

### Unified Transformer Sequence

Training-time input sequence (frozen baseline, CLS-only visual, no SP):
```
[l_1, l_2, ..., l_n, z_agent, z_hand, z_proprio, BOS, T_1^GT, T_2^GT, ..., T_k^GT, PAD, ..., PAD]
 ←── perception prefix (bidirectional) ──→       ←── action tokens (causal) + padding ──→
```

When `visual_tokens.pool_grid = G > 0` (multi-token visual prefix), the
single `z_agent` / `z_hand` slots each expand to `1 + G*G` slots (CLS +
spatially-pooled patches). Each slot is still a single sequence position
with its own 1D positional encoding; `_build_attn_mask` treats them as
ordinary prefix tokens (bidirectional among themselves, blocked from
action / query / PAD). See "Multi-token Visual Prefix (V17)" below for
the per-view embedding and 2D positional addition.

When `loss.pred_weight > 0` (state prediction), three STATE_QUERY tokens
`Q_ag`, `Q_hd`, `Q_pr` are appended at the very end of the sequence —
see "State-Prediction Extension".

Inference:
```
[l_1, ..., l_n, z_agent, z_hand, z_proprio, BOS] → autoregressively generate T_1, ..., T_k, <EOS>
→ FAST decode → continuous action chunk (H=20 steps × 7 dims, anchor-relative)
→ closed-loop execution (see "Closed-loop chunk execution" below)
```
STATE_QUERY tokens are NOT appended at inference. Multi-token visual
prefix is preserved exactly as in training.

**Inference decoding rules (must be followed exactly):**
1. **Stop generation** when either `<EOS>` (id=1025) is sampled OR `max_action_tokens` tokens have been generated, whichever comes first.
2. **Post-process before FAST decode**: strip `<BOS>` from the beginning, truncate at `<EOS>` (discard EOS itself and anything after), discard any trailing tokens beyond `max_action_tokens`.
3. **Only token IDs 0..1023** (real FAST vocab) go into the FAST decoder. Never pass BOS/EOS/PAD to FAST decode.
4. Note: `action_head` outputs 1026 logits (0..1025), so PAD (id=1026) **cannot** be sampled by design — do NOT add extra masking for PAD during generation.
5. `generate()` additionally forces `logits[:, BOS_TOKEN_ID] = -inf` at every step — BOS is a start marker, never a valid body token. Without this mask the model could (rarely) re-emit id=1024, which the FAST BPE decoder does not expect.

Where:
- `l_1...l_n`: language token embeddings from frozen T5-small (variable length, typically 5-15 tokens)
- `z_agent`: CLS token from ViT encoding of agentview image
- `z_hand`: CLS token from ViT encoding of eye-in-hand image
- `z_proprio`: MLP encoding of 9d proprioceptive state: base-frame EE position(3) + xyzw quaternion(4) + raw gripper finger positions(2)
- `BOS`: beginning-of-action token (id=1024)
- `T_1...T_k`: FAST action tokens (discrete, vocab=0..1023, variable-length). On LIBERO with H=20 the observed max across all 4 suites is 75 tokens per chunk; `max_action_tokens=80` gives a small margin. Preprocessing asserts `max(fast_length) <= max_action_tokens` and errors out if the bound is ever breached.
- `PAD`: padding token (id=1026) for variable-length batching

### Attention Mask (Training)

**Prefix-bidirectional + action-causal hybrid mask.** Three zones:

```
           lang  z_ag  z_hd  z_pr  BOS   T_i   PAD
lang     [  Y     Y     Y     Y     -     -     -  ]
z_ag     [  Y     Y     Y     Y     -     -     -  ]
z_hd     [  Y     Y     Y     Y     -     -     -  ]
z_pr     [  Y     Y     Y     Y     -     -     -  ]
BOS      [  Y     Y     Y     Y     Y     -     -  ]
T_i      [  Y     Y     Y     Y     Y    csl    -  ]
PAD      [  -     -     -     -     -     -     -  ]
```

Y = can attend, - = blocked, csl = causal (attend to left only within action group)

Rules:
- **Perception prefix** (lang + z_ag + z_hd + z_pr): fully bidirectional among themselves, cannot see action tokens or PAD
- **Action tokens** (BOS + T_1...T_k): can see all real prefix tokens, causal within action group (each T_i sees BOS, T_1...T_i but not T_{i+1}...T_k), cannot see PAD
- **PAD tokens** (both action PAD and language PAD): attend to nothing, no other token attends to them
- **Language padding**: T5 tokenizer produces variable-length output. Shorter instructions are padded. Language PAD tokens must also be masked out — no real token should attend to them. Use `lang_lengths` to determine which language positions are real.

**Implementation**: Construct explicit `attn_mask = (B, 1, L, L)` bool tensor combining prefix-bidir, action-causal, language-padding, and action-padding rules. Do NOT use `is_causal=True` — it only supports pure causal masks.

Standard causal teacher forcing within action zone: position i's target is the next token (i+1). Each position sees itself (no leak because target is shifted).

### Targets and Loss

**Output and target shapes — read carefully:**

`action_logits` is extracted **only from action-zone positions** (BOS + T_1...T_k + PAD positions), NOT from the full sequence. The prefix positions (lang, z_ag, z_hd, z_pr) do NOT go through `action_head` and do NOT produce logits.

```
action_logits shape: (B, 1 + max_action_tokens, 1026)
                          ^ BOS + T_1...T_k + PAD positions only

targets shape:       (B, 1 + max_action_tokens)
  BOS position:      target = T_1    (first real action token)
  T_1^GT position:   target = T_2
  ...
  T_k^GT position:   target = <EOS>  (id=1025)
  PAD positions:     target = -100   (ignored by cross_entropy)
```

Both tensors have the same first two dimensions. No prefix positions appear in either tensor.

**Loss = L_CE only.**
```python
L_CE = F.cross_entropy(
    action_logits.reshape(-1, 1026),
    targets.reshape(-1),
    ignore_index=-100,
    label_smoothing=0.1,  # configurable via `cfg.label_smoothing`; 0 disables
)
```

Baseline uses `label_smoothing=0.1` as a mild regularizer, which puts a CE
floor at approximately 1.02 on a 1026-way vocabulary — compare curves
against this floor rather than 0. `token_accuracy` (argmax match rate
excluding PAD/prefix, logged in `train.py`) is the cleaner diagnostic when
comparing smoothed vs. non-smoothed runs.

No L_pred (world model prediction loss). No L_sigreg (anti-collapse regularizer). These may be added back as ablations later.

### Special Tokens

- Vocab 0..1023: FAST action tokens
- Token 1024: `<BOS>` — beginning of action sequence
- Token 1025: `<EOS>` — end of action sequence (target for last real action position)
- Token 1026: `<PAD>` — padding for variable-length batching

`nn.Embedding` size = 1027 (0..1026). `action_head` output size = 1026 (predict 0..1025; PAD is never a prediction target).

### Temporal Indexing Convention

**CRITICAL — read this before implementing anything.**

All timestep indices are **raw-step indices** (not chunk-level). Preprocessing
uses a **sliding window with stride=1** (standard VLA/ACT/π₀ practice) so
consecutive training samples share H-1 actions but each gets a fresh
observation anchor, maximizing data coverage.

- `H` = action chunk length in raw environment steps = 20 (1 second at LIBERO's 20Hz control frequency)
- `stride` = sliding window stride in raw steps, default **1** (see `preprocess_libero.py --stride`)
- `t` = raw step index (0, 1, 2, ..., T-H-1 are valid chunk anchors for a demo of length T)
- Observation `o_t` = frame at raw step `t` (NOT `t * H`)
- Chunk at anchor `t` requires raw steps `[t, t+1, ..., t+H]` of observations
  (H+1 frames — the extra frame is needed to compute the last anchor-relative target)
  and raw steps `[t, ..., t+H-1]` of actions (H gripper commands)
- FAST tokens: `[T_1...T_k]` = `FAST_encode(action_chunk_t)` where `action_chunk_t`
  is the anchor-relative chunk defined in **"Anchor-relative action chunk"** below

Per-demo sample count with stride=1: `max(0, T - H)`, roughly ~H× more samples
than the old non-overlapping slicing (which used `T // H`).

Since this baseline has no world model prediction, each training sample only
needs the obs at step `t` (two views + proprio) plus the action chunk and
language instruction.

### Anchor-relative action chunk

Each chunk element `k` (for `k = 0..H-1`) is the **cumulative displacement from
the chunk-start anchor state `(p_t, R_t)`**, not a step-to-step delta. This is
the key difference from a naive VLA baseline and removes the "implicit world
model" the model would otherwise have to learn.

```
chunk[k, 0:3] = obs/ee_pos[t+k+1] - obs/ee_pos[t]                       # base frame, meters
chunk[k, 3:6] = rotvec(R_{t+k+1} · R_t^{-1})                            # base frame, axis-angle rad
               where R_x = scipy.Rotation.from_rotvec(obs/ee_ori[x])
chunk[k, 6]   = actions[t+k, 6]                                         # gripper command, unchanged
```

**Computed from observed HDF5 poses**, not by accumulating the recorded
delta-action commands — OSC PID tracking is imperfect, so the observed poses
are the true ground truth for the positions the robot actually visited.

Stored in physical units (m / rad / command) inside the HDF5, then
1st/99th-percentile normalized to [-1, 1] before FAST tokenization.

**Gripper note**: dim 6 is NOT anchor-relative. It's the raw gripper command at
each step, which is already a "direct command" value in [-1, 1] (not a pose).
It goes into the chunk unchanged.

### Action format conventions

All of the following are **empirically verified** on LIBERO's demo HDF5 and
robosuite's installed sources. Do not second-guess these without re-verifying.

- **`action[0:3]`** — position delta in **base frame**, robosuite input range
  `[-1, 1]`. Internally the controller multiplies by `pos_scale = 0.05 m` to
  get a physical delta per step (robosuite `OSC_POSE` `output_max[0:3]`).
- **`action[3:6]`** — **axis-angle rotation vector delta** (NOT Euler), base
  frame, `[-1, 1]`. Internally multiplied by `rot_scale = 0.5 rad` (robosuite
  `output_max[3:6]`). This is the SAME representation as `obs/ee_ori`.
- **`action[6]`** — gripper command, `[-1, 1]`. Direct command value, **not a
  delta** and not scaled by OSC; passed through unchanged.
- **`robot_states[5:9]`** — unit-norm **xyzw** quaternion (scipy / robosuite
  default, verified against `scipy.from_rotvec(obs/ee_ori)` at 2e-16 precision).
- **`obs/ee_ori`** — 3d axis-angle rotation vector, same convention as
  `action[3:6]`. Continuous across all libero_spatial demos, no ±π wraparound.

### Closed-loop chunk execution at inference

The predicted chunk stores **anchor-relative** displacements from the chunk-
start observation state `(p_t, R_t)`. To execute it against the env we close
the loop step-by-step rather than open-loop, which lets the OSC controller
self-correct when the robot drifts away from the predicted trajectory.

```python
# Once per chunk: snapshot the anchor (p_t, R_t)
anchor_pos  = obs["robot0_eef_pos"].copy()
anchor_quat = obs["robot0_eef_quat"].copy()       # xyzw
R_anchor    = scipy.Rotation.from_quat(anchor_quat)

for k in range(H):
    # 1. Target derived from the anchor + predicted displacement
    target_pos = anchor_pos + chunk[k, 0:3]                 # physical meters
    target_R   = Rotation.from_rotvec(chunk[k, 3:6]) * R_anchor
    gripper    = chunk[k, 6]                                 # command, unchanged

    # 2. Residual delta from the robot's *current* pose to the target
    cur_pos = obs["robot0_eef_pos"]
    cur_R   = Rotation.from_quat(obs["robot0_eef_quat"])
    step_delta_pos    = target_pos - cur_pos
    step_delta_rotvec = (target_R * cur_R.inv()).as_rotvec()

    # 3. Scale back into robosuite's [-1, 1] action range and step
    action_input = np.concatenate([
        np.clip(step_delta_pos / POS_SCALE, -1, 1),
        np.clip(step_delta_rotvec / ROT_SCALE, -1, 1),
        [gripper],
    ])
    obs, reward, done, _ = env.step(action_input)
    if done: break
```

The `np.clip(..., -1, 1)` is for readability only — robosuite's
`Controller.scale_action` already does the exact same clip internally before
linear rescaling. See "Robosuite OSC constants" below.

### Robosuite OSC constants

Verified by `inspect.getsource` on the server's robosuite install.

- **`pos_scale = 0.05 m`** per unit of action input (`output_max[0:3]`)
- **`rot_scale = 0.5 rad`** per unit of action input (`output_max[3:6]`)
- **`input_max = 1.0`, `input_min = -1.0`** — actions outside this range are
  clipped first, then linearly rescaled to `[output_min, output_max]`.
- **`control_delta = True`** — actions are interpreted as deltas from the
  current EE pose at the moment `env.step` is called.
- **`uncouple_pos_ori = True`** — position and orientation are controlled by
  independent PID loops.
- **LIBERO override check**: none. LIBERO calls
  `suite.load_controller_config(default_controller="OSC_POSE")` and forwards
  the result unchanged (`env_wrapper.py:47`, grep-verified).
- **Best practice in `eval_libero.py`**: read `POS_SCALE` / `ROT_SCALE` at
  runtime from `env.env.robots[0].controller.output_max[0]` / `[3]` with
  assertions for uniformity, rather than hardcoding. If LIBERO or robosuite
  ever change their defaults, the eval picks up the new value automatically
  and the assertions catch any non-uniform scale.

### Multi-token Visual Prefix (V17)

Default (`visual_tokens.pool_grid = 0`): each view contributes a single
CLS token. With `pool_grid = G > 0` the encoder output `(B, 1+P, D)` is
post-processed in `JEPA._pool_visual_tokens` as:
- `cls = hidden[:, :1]` — keep CLS as-is
- `patches = hidden[:, 1:]` reshaped to `(B, D, side, side)` where
  `side² = P`. With `img_size=224, patch_size=14`, `side = 16` and
  `P = 256`. `_pool_visual_tokens` will `raise ValueError` if P is not
  a perfect square (rules out DINOv2-with-registers, which inserts
  4 register tokens after CLS).
- `pooled = adaptive_avg_pool2d(feat, (G, G))` then reshape +
  transpose to `(B, G*G, D)` (row-major flattening).
- Return `torch.cat([cls, pooled], dim=1)` → `(B, 1+G*G, D)`.

So with `G = 4`, `n_visual_per_view = 1 + 16 = 17` (hence "V17"). Both
agent and hand views go through the SAME `_pool_visual_tokens`.

In `JEPA.encode` the (2B, N, D) post-pool tokens are reshaped to
`(2B*N, D)` before the projector and reshaped back to `(2B, N, D)`
afterwards — this is the **Bug #1 fix** (Apr 2026) that unblocked
BatchNorm projector + multi-token. `encode_future_visual` mirrors the
same shape exactly so both source and target paths feed the projector's
BN1d with the same (B*N, D) distribution — see "State-Prediction
Extension" → "Bug #1 / #2 fixes" below.

In `ARPredictor.forward` each visual token receives:
- `type_embedding[1]` (visual)
- `view_embedding[0]` (agent) or `view_embedding[1]` (hand) — only when
  `n_visual_per_view > 1`
- For patch tokens at grid cell `(r, c)`:
  `agent_patch_2d_pos[r, c]` or `hand_patch_2d_pos[r, c]` —
  **separate per-view 2D positional grids**, never shared.
- A 1D positional encoding from `pos_embedding` (every token in the
  sequence gets one).
- CLS tokens within a view do NOT get a 2D positional embedding; only
  view embedding + 1D pos.

`pos_embedding` is sized
`max_lang_tokens + 2*n_visual_per_view + 1 + 1 + max_action_tokens
+ n_state_query`, so for V17 + SP this is `25 + 34 + 1 + 1 + 80 + 3 = 144`
positions; for V17 without SP it is 141.

**Attention rules under multi-token visual**: All 1+G*G tokens of each
view are part of the prefix zone (bidirectional among themselves and
with the other view, lang, and proprio). The action zone still sees
the full real prefix (now 2*(1+G*G) + lang + 1 proprio tokens) and is
causal within itself.

### Mixture-of-Transformers (MoT)

Off by default. When `predictor.use_mot = true` each `Block` in
`ARPredictor` swaps its single `FeedForward` for a `MoTFeedForward`
with one expert per modality (attention is still shared across all
tokens). Modality is a hard partition by absolute position — no
learned gating:

```
M = 0  →  prefix (lang + visual + proprio, positions 0 .. n_lang + 2*nv)
M = 1  →  action zone (BOS + action tokens + PAD)
M = 2  →  state-query (Q_ag / Q_hd / Q_pr — only when use_state_prediction)
```

`n_modalities` is fixed at construction: 3 with SP, 2 without.
`MoTFeedForward.forward(x, modality_ids)` runs **every expert on the
full sequence** and merges by per-position modality mask — about 2×
slowdown on the FFN layer in exchange for code simplicity (scatter /
gather would be faster but messier).

Param impact at the default predictor size (ViT-tiny 192 / depth 6 /
mlp 2048):
- baseline (MoT off, SP on): 11.92 M predictor params
- MoT on, SP on (3 modalities): 21.39 M (+79 %)
- MoT on, SP off (2 modalities): 14.65 M (+23 %)

The `_build_modality_ids` helper produces an `(B, L)` long tensor
deterministically from the sequence layout. The three `ARPredictor`
forward paths (`forward`, `generate`, `predict_gripper_aux`) all build
their own `modality_ids` and pass them through each `Block`.

When `use_mot = false` the Block's `FeedForward` is the original single
shared MLP; `modality_ids` is `None` and is ignored. The baseline
checkpoint at `7008f15` was trained without MoT and is bit-identical
under `use_mot = false`.

### Gripper-auxiliary head

Off by default. When `loss.gripper_aux_weight > 0` the predictor grows
a small bypass head:

```python
self.gripper_aux_head = nn.Sequential(
    nn.Linear(embed_dim, embed_dim * 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(embed_dim * 2, self.gripper_chunk_size),
    nn.Tanh(),
)
```

`gripper_chunk_size` is fed in from `cfg.data.dataset.chunk_size`
(defaults to `20`, i.e., the chunk length `H`). The head reads the
hidden state at the **BOS position** (`x[:, n_prefix]`) and predicts
the entire (H,) gripper-command sequence in one shot, in `[-1, 1]`
thanks to the trailing `Tanh`.

Why BOS specifically: under the prefix-bidir + action-causal mask BOS
attends only to the prefix tokens (lang + visual + proprio); it does
**not** see any action token. The hidden state at BOS is therefore
identical at training (when the full action zone is teacher-forced
into the sequence) and at inference (when no action tokens have been
emitted yet). No exposure-bias mismatch.

Inference plumbing:
- `module.py::ARPredictor.predict_gripper_aux` runs a prefix-plus-BOS
  forward (no action tokens at all) and applies `gripper_aux_head` to
  the BOS hidden state. Uses the same `_build_generate_mask` and
  `_build_modality_ids(has_query=False)` as `generate()`.
- `jepa.py::JEPA.predict_gripper_aux` is the matching thin wrapper.
- `eval_libero.py` auto-detects with `getattr(model.predictor,
  "use_gripper_aux", False)` and overwrites
  `actions_phys[..., 6] = pred_grip[0].detach().cpu().numpy()` after
  the FAST decode but before the closed-loop chunk execution. No CLI
  flag is needed; the override is keyed off the loaded checkpoint's
  architecture.

Training loss:
```python
gripper_seq = batch["gripper_seq"]   # (B, H) float32 in [-1, 1]
output["gripper_aux_loss"] = F.mse_loss(pred_grip, gripper_seq)
total_loss = total_loss + gripper_aux_weight * output["gripper_aux_loss"]
```
`libero_dataset.py` already supplies `gripper_seq` for every sample
(unconditionally, since the cost is negligible): it slices
`continuous_actions[i, :, 6]` which is the per-task-normalized gripper
command — but the per-task `(action_low, action_high)` for dim 6 is
always `(-1, +1)` (constant command range) so the normalization is the
identity and the values match the raw OSC gripper commands.

Loading old checkpoints: `eval_libero.py::load_checkpoint` now refuses
to load `_weights.ckpt` (Lightning state_dict) when it sees any of three
non-baseline key patterns:
1. `state_pred_head_*` / `state_query_embeddings` (SP-trained)
2. `projector.net.1.running_mean` (BatchNorm projector / SIGReg-trained)
3. `gripper_aux_head` (gripper-aux-trained)

The error message points the user at the matching `_object.ckpt`
(pickled JEPA from `ModelObjectCallBack`), which preserves the full
architecture without needing the loader to know about every toggle.

### 4-suite joint training

Frozen baseline trained on the 10 libero_spatial tasks. The current
primary training mode is **joint over all 4 LIBERO suites** (spatial +
object + goal + 10 = 40 tasks).

How it is enabled — pure config, no code switch:
- `data.dataset.hdf5_dir` points at a flat directory containing 40
  preprocessed `.h5` files. The recommended layout is a symlink farm:
  `/Data/lyw/libero_processed_v5/all4_flat/` with names like
  `spatial_pick_up_..._on_the_plate.h5` pointing into the per-suite
  preprocessed dirs.
- `LiberoDataset.__init__` is non-recursive (`hdf5_dir.glob('*.h5')`),
  so the flat dir is required.

What the dataset emits for 4-suite training (verifiable in
`libero_dataset.py::__getitem__`):
- `task_id`: int — `self._file_to_task_id[fpath]`, stable across runs
  (sorted file order).
- `gripper_seq`: (H,) float32 — for the gripper-aux head (always
  populated).
- Standard fields: `pixels_agent`, `pixels_hand`, `proprio`,
  `fast_tokens`, `fast_lengths`, plus `lang_input_ids` /
  `lang_attention_mask` when `use_language=True` and
  `pixels_*_future` / `proprio_future` when `use_state_prediction=True`.

3-level balanced sampler — `LiberoDataset.get_sampler_weights` computes
`w_i = 1 / (n_tasks * n_demos[task_i] * n_chunks[(task_i, demo_i)])`,
fed into `torch.utils.data.WeightedRandomSampler(..., replacement=True)`.
This makes the marginal distribution uniform across tasks, demos within
a task, and time within a demo (overriding the natural-frequency bias
toward tasks / demos with more chunks).

Per-task validation CE — `train.py::lejepa_forward` logs
`validate/ce_loss/task_{t}` (one scalar per task in the batch's
`task_id` set) during the validate stage only. The
`TaskBalancedCEMetric` callback in `utils.py` aggregates these into a
single `validate/ce_loss_taskbal` scalar at `on_validation_epoch_end`.
This is the **ckpt-selection metric** for joint training — task-balanced,
unweighted by task chunk count. `pick_best_ckpt.py` reads it from the
TB event files and returns the top-K `_object.ckpt` paths.

`ModelObjectCallBack` (in `utils.py`) supports both epoch-based and
step-based ckpt cadence:
- Default (no `trainer.max_steps`): dumps a `_object.ckpt` every epoch
  end, keeps top-K by `validate/total_loss_epoch`.
- With `step_interval` set (= `cfg.trainer.val_check_interval` when
  step-based training is active): dumps a `_object.ckpt` every val pass,
  keeps top-K by `validate/ce_loss_taskbal`, and maintains a
  `lewm_latest_object.ckpt` symlink at the most-recent surviving step.
  The symlink is refreshed by `_refresh_latest_link` whenever the top-K
  heap evicts a ckpt, so the symlink never dangles.

### Scope: What the Frozen Baseline Does NOT Include

These bullets describe the **frozen baseline** (all toggles off,
`pred_weight = sigreg_weight = gripper_aux_weight = 0`,
`visual_tokens.pool_grid = 0`, `predictor.use_mot = false`).

- **No world model prediction (in baseline)**: no STATE_QUERY, no L_pred, no future frame encoding.
- **No SIGReg (in baseline)**: class kept in `module.py` but only instantiated by `train.py` when `sigreg_weight > 0`.
- **No CEM planning (ever)**: actions are directly generated via autoregressive decoding, not searched.
- **No multi-token visual prefix in baseline**: `visual_pool_grid = 0` → CLS only per view.
- **No MoT in baseline**: each block has a single shared FFN.
- **No gripper aux head in baseline**: gripper dim 6 is decoded purely from FAST.
- **Joint training across all 10 LIBERO-Spatial tasks** (one checkpoint, ~65 K chunks total). The language instruction is the sole per-task disambiguation signal — chunks from different tasks have different `language_instruction` strings tokenized by frozen T5. The frozen baseline at `7008f15` was trained this way (see `/Data/lyw/checkpoints/multitask_ln_100ep/config.yaml` on the GPU server: `hdf5_dir: /Data/lyw/libero_processed/libero_spatial` points at the suite directory, so `LiberoDataset` globs all 10 `.h5` files).
- **No history context**: single-frame input per view — each observation is a single timestep (one agentview + one eye_in_hand + one proprio reading). Multi-frame history is a potential follow-up.

## State-Prediction Extension

This section documents the additive variant on top of the frozen baseline,
gated by `cfg.loss.pred_weight > 0` (state prediction) and
`cfg.loss.sigreg_weight > 0` (SIGReg). When both weights are 0, the
codebase reduces exactly to the frozen baseline.

### Architecture additions

When `use_state_prediction=True` (auto-set by `train.py` from
`pred_weight > 0`), three learnable query tokens `Q_ag`, `Q_hd`, `Q_pr`
are appended at the **end** of the training-time sequence:

```
[lang..., z_ag, z_hd, z_pr, BOS, T_1...T_k, PAD..., Q_ag, Q_hd, Q_pr]
```

Sequence length grows from `n_lang + 3 + 1 + max_action_tokens` (109 by
default) to `... + 3` (112 by default). `pos_embedding` is sized
accordingly when the flag is on.

**Attention rules for STATE_QUERY rows** (extends the prefix-bidir +
action-causal mask of the baseline):

- Each `Q_x` sees: all real prefix (lang + visual + proprio) + the entire
  real action zone (BOS + non-PAD `T_i`) + itself.
- Each `Q_x` does NOT see: PAD, OR any other `Q_y` (preserves three-way
  prediction independence — `pred_pr` cannot peek at `Q_ag`'s hidden
  state).
- No token outside the query block attends to `Q_*` (queries are pure
  read-outs; their column in the attention matrix is all-False).

**Three prediction heads** read out the post-transformer hidden state at
the corresponding `Q_x` position. Per LeWM paper Section 3 ("the
predictor is also followed by a projector network with the same
implementation as the one used for the encoder") and confirmed by
upstream `train.py` (`predictor_proj = MLP(input_dim=hidden_dim,
output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)`),
each head is a `MLP(D → 2048 → out, norm_fn=...)` with the **same** norm
type as the encoder-side projector — passed through to `ARPredictor` via
the `state_head_norm_type` constructor arg, which `train.py` mirrors
from `cfg.projector.norm_type`.

| Head | Input | Output (`module.py`) | Target (`train.py`) |
|------|-------|---------------------|---------------------|
| `state_pred_head_ag` | `h_{Q_ag} ∈ ℝ^D` | `MLP(D → 2048 → D*nv)` then `.reshape(B, nv, D)` → `(B, nv, D)` | `z_agent_future` shape `(B, nv, D)` (Bug #2 fix) |
| `state_pred_head_hd` | `h_{Q_hd} ∈ ℝ^D` | same shape as `state_pred_head_ag` | `z_hand_future` shape `(B, nv, D)` |
| `state_pred_head_pr` | `h_{Q_pr} ∈ ℝ^D` | `MLP(D → 2048 → proprio_dim)` → `(B, 9)` | re-normalized `proprio_{t+H}` (9d), same `normalize_proprio` treatment as the predictor input |

`nv = n_visual_per_view`. For CLS-only (`visual_pool_grid = 0`), `nv = 1`
and the head output collapses to `D` — bit-identical to the original
frozen-baseline architecture; old SP checkpoints load via `_object.ckpt`
without surgery. For V17 (`pool_grid = 4`), `nv = 17`, and the head's
last linear grows from `D → D` to `D → 17*D`. The total predictor params
under SP grow from 11.92 M (nv=1) to 24.52 M (nv=17).

The proprio head terminates in `proprio_dim` (=9) and predicts the
**re-normalized 9d proprio vector at t+H** (re-normalize = unit-length
quaternion at `[3:7]`, ee_pos and gripper passthrough; same
`preprocess_libero.py::normalize_proprio` applied to current and future
proprio symmetrically — `proprio` and `proprio_future` use the same
treatment so the MSE target's scale matches the predictor input's scale).

### `L_pred` (state-prediction loss)

```
L_pred = MSE(pred_ag, z_agent_future)    # both (B, nv, D); per-token MSE
       + MSE(pred_hd, z_hand_future)     # both (B, nv, D); per-token MSE
       + MSE(pred_pr, proprio_future)    # raw 9d MSE
```

`z_agent_future` and `z_hand_future` come from
`JEPA.encode_future_visual`, which after the Bug #1 fix uses **the same
pool + reshape + projector pipeline as `encode()`** — so for CLS-only
the visual targets are `(B, 1, D)` and for V17 they are `(B, 17, D)`.
`train.py::lejepa_forward` normalizes both sides to 3D before the MSE
so the `nv=1` case is bit-identical to the legacy `(B, D)` MSE.

**No `target.detach()`.** Per LeWM paper Section 3:
> "We do not employ stop-gradient, exponential moving averages, or
> additional stabilization heuristics. Gradients are propagated through
> all components of the loss, and all parameters are optimized jointly
> in an end-to-end manner."

The target ViT branch (encoding the t+H frames) shares weights with the
source branch and contributes gradients. SIGReg is what holds off
collapse — see below.

### `L_sigreg` (anti-collapse regularizer)

Following **LeWM paper Algorithm 1** strictly (which is the upstream
`train.py::lejepa_forward` source of truth — paper Figure 1 ambiguously
visualizes SIGReg on multiple stream types, but Algorithm 1's pseudocode
applies it only to encoder outputs `emb`, not predictor outputs
`next_emb`):

```python
z_ag_cls         = z_agent[:, 0]          if z_agent.dim() == 3 else z_agent
z_hd_cls         = z_hand[:, 0]           if z_hand.dim() == 3 else z_hand
z_ag_future_cls  = z_agent_future[:, 0]
z_hd_future_cls  = z_hand_future[:, 0]
sigreg_input = torch.stack(
    [z_ag_cls, z_hd_cls, z_ag_future_cls, z_hd_future_cls], dim=0,
)  # (T=4, B, D), encoder-only, CLS only
L_sigreg = sigreg(sigreg_input)
```

SIGReg's input is always the **CLS slice** of the encoder output, even
under V17 (where the encoder also emits 16 patch tokens per view).
Rationale: SIGReg's job is to push a single per-view embedding toward
isotropic Gaussian; the patch tokens are constrained instead by the
per-token SP MSE above (Bug #2 fix).

**Predictor outputs** (`pred_ag`, `pred_hd`, `pred_pr`) are intentionally
NOT included in the SIGReg input stack. They are pulled toward
encoder outputs via `L_pred`, and the encoder outputs are constrained by
SIGReg, so predictor outputs are indirectly anchored to an isotropic
Gaussian distribution.

When `pred_weight = 0` but `sigreg_weight > 0` (rare configuration), the
input degenerates to `(T=2, B, D)` with current views only, since
`z_*_future` are not encoded.

### Bug #1 / Bug #2 fixes for multi-token visual

Two architectural fixes landed in May 2026 to make SP / SIGReg work
under V17. Background symptom: V17 + sp_sigreg caused
`validate/pred_loss` to be ~8.6 × higher than CLS-only (0.053 → 0.459)
and broad task regression (spatial 83 % → 55.5 %, libero_10
21.5 % → 0.5 %) vs the CLS baseline.

- **Bug #1 — BN projector distribution bias** (`fix 97ff7d8`,
  `jepa.py::encode_future_visual`). Before the fix, `encode()` fed
  `(B*N, D)` to the projector (N=17 under V17) while
  `encode_future_visual()` took only `last_hidden_state[:, 0]` →
  `(B, D)`. BN1d running mean/var got dominated by the source path's
  17×-sample mixed CLS+patch distribution, biasing the eval-time
  normalization of CLS-only targets. Fix: `encode_future_visual` now
  uses the same `_pool_visual_tokens + reshape + projector + reshape`
  pipeline; both source and target feed BN with identical (B*N, D)
  distributions.

- **Bug #2 — Per-token SP supervision** (`fix a30106d`,
  `module.py::ARPredictor` + `train.py::lejepa_forward`). Before the
  fix, the SP heads emitted `(B, D)` and the loss measured MSE against
  the CLS slice only, so the 16 patch tokens of the source path had no
  direct SP gradient — they only received gradient via the action CE
  through attention from action tokens. Patches drifted into noise.
  Fix: `state_pred_head_ag/hd` output `embed_dim * nv` and reshape to
  `(B, nv, D)`; the SP MSE compares full multi-token target. Now
  source patches at position i get direct SP gradient through Q_ag's
  attention back to themselves (Q_ag must encode the spatial structure
  of all `nv` target tokens, which forces source patches to carry
  useful spatial info).

SIGReg input still uses CLS slice only — patch-level SIGReg was
deliberately not enabled (would change the semantic of the per-view
anti-collapse constraint and add `2 * G²` extra streams to the
4-stream stack).

Backward compatibility: both fixes are bit-identical under `nv = 1`.
SP checkpoints trained before the fixes that used `nv = 1` still load
cleanly via `_object.ckpt`. V17 SP checkpoints from before the fixes
were on the broken design and cannot be revived without retraining.

### Total loss

```
L_total = L_CE + pred_weight · L_pred + sigreg_weight · L_sigreg
```

**Default weights (LeWM paper)**: `pred_weight = 1.0` (paper has no
explicit weight on `L_pred`; the only hyperparameter is λ on SIGReg) and
`sigreg_weight = 0.1` (paper Section 3, "Unless otherwise specified, we
use M = 1024 projections and λ = 0.1"). The frozen baseline runs with
both at 0.

### Known caveats of switching to BatchNorm projector

When `sigreg_weight > 0`, `train.py` enforces
`projector.norm_type = 'batch'` because LeWM paper Section 3 says:
> "The projection step ... uses a 1-layer MLP with **Batch
> Normalization**. This step is necessary because the final ViT layer
> applies a Layer Normalization, **which prevents our anti-collapse
> objective from being optimized effectively**."

The frozen baseline was trained with `LayerNorm` projector (no SIGReg),
which is why `norm_type` defaults to `'layer'`. Flipping to BatchNorm
introduces three side effects worth knowing about:

1. **Train/eval statistics mismatch.** `BatchNorm1d` tracks a running
   mean/variance during training and uses those stats during eval. If
   the eval batch size or distribution differs significantly from the
   training mini-batch (e.g., evaluating on a single trajectory at a
   time, or on out-of-distribution scenes), the running stats may not
   match the encountered distribution and outputs will drift relative
   to training. The frozen baseline's LayerNorm path doesn't have this
   problem because LayerNorm is per-sample.

2. **Batch-size sensitivity.** SIGReg's normality test on
   1024-projection samples requires enough samples per mini-batch to
   estimate distribution moments stably. BatchNorm itself also wants
   ≥ 32 samples per batch for stable running stats. With our default
   `batch_size=128`, both are well above the threshold; if you ever
   reduce to `batch_size=16` or below for VRAM reasons, expect SIGReg
   to be noisy and BN running stats to drift. Don't go below
   `batch_size=32` with this configuration.

   **Specific consequence for `overfit.yaml`:** the 1-demo overfit config
   defaults to `batch_size=8`, which is correct for the baseline pipeline
   sanity check but is too small for SP+SIGReg (BN stats become noise,
   Epps–Pulley statistic becomes high-variance). When running
   state-prediction inside `overfit.yaml`, override
   `loader.batch_size=32` (or higher) on the CLI:

   ```bash
   python train.py --config-name=overfit \
       loss.pred_weight=1.0 loss.sigreg_weight=0.1 \
       projector.norm_type=batch \
       loader.batch_size=32 \
       data.dataset.hdf5_dir=...
   ```

   `overfit.yaml` carries this warning in its header comment. We chose
   to keep the baseline `batch_size=8` default and require an explicit
   override (rather than switch overfit to GroupNorm or replace the
   1-demo overfit with a 10-demo×5-epoch sanity) because (a) we want the
   SP run to track paper architecture exactly, and (b) the baseline
   sanity at `batch=8` is a useful tool worth preserving.

3. **Multi-GPU SyncBN requirement.** With `devices > 1`, vanilla
   `BatchNorm1d` computes statistics per-GPU, which means each GPU sees
   a different normalization. For correct end-to-end behavior the
   projector should use `nn.SyncBatchNorm` (via
   `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` after model
   construction). The frozen baseline's `LayerNorm` is per-sample so
   has no such concern. Single-GPU training (`devices=1`) is unaffected.
   The current `train.py` does NOT auto-convert to SyncBN; flag this
   when adding multi-GPU training.

### File flow

| Component | What it does when `use_state_prediction=True` |
|-----------|---------------------------------------------|
| `preprocess_libero.py` | Extracts and stores `image_agent_future`, `image_hand_future`, `proprio_future` (the t+H frame) alongside the existing fields. Old preprocessed HDF5 must be regenerated. |
| `libero_dataset.py` | Asserts the future fields exist in HDF5; loads them in `__getitem__` and yields `pixels_*_future` and `proprio_future` as part of the batch. Old HDF5 still works when `use_state_prediction=False`. |
| `module.py::ARPredictor` | Adds `state_query_embeddings` (3,D), three prediction heads, an extended attention mask, and an extra type embedding row (5 instead of 4). `state_pred_head_ag/hd` output `embed_dim * n_visual_per_view` (Bug #2 fix); `forward()` reshapes their outputs to `(B, nv, D)` before returning. `forward()` returns `(action_logits, pred_ag, pred_hd, pred_pr)` plus an optional `pred_grip` (5-tuple when SP and grip-aux are both on). `generate()` is **unchanged** — STATE_QUERY tokens are training-only. |
| `jepa.py::JEPA` | Adds `encode_future_visual()` that runs the SHARED ViT + projector on the t+H frames with NO `.detach()`. Post Bug #1 fix, it mirrors `encode()` exactly: pool → reshape adapter `(B, N, D) → (B*N, D) → projector → (B, N, D')`. Returns `(B, N, D)` so both source and target paths feed the projector's BN1d with the same distribution. |
| `train.py::lejepa_forward` | Computes `L_pred` (3 MSE) when `pred_weight > 0`, computes `L_sigreg` on the encoder-only 4-stream CLS-slice stack when `sigreg_weight > 0`, sums into `L_total`. Normalizes future-visual targets to 3D before the MSE so the loss code path is identical for `nv=1` and `nv>1`. Validates `projector.norm_type == 'batch'` when SIGReg is on. |
| `eval_libero.py` | Inference flow `[lang, z_ag, z_hd, z_pr, BOS] → AR decode → FAST` is unchanged. The only deltas: (1) `load_checkpoint()` now hard-rejects non-baseline `_weights.ckpt` payloads (SP keys OR BatchNorm projector running stats OR `gripper_aux_head` keys) with a clear message pointing at `_object.ckpt`; (2) after the FAST decode, if `model.predictor.use_gripper_aux` is True, `actions_phys[..., 6]` is overwritten with `model.predict_gripper_aux(...)[0]`. SP-trained, "SIGReg-only no SP", and gripper-aux-trained checkpoints all must be evaluated via the per-step `_object.ckpt` produced by `ModelObjectCallBack`. |

### How to enable

```bash
# Re-preprocess (one-time, populates image_*_future + proprio_future)
python preprocess_libero.py --input <raw.hdf5> --output <out.h5> ...

# Train with paper-default weights
python train.py data=libero \
    loss.pred_weight=1.0 \
    loss.sigreg_weight=0.1 \
    projector.norm_type=batch        # required by SIGReg

# Eval (use the per-step object checkpoint produced by ModelObjectCallBack;
# epoch-based runs name them lewm_epoch_{N}_object.ckpt instead)
python eval_libero.py \
    --checkpoint /path/to/lewm_step_{N}_object.ckpt \
    --tokenizer /path/to/fast_tokenizer \
    --processed-dir /path/to/libero_processed/<suite>/ \
    --suite libero_spatial
```

### Known concerns (not auto-fixed)

These are issues identified during code audit that we deliberately did
NOT auto-patch, because the right fix requires data-level evidence we
can only collect on the GPU server.

**Quaternion sign ambiguity in `proprio_future` MSE.**
A unit quaternion `q` and its negation `-q` represent the same rotation,
but `MSE(q, -q) = 4` (large). If LIBERO's MuJoCo simulator ever flips
the sign of `obs/robot_states[:, 5:9]` between raw step `t` and `t+H`
within a single demo (e.g., when crossing the antipodal hemisphere
boundary), `loss_pred_pr` will spike at those samples even though the
predicted rotation is correct.

We did NOT add hemisphere canonicalization (`if q[3] < 0: q = -q`) to
`normalize_proprio` because:
1. We can't verify locally whether LIBERO actually produces sign flips
   (likely rare for short H=20 chunks in continuous trajectories, but
   not impossible).
2. `eval_libero.py::preprocess_obs` calls `normalize_proprio` at runtime
   on environment observations. Adding canonicalization would change
   the proprio distribution that the **frozen baseline** (trained on
   un-canonicalized data, ckpt at `7008f15`) sees at eval time —
   risking unmeasured eval drift on the baseline.

If `loss_pred_pr` shows unexplained spikes during SP training, add this
to `normalize_proprio`:
```python
# Hemisphere canonicalization: q and -q encode the same rotation.
# Force w >= 0 so MSE(pred, target) doesn't pay the antipodal cost.
neg_w = out[..., 6] < 0  # w is the last quat component (xyzw layout, [3:7])
out[..., 3:7] = np.where(neg_w[..., None], -out[..., 3:7], out[..., 3:7])
```
…and re-preprocess. Keep the baseline ckpt out of any eval that uses
the canonicalized version (use a separate processed-dir).

## Development Environment

- **OS**: macOS (development / debugging), Linux GPU server (training, `ssh zju`).
- **Conda**: miniforge, environment name `vla`, Python 3.10.
- **Project path**: `~/Documents/le-wm/`.

```bash
conda activate vla
pip install -r requirements.txt   # transformers pin >=4.48,<5 matters for FAST

python train.py data=libero       # joint training over a directory of preprocessed .h5 (see run_all4.sh)

export STABLEWM_HOME=/path/to/storage
```

There is no pytest suite or CI pipeline, but three standalone scripts provide
end-to-end validation:

- **`smoke_test.py`** — full VLA model on a real preprocessed HDF5 sample: forward, backward, gradient-flow checks, T5 frozen-params check, autoregressive generation, VRAM profile at batch sizes 128/64/32/16.
- **`test_attn_mask.py`** — unit-level check that `_build_attn_mask()` produces the expected prefix-bidir + action-causal + PAD-isolated pattern on a hand-crafted example (lang PAD, action PAD, 4 real action tokens).
- **`check_fast_roundtrip.py`** — recomputes the GT anchor-relative chunk from the raw LIBERO HDF5 (using the same formulas as `preprocess_libero.py`), decodes the stored FAST tokens, and reports per-dim / per-step L1 error in normalized space (pure codec error) and physical space (what the robot sees).

## Baseline Implementation (frozen @ `7008f15`)

| File | Role |
|------|------|
| `jepa.py` | `JEPA` container. Methods: `encode`, `predict`, `encode_future_visual`, `predict_actions`, `predict_gripper_aux`, `_pool_visual_tokens`. `train()` is overridden to keep `lang_encoder` in `.eval()` mode even during training — blocks Lightning from re-enabling T5 dropout every step. Encoder construction in `train.py` uses `spt.backbone.utils.vit_hf` (LeWM ViT factory, configured via `cfg.encoder_scale`). |
| `module.py` | `ARPredictor`, `Block`, `Attention`, `MLP`, `FeedForward`, `MoTFeedForward`, FAST vocab constants (`BOS_TOKEN_ID=1024`, `EOS_TOKEN_ID=1025`, `PAD_TOKEN_ID=1026`, `TOTAL_VOCAB_SIZE=1027`, `ACTION_HEAD_SIZE=1026`). `SIGReg` is retained for future ablation but the baseline never instantiates it. `ARPredictor.__init__` accepts `use_state_prediction`, `visual_pool_grid`, `use_gripper_aux`, `gripper_chunk_size`, `use_mot` — all toggles default to off / 0 / False. |
| `train.py` | Hydra + Lightning + TensorBoard. `lejepa_forward` = L_CE always, plus optional `L_pred`, `L_sigreg`, `L_gripper_aux` based on config knobs. Demo-level train/val split (avoids leaking chunks from the same demo across splits) using `(fpath, demo_idx)` tuples as keys. Supports the `overfit_demo` 1-demo sanity mode. Calls `pl.seed_everything(cfg.seed, workers=True)` at the top of `run()` and passes `seed=cfg.seed` to `spt.Manager` so model init, dropout, DataLoader workers, and Manager's internal re-seeding all use the configured seed. |
| `libero_dataset.py` | Dual-view + 9D proprio + T5 tokenize + FAST tokens + always-on `gripper_seq` field. Supports `use_language=False`. Lazy per-worker HDF5 handles. Stable per-file `task_id` and `get_sampler_weights()` for the 3-level balanced sampler. |
| `preprocess_libero.py` | Anchor-relative chunks + stride=1 sliding window + FAST BPE. `normalize_proprio` is the single source of truth for the 9D layout and is re-used by `eval_libero.py::preprocess_obs`. |
| `eval_libero.py` | Closed-loop chunk execution. OSC `pos_scale` / `rot_scale` are read from the running controller at runtime (not hardcoded) with uniformity assertions. Supports both `_weights.ckpt` (Lightning) and `_object.ckpt` (pickled JEPA) formats, plus `--no-language` ablation. Auto-detects gripper-aux head and overrides `actions_phys[..., 6]` at inference. |
| `fast_utils.py` | `fast_decode` with pad/truncate fallback (the built-in FAST decoder silently zeros the chunk on length mismatch, which freezes the robot; our wrapper preserves as much of the signal as possible) + `denormalize_actions`. |
| `utils.py` | `ModelObjectCallBack` (per-step or per-epoch pickled model dump, top-K + latest-symlink invariant) + `TaskBalancedCEMetric` (aggregates per-task validation CE into `validate/ce_loss_taskbal`) + `EarlyProbeCallback` (optional online success-rate probe via `quick_probe_eval.py`) + `PeriodicPrintCallback` (terminal progress every N epochs for long overfit runs). |
| `pick_best_ckpt.py` | Reads `validate/ce_loss_taskbal` (or fallback `validate/total_loss_epoch`) from TB event files in `<ckpt-dir>/tb_logs/vla_baseline/version_*` and prints the top-K matching `_object.ckpt` paths as JSON. |
| `quick_probe_eval.py` | Stage-A online health probe — 1 episode × 1 task per suite × 4 suites in LIBERO; invoked optionally by `EarlyProbeCallback` at user-configured step(s). |
| `fit_tokenizer_all4.py` | Fits a single FAST tokenizer on the per-task-normalized chunks of all 40 LIBERO tasks (Option B) and writes it to `/Data/lyw/fast_tokenizer_all4/`. Called by `preprocess_all4.sh`. |
| `run_all4.sh` | End-to-end driver for 4-suite joint training: trains a single ckpt on the flat 40-task dir, picks best ckpt by `validate/ce_loss_taskbal`, runs per-suite eval. Optional grip-aux / V17 variants live in `run_all4_grip.sh` / `run_all4_visual17.sh`. |
| `run_object_baseline.sh` | Single-suite control: trains on libero_object only with the frozen-baseline config; used historically to disambiguate suite-level failures from joint-training distribution effects. |
| `preprocess_all4.sh` | Driver that runs `preprocess_libero.py` over all 40 raw LIBERO `.hdf5` (using the unified tokenizer from `fit_tokenizer_all4.py`) and writes them to `/Data/lyw/libero_processed_v5/libero_{spatial,object,goal,10}/`. |
| `launch_all4.sh` | tmux session wrapper around `run_all4.sh`. |
| `eval_videos.sh` + `launch_eval_videos.sh` | Eval-only re-run with `--save-videos` to populate `abl_videos/` for slide / paper rendering. |
| `smoke_test.py` | Full VLA-pipeline smoke test — instantiates the JEPA stack, runs forward/backward/inference on a real preprocessed HDF5 sample, profiles VRAM at multiple batch sizes. |
| `test_attn_mask.py` | Unit-level check on `_build_attn_mask` (does NOT cover the multi-token-visual or MoT extensions). |
| `check_fast_roundtrip.py` | FAST encode→decode roundtrip on stored GT chunks; reports per-dim normalized + physical L1 error. |
| `config/train/lewm.yaml` | Primary training config. Toggles for SP / SIGReg / gripper-aux / MoT live here under `loss`, `projector.norm_type`, and `predictor.use_mot`. Multi-token visual prefix uses the `visual_tokens.pool_grid` Hydra-CLI override (no default key — pass `+visual_tokens.pool_grid=4` to enable). |
| `config/train/overfit.yaml` | 1-demo pipeline sanity-check config. |
| `config/train/data/libero.yaml` | Dataset config + `use_language` ablation switch + proprio layout. |
| `requirements.txt` | Pinned deps. `transformers>=4.48,<5` is critical (earlier misses `TimmWrapperModel`; v5 breaks the FAST processor). |

The upstream LeWM `eval.py` (CEM/Adam latent-planning entry) and its
`config/eval/` directory have been removed (commit `d3b172a`) — the VLA
inference path only uses `eval_libero.py`.

## Hyperparameters (Frozen Baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Action chunk `H` | 20 raw steps | 1 s at LIBERO's 20 Hz control frequency |
| Sliding-window stride | 1 | Standard VLA practice; ~H× more samples than stride=H |
| FAST vocab | 1024 | BPE fitted on LIBERO-Spatial via `--fit-tokenizer`; unified 4-suite tokenizer at `/Data/lyw/fast_tokenizer_all4/` produced by `fit_tokenizer_all4.py`. |
| `max_action_tokens` | 80 | Observed max = 75 across all suites |
| `max_lang_tokens` | 25 | Covers the longest T5-tokenized LIBERO instruction |
| Embed dim `D` | 192 | ViT-Tiny hidden |
| Predictor (depth / heads / dim_head / mlp_dim) | 6 / 16 / 64 / 2048 | |
| Predictor dropout | 0.2 | Config overrides the class default of 0.1 |
| `emb_dropout` | 0.0 | |
| Proprio input dim | 9 | `ee_pos(3) + xyzw_quat(4) + gripper_raw(2)` |
| Projector | `MLP(hidden_dim→2048→D)` with `LayerNorm` | Explicitly **not** BatchNorm by default — avoids train/eval statistics mismatch. Switch via `cfg.projector.norm_type='batch'`; **required** when `cfg.loss.sigreg_weight > 0` (LeWM paper Section 3). See "Known caveats of switching to BatchNorm projector". |
| T5 variant | `t5-small`, fully frozen | 512-dim hidden, projected to D via trainable `lang_proj` |
| Optimizer | `AdamW`, lr=5e-5, weight_decay=0.05 | `LinearWarmupCosineAnnealingLR` built in `train.py` with `interval='step'`, `max_steps = cfg.trainer.max_steps` (if set, else `max_epochs × len(train_loader)`), `warmup_steps = min(2000, 0.02 × max_steps)`. |
| Batch size | 128 | |
| Max epochs | 100 (default) | Used only when `trainer.max_steps` is unset. Step-based runs (4-suite) override with `trainer.max_steps=100000`, `trainer.max_epochs=999` (sentinel so the epoch limit never fires), `trainer.val_check_interval=4000`, `+trainer.check_val_every_n_epoch=null`. **Checkpoint selection: pick lowest `validate/ce_loss_taskbal` step (4-suite) or lowest `validate/total_loss_epoch` epoch (single-suite legacy) — never the last step / epoch**, the baseline overfits before reaching the budget. |
| Precision | `bf16` mixed | |
| Gradient clip | 1.0 | |
| Label smoothing (CE) | 0.1 | Configurable via `cfg.label_smoothing`; 0 disables. CE floor ≈ 1.02 on a 1026-way vocab — read training curves against this floor, not zero. |
| Train / val split | 0.9 / 0.1 | **Demo-level** using `(fpath, demo_idx)` tuple keys (see `train.py`: reads `demo_idx` per file via `LiberoDataset._demo_ids` and partitions whole demos, not individual chunks; tuple key avoids cross-task demo-id collisions under 4-suite joint training). |
| Sampler | `WeightedRandomSampler` (4-suite) / `shuffle=True` (single-suite) | `LiberoDataset.get_sampler_weights()` produces per-sample weights `1 / (n_tasks * n_demos[task] * n_chunks[(task, demo)])` so marginal task / per-task-demo / per-demo-time distributions are all uniform. Skipped in overfit mode and when only one suite is in the hdf5_dir. |
| Logging | TensorBoard | `lightning.pytorch.loggers.TensorBoardLogger`, view with `tensorboard --logdir <run_dir>/tb_logs`. Per-task validation CE scalars (`validate/ce_loss/task_{N}`) are logged during validate only; `TaskBalancedCEMetric` aggregates them into `validate/ce_loss_taskbal` at epoch end. |
| Seed propagation | `cfg.seed=3072` | `pl.seed_everything(cfg.seed, workers=True)` is called at the top of `run()` BEFORE any model construction, and `seed=cfg.seed` is also passed to `spt.Manager(...)` so its internal `pl.seed_everything` call inside `__call__` aligns. Encoder / predictor / projector / `lang_proj` init, dropout draws, DataLoader workers, and demo-split shuffle all share the same seed. |
| `loss.pred_weight` | 0.0 | > 0 enables state prediction (3 STATE_QUERY tokens + 3 MSE losses, per-token MSE when `nv > 1`). Paper-default starting point: `1.0`. See "State-Prediction Extension". |
| `loss.sigreg_weight` | 0.0 | > 0 enables SIGReg anti-collapse on encoder CLS outputs. Paper default: `0.1` (Section 3). Forces `projector.norm_type='batch'`. |
| `loss.gripper_aux_weight` | 0.0 | > 0 enables the gripper bypass head — direct (H,)-regression MLP off the BOS hidden state. `eval_libero.py` auto-overrides `actions_phys[..., 6]`. See "Gripper-auxiliary head". |
| `predictor.use_mot` | `false` | `true` activates Mixture-of-Transformers per-modality FFN (3 modalities with SP, 2 without). See "Mixture-of-Transformers (MoT)". |
| `visual_tokens.pool_grid` | (unset, equals 0) | `G > 0` activates V17-style multi-token visual prefix with `1 + G²` tokens per view. Pass via Hydra CLI override `+visual_tokens.pool_grid=4`. See "Multi-token Visual Prefix (V17)". |
| `loss.sigreg.kwargs` | `knots=17, num_proj=1024` | Paper defaults; safe to leave alone (paper Fig. 15: barely sensitive to either). |

## Ablation Modes

All toggles are wired end-to-end (preprocess / train / eval) and
baseline-compatible. Any subset can be combined in one run.

### 1. `use_language=False` (no-language)

Drops T5 entirely. Prefix becomes `[z_agent, z_hand, z_proprio, BOS, ...]`;
neither `lang_encoder` nor `lang_proj` is constructed. The `_build_attn_mask`
and `_build_generate_mask` paths both branch on `n_lang > 0`.

```bash
# Train
python train.py data=libero data.dataset.use_language=false

# Eval (flag must match training-time setting; checkpoint loader errors out on mismatch)
python eval_libero.py --no-language ...
```

### 2. `overfit_demo=N` (1-demo pipeline sanity check)

Takes every chunk whose `demo_idx == N` and uses the same subset for train
and val. If the model cannot drive `ce_loss < 0.1` and
`token_accuracy > 0.95` after a few hundred epochs, the bug is in the
pipeline (data → forward → loss → backward), **not** capacity or
optimization.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=overfit \
    data.dataset.hdf5_dir=/path/to/single_task_dir/ \
    subdir=overfit_sanity
```

`config/train/overfit.yaml`: batch=8, lr=1e-3, weight_decay=0, dropout=0,
`max_epochs=2000`, single GPU, periodic terminal progress via
`PeriodicPrintCallback`.

### 3. `label_smoothing=0` (no-regularization)

```bash
python train.py data=libero \
    label_smoothing=0 \
    predictor.dropout=0 \
    optimizer.weight_decay=0
```

Empirically the regularized baseline (`label_smoothing=0.1` + `dropout=0.2`
+ `weight_decay=0.05`) outperforms the no-reg run, and within the no-reg
run the best-val checkpoint beats the last checkpoint by a wide margin —
which is why we take best-val, not last.

### 4. State-prediction + SIGReg (LeWM-style additive method)

Enables the State-Prediction Extension described in the dedicated section
above. Three STATE_QUERY tokens are appended at training time, three
MSE losses are added (per-token under V17, single-token under CLS-only),
and SIGReg is enforced on encoder CLS outputs.

```bash
# (One-time) re-preprocess to populate image_*_future + proprio_future
python preprocess_libero.py ...        # see "How to enable" above

# Train with paper defaults
python train.py data=libero \
    loss.pred_weight=1.0 \
    loss.sigreg_weight=0.1 \
    projector.norm_type=batch          # required when sigreg_weight > 0

# Evaluate via the per-step object checkpoint (eval_libero.py is unchanged
# but cannot reconstruct the SP architecture from `_weights.ckpt` alone)
python eval_libero.py \
    --checkpoint /path/to/lewm_step_{N}_object.ckpt \
    --tokenizer /path/to/fast_tokenizer \
    --processed-dir /path/to/libero_processed/<suite>/ \
    --suite libero_spatial
```

Setting only `pred_weight > 0` (without SIGReg) trains state prediction
without anti-collapse. Setting only `sigreg_weight > 0` (without state
prediction) regularizes the current encoder outputs to isotropic Gaussian
with no future-frame loss; SIGReg's input degenerates to 2 streams in
that case (`z_ag_t`, `z_hd_t`).

### 5. Gripper auxiliary head

`loss.gripper_aux_weight > 0` activates the gripper-aux bypass head.
The head is built only when this is enabled; `eval_libero.py`
auto-detects the head at load time. The dataset always supplies
`gripper_seq`, so no preprocessing change is required.

```bash
# Train baseline + grip
python train.py data=libero loss.gripper_aux_weight=1.0

# Combine with SP + SIGReg (4-suite run_all4_grip.sh recipe)
python train.py data=libero \
    loss.pred_weight=1.0 loss.sigreg_weight=0.1 loss.gripper_aux_weight=1.0 \
    projector.norm_type=batch
```

### 6. Multi-token visual prefix (V17)

`visual_tokens.pool_grid > 0` activates the multi-token visual prefix.
`G=4` (= V17) is the only configuration that has been exercised
end-to-end; other values should work but are untested. Multi-token can
be combined with SP / SIGReg / MoT / gripper-aux in any combination
that the train.py guards allow. Use Hydra `+` prefix because
`visual_tokens` has no defaults in `lewm.yaml`:

```bash
python train.py data=libero \
    +visual_tokens.pool_grid=4 \
    loss.pred_weight=1.0 loss.sigreg_weight=0.1 \
    projector.norm_type=batch
```

### 7. Mixture-of-Transformers (MoT)

`predictor.use_mot=true` switches every transformer block's FFN to a
per-modality MoT routing. Attention stays shared. Backward-compatible
with the frozen baseline only when `use_mot=false`; old `_object.ckpt`
files do not contain the MoT expert weights and will mismatch.

```bash
python train.py data=libero \
    predictor.use_mot=true \
    loss.pred_weight=1.0 loss.sigreg_weight=0.1 \
    projector.norm_type=batch
```

### 8. 4-suite joint training

Drive `data.dataset.hdf5_dir` at a flat directory of 40 preprocessed
`.h5` (one per task, symlinked from the per-suite preprocessed dirs).
`run_all4.sh` is the canonical recipe — it sets the step-based trainer
budget, uses the unified tokenizer, runs training, picks best ckpt by
`validate/ce_loss_taskbal`, then loops eval over all 4 suites. The
sampler / per-task val CE / task-balanced ckpt selection all activate
automatically because the dataset has more than one task in the
hdf5_dir.

```bash
bash run_all4.sh          # baseline arm (sp_sigreg), seed=3072 by default
ARM=baseline bash run_all4.sh
ARM=sp_sigreg SEED=2024 bash run_all4.sh
```

Variants:
- `run_all4_grip.sh` — adds `loss.gripper_aux_weight=1.0` on top.
- `run_all4_visual17.sh` — adds `+visual_tokens.pool_grid=4` on top.

## External Libraries

- **`stable-worldmodel` (`swm`)** — env wrappers, `swm.data.utils.get_cache_dir()` for checkpoint locations.
- **`stable-pretraining` (`spt`)** — training orchestration (`spt.Module`, `spt.Manager`, `spt.data.DataModule`), ViT backbone factory (`spt.backbone.utils.vit_hf`).
- **`transformers`** — `AutoProcessor` (FAST tokenizer, `trust_remote_code=True`), `T5EncoderModel` + `T5Tokenizer`.
- **`lightning.pytorch.loggers.TensorBoardLogger`** — scalar logging.

## Key Details

- **Data layout**: training HDF5 under `${STABLEWM_HOME}/libero/`; per-suite preprocessed output under `${DATA_ROOT}/libero_processed/<suite>/` (one `.h5` per task — the **directory** is what `LiberoDataset` consumes for joint training, not individual files). Frozen baseline checkpoint lives at `/Data/lyw/checkpoints/multitask_ln_100ep/lewm_weights.ckpt` (single ckpt joint-trained on the suite). 4-suite joint training uses a flat-symlink dir `${DATA_ROOT}/libero_processed_v5/all4_flat/` so `LiberoDataset` can glob all 40 `.h5` from one root — see `run_all4.sh`.
- **FAST tokens**: variable-length int32 (`h5py.vlen_dtype`) per sample; each preprocessed HDF5 stores its own `action_low` / `action_high` percentile bounds (for inverse normalization at eval time), `chunk_size`, `chunk_stride`, and `language_instruction` in the file attrs. The unified 4-suite FAST tokenizer (fit by `fit_tokenizer_all4.py`) lives at `/Data/lyw/fast_tokenizer_all4/`; all 4-suite runs reuse it via `--load-tokenizer`.
- **Device handling**: no hardcoded `cuda` — tensor device is inferred from inputs; the caller moves `JEPA` to the target device.
- **Checkpoint formats**:
  - `lewm_weights.ckpt` — Lightning state_dict saved by `spt.Manager`; loaded via `load_checkpoint()` which strips the `model.` prefix and skips `lang_encoder.*` keys (T5 is reloaded from pretrained). Rejected at load time when the saved state has SP / BN-projector / gripper-aux keys — those configurations must use `_object.ckpt`.
  - `lewm_step_{N}_object.ckpt` / `lewm_epoch_{N}_object.ckpt` — `torch.save(model)` pickle from `ModelObjectCallBack`; loaded directly via `torch.load(..., weights_only=False)`. Step-based ckpts are produced when `trainer.max_steps` is set and `step_interval` is forwarded into the callback; the callback also maintains a `lewm_latest_object.ckpt` symlink to the most recent surviving step.
- **Checkpoint selection convention**: under 4-suite step-based training, pick the step with the lowest `validate/ce_loss_taskbal` (the task-balanced CE aggregated by `TaskBalancedCEMetric`). Under single-suite epoch-based runs, pick the lowest `validate/total_loss_epoch`. Never the last step / epoch — the baseline consistently overfits before the budget is reached.
- **Seed propagation**: `cfg.seed` is enforced in three places — `pl.seed_everything(cfg.seed, workers=True)` at the top of `run()`, `seed=cfg.seed` passed to `spt.Manager(...)`, and `torch.Generator().manual_seed(cfg.seed)` for the demo-split shuffle + WeightedRandomSampler. Without all three, `spt.Manager.__call__` falls back to seed 0 and model init drifts.
