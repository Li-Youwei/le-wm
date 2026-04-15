# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLA (Vision-Language-Action) baseline built on top of the LeWorldModel (LeWM) codebase. The model takes dual-view images, proprioceptive state, and a language instruction as input, and autoregressively predicts FAST action tokens. Trained on LIBERO-Spatial (single-task per model).

**This is a pure action prediction baseline, NOT original LeWM training, NOT Fast-WAM.**
- No `L_pred` (next-embedding prediction loss) — this is LeWM's core training signal, we explicitly remove it
- No `L_sigreg` (anti-collapse regularizer) — only needed when encoder is trained via prediction loss
- No `rollout()`, `CEM`, or latent planning — actions are directly generated, not searched
- No video co-training — Fast-WAM keeps video prediction during training; we don't
- **The only training loss is `L_CE` (cross-entropy on FAST action tokens)**

Do NOT re-add any of these components unless explicitly instructed. If you find yourself writing `SIGReg`, `pred_loss`, `state_query`, `criterion`, `rollout`, or `CEM` in new code, stop — you are reverting to the old architecture.

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

Training-time input sequence:
```
[l_1, l_2, ..., l_n, z_agent, z_hand, z_proprio, BOS, T_1^GT, T_2^GT, ..., T_k^GT, PAD, ..., PAD]
 ←── perception prefix (bidirectional) ──→       ←── action tokens (causal) + padding ──→
```

Inference:
```
[l_1, ..., l_n, z_agent, z_hand, z_proprio, BOS] → autoregressively generate T_1, ..., T_k, <EOS>
→ FAST decode → continuous action chunk (H=20 steps × 7 dims, anchor-relative)
→ closed-loop execution (see "Closed-loop chunk execution" below)
```

**Inference decoding rules (must be followed exactly):**
1. **Stop generation** when either `<EOS>` (id=1025) is sampled OR `max_action_tokens` tokens have been generated, whichever comes first.
2. **Post-process before FAST decode**: strip `<BOS>` from the beginning, truncate at `<EOS>` (discard EOS itself and anything after), discard any trailing tokens beyond `max_action_tokens`.
3. **Only token IDs 0..1023** (real FAST vocab) go into the FAST decoder. Never pass BOS/EOS/PAD to FAST decode.
4. Note: `action_head` outputs 1026 logits (0..1025), so PAD (id=1026) **cannot** be sampled by design — do NOT add extra masking for PAD during generation.

Where:
- `l_1...l_n`: language token embeddings from frozen T5-small (variable length, typically 5-15 tokens)
- `z_agent`: CLS token from ViT encoding of agentview image
- `z_hand`: CLS token from ViT encoding of eye-in-hand image
- `z_proprio`: MLP encoding of 9d proprioceptive state: base-frame EE position(3) + xyzw quaternion(4) + raw gripper finger positions(2)
- `BOS`: beginning-of-action token (id=1024)
- `T_1...T_k`: FAST action tokens (discrete, vocab=0..1023, variable-length. With H=20, token count is TBD — must be determined by preprocessing. Previous H=10 produced ~20-40 tokens; H=20 will likely produce significantly more)
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
L_CE = F.cross_entropy(action_logits.reshape(-1, 1026), targets.reshape(-1), ignore_index=-100)
```

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

### Scope: What This Baseline Does NOT Include

- **No world model prediction**: no STATE_QUERY, no L_pred, no future frame encoding
- **No SIGReg**: no anti-collapse regularization
- **No CEM planning**: actions are directly generated via autoregressive decoding
- **No multi-task training**: one model per LIBERO-Spatial task (language is still used as input, but all samples within a task share the same instruction)
- **No history context**: single-frame input per view. The original LeWM uses `history_size=3` — do NOT copy this. Each observation is a single timestep (one agentview + one eye_in_hand + one proprio reading). Multi-frame history is a potential follow-up.

## Development Environment

- **OS**: macOS (development/debugging), Linux GPU server (training)
- **Conda**: miniforge, environment name `vla`, Python 3.10
- **Project path**: `~/Documents/le-wm/`

```bash
# Activate environment
conda activate vla

# Install dependencies
pip install stable-worldmodel[train,env]
pip install transformers  # for FAST tokenizer + T5-small

# Train
python train.py data=libero

# Data location
export STABLEWM_HOME=/path/to/storage
```

There are no tests, linting, or CI pipelines in this repository.

## Architecture (Original LeWM — for reference)

Four Python files implement the original core:

- **`jepa.py`** — `JEPA` class. Key methods: `encode()`, `predict()`, `rollout()`, `criterion()` (planning-time latent cost, NOT training loss), `get_cost()`.
- **`module.py`** — Building blocks: `SIGReg`, `Attention`, `Block`, `ConditionalBlock` (AdaLN-zero), `Transformer`, `ARPredictor`, `Embedder`, `MLP`. Also contains the already-added `UnifiedPredictor` from the previous phase.
- **`train.py`** — Training entry point. Contains `lejepa_forward()` (training step and loss). Hydra config, PyTorch Lightning.
- **`eval.py`** — Evaluation/planning (original CEM-based, to be replaced).
- **`utils.py`** — Image preprocessing, StandardScaler, checkpoint callback.

### Already Modified Files (from previous phase)

The following modifications were made in the first iteration (unified predictor with world model co-training). These will be **further modified** for the new baseline:

- `module.py`: Added `UnifiedPredictor`, modified `Attention.forward` and `Block.forward` to accept `attn_mask`
- `jepa.py`: Modified `JEPA.__init__`, `encode()`, `predict()`, added `predict_actions()`
- `train.py`: Modified `lejepa_forward()`, model assembly, added `LiberoDataset` import
- `libero_dataset.py`: New file for LIBERO HDF5 data loading
- `preprocess_libero.py`: New file for data preprocessing + FAST tokenization

## Modification Plan (New Baseline)

### What Changes from Previous Iteration

| Component | Previous | New Baseline |
|-----------|----------|--------------|
| Visual input | Single view (agentview only) | **Dual view** (agentview + eye_in_hand), shared ViT |
| Proprioception | None | **EE pose + ori + gripper (8d) → MLP → D** |
| Language | None | **T5-small frozen → prefix tokens** |
| STATE_QUERY | Yes (at end of sequence) | **Removed** |
| L_pred (MSE) | Yes | **Removed** |
| L_sigreg | Yes | **Removed** |
| Loss | L_CE + α*L_pred + λ*L_sigreg | **L_CE only** |
| Attention mask | Pure causal + padding | **Prefix-bidirectional + action-causal + padding** |
| Future frame | Encoded for L_pred target | **Not needed** |
| Logging | WandB | **TensorBoard** |

### File-by-File Modification Plan

#### `module.py` — Modify UnifiedPredictor

**Modify `UnifiedPredictor`:**
- Remove `state_query` parameter and all state prediction logic
- Remove state prediction output from `forward()` — return only `action_logits`
- Add `proprio_encoder = MLP(9, hidden, embed_dim)` for proprioceptive state encoding (9d = ee_pos(3) + xyzw_quat(4) + gripper_raw(2))
- Change `type_embedding` from `nn.Embedding(3, D)` to `nn.Embedding(4, D)`: 0=language, 1=visual, 2=proprioception, 3=action
- Update `pos_embedding` max_seq_len: `max_lang_tokens + 3 + 1 + max_action_tokens` (lang + z_ag/z_hd/z_pr + BOS + tokens). `max_lang_tokens` must be determined by scanning all LIBERO-Spatial instructions through T5 tokenizer first — do NOT hardcode without checking.
- Modify `forward()` signature: `forward(z_agent, z_hand, z_proprio_raw, lang_embeds, lang_lengths, action_tokens, action_lengths)`
  1. Encode proprio: `z_proprio = self.proprio_encoder(z_proprio_raw)` → (B, D)
  2. Build prefix: concat `[lang_embeds + type[0], z_agent + type[1], z_hand + type[1], z_proprio + type[2]]`
  3. Build action part: `[BOS + type[3], T_1 + type[3], ..., PAD...]`
  4. Concat prefix + action → full sequence
  5. Add pos_embedding
  6. Build hybrid attention mask (prefix-bidir + action-causal + non-padding)
  7. Run through transformer blocks with attn_mask
  8. Extract action_logits from BOS+T positions → through action_head
  9. Return action_logits only
- Modify `generate()`: start with `[prefix..., BOS]`, autoregressive from there. Prefix uses bidirectional attention, action tokens use causal.

**Keep unchanged:** `Attention` (already supports attn_mask), `Block` (already supports attn_mask), `SIGReg` (keep for potential future ablation), `MLP`, etc.

#### `jepa.py` — Modify JEPA class

**Modify `__init__`:**
- Add `lang_encoder` (T5EncoderModel, frozen)
- Add `lang_proj = nn.Linear(T5_hidden_dim, embed_dim)` to project T5 outputs to D
- Remove `pred_proj` (no state prediction)

**Modify `encode()`:**
- Encode both views through shared ViT → `z_agent`, `z_hand`
- Encode language via frozen T5 → project via `lang_proj` → `lang_embeds`
- Return `z_agent`, `z_hand`, `lang_embeds`, `lang_lengths`
- **Proprio is NOT encoded in `encode()`.** Raw 9d proprio vector is passed directly from the batch to `predict()`, which forwards it to `UnifiedPredictor` where the MLP encoding happens. This keeps the MLP inside the predictor module.

**Modify `predict()`:**
- New signature: `predict(z_agent, z_hand, proprio, lang_embeds, lang_lengths, action_tokens, action_lengths)`
- Returns `action_logits` only

**Modify `predict_actions()`:**
- Include full prefix (lang + visual + proprio) in initial sequence for autoregressive generation

#### `train.py` — Modify training loop

**Modify `lejepa_forward()`:**
- Read from batch: `pixels_agent`, `pixels_hand`, `proprio`, `language`, `fast_tokens`, `fast_lengths`
- Encode: `z_agent, z_hand, lang_embeds, lang_lengths = model.encode(...)`
- Call: `action_logits = model.predict(z_agent, z_hand, proprio, lang_embeds, lang_lengths, fast_tokens, fast_lengths)`
- Build CE targets: shape `(B, 1 + max_action_tokens)` — only action-zone positions. BOS→T_1, T_1→T_2, ..., T_k→EOS, PAD→-100. No prefix positions in this tensor (see "Targets and Loss" section).
- `L_total = L_CE`

**Remove:** L_pred computation, L_sigreg computation, future frame encoding, SIGReg instance

**TensorBoard logging:**
- Replace WandB with TensorBoard. Use PyTorch Lightning's built-in `TensorBoardLogger`:
  ```python
  from pytorch_lightning.loggers import TensorBoardLogger
  logger = TensorBoardLogger("tb_logs", name="vla_baseline")
  ```
- Log at minimum these scalars per step:
  - `train/ce_loss` — action token cross-entropy (the only loss)
  - `train/total_loss` — same as ce_loss in baseline (but separate key for when ablations add more losses)
  - `train/lr` — learning rate (for schedule debugging)
  - `train/token_accuracy` — fraction of correctly predicted action tokens (excluding PAD/prefix, useful diagnostic)
- Log per epoch:
  - `epoch/ce_loss` — epoch-averaged CE loss
- View with: `tensorboard --logdir tb_logs`

#### `libero_dataset.py` — Update dataset

- Add second image view (`eye_in_hand_rgb`)
- Add proprioception: `obs/ee_pos` (3d) + `robot_states[5:9]` (4d xyzw quaternion) + `obs/gripper_states[0:2]` (2d raw, no averaging) = 9d vector. `obs/ee_ori` is axis-angle (not Euler as earlier versions of this doc incorrectly said), but we continue to use the quaternion form for proprio orientation.
- Add language instruction: extract from `data.attrs["problem_info"]` (JSON string, parse and read the `"language_instruction"` key). Example: `"pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"`. Do NOT use the HDF5 filename, task ID, or internal key name as the language input — these are formatted strings (e.g. `KITCHEN_SCENE10_close_the_top_drawer...`) that T5 was not trained on, and will degrade language encoding quality.
- Remove `image_future` (no longer needed)

#### `preprocess_libero.py` — Update preprocessing

- Add extraction of `eye_in_hand_rgb` alongside `agentview_rgb`
- Add extraction of proprioceptive state (9d): `obs/ee_pos` (3) + `robot_states[5:9]` xyzw quaternion (4) + `obs/gripper_states[0:2]` raw (2). Never average or abs the gripper fingers — the signal collapses to zero or halves in range.
- Add language instruction per task: parse `data.attrs["problem_info"]` JSON, extract `"language_instruction"` value. Store one canonical instruction string per task in the output HDF5.
- Remove `image_future` from output
- **FAST preprocessing rule**: actions MUST be quantile-normalized to [-1, 1] (1st/99th percentile per dim) BEFORE DCT/BPE (FAST paper Section V-B). Verify this is correctly implemented in the preprocessing script. Baseline uses `--fit-tokenizer` to train a LIBERO-Spatial-specific BPE vocabulary (not FAST+ universal).
- **Token length assertion (CRITICAL)**: After preprocessing the full training set, `assert max(fast_length) <= max_action_tokens`. If this fails, the script must raise an error and print the actual max length — then `max_action_tokens` and `pos_embedding` `max_seq_len` must be increased accordingly. Do NOT silently truncate sequences that exceed the limit. Silent truncation causes training to appear normal while quietly destroying action quality — this is the hardest kind of bug to find.

#### `eval.py` — Rewrite for direct inference

**The original `eval.py` uses CEM/Adam rollout planning in latent space. This is completely replaced.**

New evaluation flow:
1. Load trained checkpoint
2. For each test episode:
   - Get observation (two views + proprio) + language instruction
   - Encode all inputs → prefix tokens
   - Call `model.predict_actions(prefix)` → autoregressive FAST token generation
   - FAST decode tokens → continuous action chunk (H=20 steps × 7d)
   - Execute action chunk in LIBERO environment
   - Repeat until episode ends or max steps
3. Report success rate per task

Do NOT reuse any of the original `eval.py` planning logic (CEM, Adam solver, latent rollout, goal embedding). The evaluation is now a simple forward-pass policy, not a search-based planner.

### Hyperparameters (Starting Point)

| Parameter | Value | Source |
|-----------|-------|--------|
| Action chunk H | 20 steps (1 sec @ 20Hz) | FAST paper recommends 1 sec; LIBERO control_freq=20Hz |
| FAST vocab size | 1024 | FAST paper default |
| max_action_tokens | TBD after preprocessing | H=20 produces longer token sequences than H=10. Run preprocessing on full dataset, take max observed length + small margin. The token length assertion in preprocessing will catch if this is set too low. |
| max_lang_tokens | Determine by scanning dataset | Tokenize all LIBERO-Spatial instructions with T5 tokenizer, take max length + small margin. Do NOT hardcode 20 without checking. |
| Embed dim D | 192 | LeWM default (ViT-Tiny) |
| Predictor depth | 6 layers | LeWM default |
| Predictor heads | 16 | LeWM default |
| T5 variant | t5-small (60M params, frozen) | Standard for VLA |
| T5 hidden dim | 512 | t5-small config |
| Proprio input dim | 9 (pos3 + xyzw_quat4 + grip_raw2) | LIBERO EE state |
| Learning rate | From LeWM config | Keep original schedule |
| Batch size | 128 | LeWM default |
| Logging | TensorBoard | Advisor requirement |

### Key Constraints

- **Only L_CE for the baseline.** No world model prediction loss, no SIGReg. These may be added back as ablation experiments later.
- **T5-small is fully frozen.** Operational rules:
  1. `lang_encoder.eval()` — always in eval mode, even during training (prevents dropout from firing)
  2. `for p in lang_encoder.parameters(): p.requires_grad_(False)` — no gradient computation
  3. Wrap forward pass in `with torch.no_grad():` — saves GPU memory by not storing activations
  4. Only `lang_proj` (the linear projection layer) is trained
- **ViT encoder is shared** between agentview and eye_in_hand. Both views go through the same encoder with the same weights.
- **Proprioception MLP is inside UnifiedPredictor**, not a separate encoder in JEPA. It takes raw 9d input and outputs D-dim embedding.
- **Attention mask is NOT pure causal.** It is a hybrid prefix-bidirectional + action-causal mask. Must be constructed explicitly as a `(B, 1, L, L)` bool tensor.
- **Config naming**: Use `chunk_horizon_raw_steps = 20` for the action chunk length (1 second at LIBERO's 20Hz). Do NOT overload the original `frameskip` variable.
- **Single-task training**: One model per LIBERO-Spatial task. Language input is the same for all samples within a task, but the architecture supports multi-task for future extension.
- **`criterion()` in original `jepa.py` is the planning-time latent cost (for CEM/MPC), NOT the training loss.** The training loss is in `lejepa_forward()` in `train.py`.

## External Libraries

- **`stable-worldmodel` (`swm`)** — Environment wrappers, evaluation APIs
- **`stable-pretraining` (`spt`)** — Training orchestration, dataset loading, ViT backbone
- **`transformers`** — FAST tokenizer (`AutoProcessor`), T5-small encoder (`T5EncoderModel`)
- **`torch.utils.tensorboard`** — Loss curve visualization

## Configuration

Hydra YAML files under `config/`:
- `config/train/lewm.yaml` — Main training config
- `config/train/data/libero.yaml` — LIBERO dataset config
- `config/eval/*.yaml` — Evaluation configs

## Key Details

- **Data format**: HDF5 files stored under `$STABLEWM_HOME`
- **FAST tokens**: Pre-computed during preprocessing, stored as variable-length integer sequences in HDF5
- **Device handling**: Uses `proj.device` (not hardcoded `cuda`)
- **Logging**: TensorBoard (not WandB)
