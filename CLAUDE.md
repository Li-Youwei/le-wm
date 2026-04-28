# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLA (Vision-Language-Action) on top of the LeWorldModel (LeWM) codebase.
The model takes dual-view images, proprioceptive state, and a language
instruction as input, and autoregressively predicts FAST action tokens.

The repository hosts **two coexisting variants**, controlled by config
loss weights:

1. **Frozen baseline** (`pred_weight = sigreg_weight = 0`, default in
   `config/train/lewm.yaml`). Loss = `L_CE` only. This is the
   reproducibility target frozen at commit `7008f15`. **Joint-trained on
   the 10 LIBERO-Spatial tasks** (single checkpoint covering all tasks;
   the model disambiguates them via the language instruction). Achieved
   65% average success across 10 tasks at evaluation.
2. **State-prediction extension** (`pred_weight > 0`, optionally
   `sigreg_weight > 0`). Adds three STATE_QUERY tokens (`Q_ag`, `Q_hd`,
   `Q_pr`) to the predictor sequence at training time, which read out
   future-frame visual latents and raw future proprio. Optionally adds
   the LeWM SIGReg anti-collapse regularizer on encoder outputs. See
   "State-Prediction Extension" below.

**Inference is identical for both variants.** STATE_QUERY tokens are
training-only — `generate()` and `eval_libero.py` are unchanged.

Things this repo still does NOT do (do not re-add without explicit
instruction):
- `criterion()` planning-time latent cost — original LeWM CEM/MPC, never
  used by the VLA inference path
- `rollout()` — multi-step latent rollouts for planning
- `CEM` — Cross-Entropy Method action search
- Video co-training (Fast-WAM)

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

### Scope: What the Frozen Baseline Does NOT Include

These bullets describe the **frozen baseline** (`pred_weight = sigreg_weight
= 0`). The state-prediction extension adds some of them back — see the
next section.

- **No world model prediction (in baseline)**: no STATE_QUERY, no L_pred, no future frame encoding
- **No SIGReg (in baseline)**: class kept in `module.py` but only instantiated when `sigreg_weight > 0`
- **No CEM planning (ever)**: actions are directly generated via autoregressive decoding, not searched
- **Joint training across all 10 LIBERO-Spatial tasks** (one checkpoint, ~65 K chunks total). The language instruction is the sole per-task disambiguation signal — chunks from different tasks have different `language_instruction` strings tokenized by frozen T5. The frozen baseline at `7008f15` was trained this way (see `/Data/lyw/checkpoints/multitask_ln_100ep/config.yaml` on the GPU server: `hdf5_dir: /Data/lyw/libero_processed/libero_spatial` points at the suite directory, so `LiberoDataset` globs all 10 `.h5` files).
- **Note on `run_all_suites.sh`**: this script runs a per-task training loop (one checkpoint per task) across all 4 suites. The frozen baseline did **not** use this script — `run_all_suites.sh` is an alternative pipeline that exists but was never the basis for the 65 % number. Do not confuse the two.
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

| Head | Input | Output | Target |
|------|-------|--------|--------|
| `state_pred_head_ag` | `h_{Q_ag} ∈ ℝ^D` | `pred_ag ∈ ℝ^D` | `z_ag_{t+H}` (future agentview latent, **NOT detached**) |
| `state_pred_head_hd` | `h_{Q_hd} ∈ ℝ^D` | `pred_hd ∈ ℝ^D` | `z_hd_{t+H}` (future hand-cam latent, **NOT detached**) |
| `state_pred_head_pr` | `h_{Q_pr} ∈ ℝ^D` | `pred_pr ∈ ℝ^9` | re-normalized `proprio_{t+H}` (9d, same `normalize_proprio` treatment as the predictor input) |

The visual heads' output dim matches the encoder projector's output
(`embed_dim`) so MSE is computed in the same latent space the encoder
puts `z_*_{t+H}` into. The proprio head terminates in `proprio_dim` (=9)
and predicts the **re-normalized 9d proprio vector at t+H** (re-normalize
= unit-length quaternion at `[3:7]`, ee_pos and gripper passthrough; same
`preprocess_libero.py::normalize_proprio` applied to current and future
proprio symmetrically — `proprio` and `proprio_future` use the same
treatment so the MSE target's scale matches the predictor input's scale).

### `L_pred` (state-prediction loss)

```
L_pred = MSE(pred_ag, z_ag_{t+H})        # gradients flow through both branches
       + MSE(pred_hd, z_hd_{t+H})        # gradients flow through both branches
       + MSE(pred_pr, proprio_{t+H})     # raw 9d MSE
```

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
sigreg_input = torch.stack([z_ag, z_hd, z_ag_future, z_hd_future], dim=0)
                                              # (T=4, B, D), encoder-only
L_sigreg = sigreg(sigreg_input)
```

**Predictor outputs** (`pred_ag`, `pred_hd`, `pred_pr`) are intentionally
NOT included in the SIGReg input stack. They are pulled toward
encoder outputs via `L_pred`, and the encoder outputs are constrained by
SIGReg, so predictor outputs are indirectly anchored to an isotropic
Gaussian distribution.

When `pred_weight = 0` but `sigreg_weight > 0` (rare configuration), the
input degenerates to `(T=2, B, D)` with current views only, since
`z_*_future` are not encoded.

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
| `module.py::ARPredictor` | Adds `state_query_embeddings` (3,D), three prediction heads, an extended attention mask, and an extra type embedding row (5 instead of 4). `forward()` returns `(action_logits, pred_ag, pred_hd, pred_pr)` instead of just `action_logits`. `generate()` is **unchanged** — STATE_QUERY tokens are training-only. |
| `jepa.py::JEPA` | Adds `encode_future_visual()` that runs the SHARED ViT + projector on the t+H frames with NO `.detach()`. |
| `train.py::lejepa_forward` | Computes `L_pred` (3 MSE) when `pred_weight > 0`, computes `L_sigreg` on the encoder-only 4-stream stack when `sigreg_weight > 0`, sums into `L_total`. Validates `projector.norm_type == 'batch'` when SIGReg is on. |
| `eval_libero.py` | Inference flow `[lang, z_ag, z_hd, z_pr, BOS] → AR decode → FAST` is unchanged. The only delta is `load_checkpoint()` now hard-rejects non-baseline `_weights.ckpt` payloads (SP keys OR BatchNorm projector running stats) with a clear message pointing at `_object.ckpt`. Both SP-trained and "SIGReg-only no SP" ablation checkpoints must be evaluated via the per-epoch `_object.ckpt` produced by `ModelObjectCallBack`. |

### How to enable

```bash
# Re-preprocess (one-time, populates image_*_future + proprio_future)
python preprocess_libero.py --input <raw.hdf5> --output <out.h5> ...

# Train with paper-default weights
python train.py data=libero \
    loss.pred_weight=1.0 \
    loss.sigreg_weight=0.1 \
    projector.norm_type=batch        # required by SIGReg

# Eval (use the per-epoch object checkpoint)
python eval_libero.py \
    --checkpoint /path/to/lewm_epoch_{N}_object.ckpt \
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

python train.py data=libero       # train one task (repeat per task via run_all_suites.sh)

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
| `jepa.py` | `JEPA` container. Methods: `encode`, `predict`, `predict_actions`. `train()` is overridden to keep `lang_encoder` in `.eval()` mode even during training — blocks Lightning from re-enabling T5 dropout every step. |
| `module.py` | `ARPredictor`, `Block`, `Attention`, `MLP`, FAST vocab constants (`BOS_TOKEN_ID=1024`, `EOS_TOKEN_ID=1025`, `PAD_TOKEN_ID=1026`, `TOTAL_VOCAB_SIZE=1027`, `ACTION_HEAD_SIZE=1026`). `SIGReg` is retained for future ablation but the baseline never instantiates it. |
| `train.py` | Hydra + Lightning + TensorBoard. `lejepa_forward` = L_CE only. Demo-level train/val split (avoids leaking chunks from the same demo across splits). Supports the `overfit_demo` 1-demo sanity mode. |
| `libero_dataset.py` | Dual-view + 9D proprio + T5 tokenize + FAST tokens. Supports `use_language=False`. Lazy per-worker HDF5 handles. |
| `preprocess_libero.py` | Anchor-relative chunks + stride=1 sliding window + FAST BPE. `normalize_proprio` is the single source of truth for the 9D layout and is re-used by `eval_libero.py::preprocess_obs`. |
| `eval_libero.py` | Closed-loop chunk execution. OSC `pos_scale` / `rot_scale` are read from the running controller at runtime (not hardcoded) with uniformity assertions. Supports both `_weights.ckpt` (Lightning) and `_object.ckpt` (pickled JEPA) formats, plus `--no-language` ablation. |
| `fast_utils.py` | `fast_decode` with pad/truncate fallback (the built-in FAST decoder silently zeros the chunk on length mismatch, which freezes the robot; our wrapper preserves as much of the signal as possible) + `denormalize_actions`. |
| `run_all_suites.sh` | End-to-end driver: preprocess → per-task train (100 epochs) → per-task eval (20 episodes) across all 4 LIBERO suites. Bootstraps the FAST tokenizer on the first task when none exists, then reuses it for the rest. |
| `config/train/lewm.yaml` | Baseline training config with regularization defaults. |
| `config/train/overfit.yaml` | 1-demo pipeline sanity-check config. |
| `config/train/data/libero.yaml` | Dataset config + `use_language` ablation switch + proprio layout. |
| `utils.py` | `ModelObjectCallBack` (per-epoch pickled model dump) + `PeriodicPrintCallback` (terminal progress every N epochs for long overfit runs). |
| `requirements.txt` | Pinned deps. `transformers>=4.48,<5` is critical (earlier misses `TimmWrapperModel`; v5 breaks the FAST processor). |

`eval.py` is the original LeWM CEM / Adam latent-planning entry point — kept
for reference only, never imported by the VLA baseline.

## Hyperparameters (Frozen Baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Action chunk `H` | 20 raw steps | 1 s at LIBERO's 20 Hz control frequency |
| Sliding-window stride | 1 | Standard VLA practice; ~H× more samples than stride=H |
| FAST vocab | 1024 | BPE fitted on LIBERO-Spatial via `--fit-tokenizer`, then reused across all 4 suites |
| `max_action_tokens` | 80 | Observed max = 75 across all suites |
| `max_lang_tokens` | 25 | Covers the longest T5-tokenized LIBERO instruction |
| Embed dim `D` | 192 | ViT-Tiny hidden |
| Predictor (depth / heads / dim_head / mlp_dim) | 6 / 16 / 64 / 2048 | |
| Predictor dropout | 0.2 | Config overrides the class default of 0.1 |
| `emb_dropout` | 0.0 | |
| Proprio input dim | 9 | `ee_pos(3) + xyzw_quat(4) + gripper_raw(2)` |
| Projector | `MLP(hidden_dim→2048→D)` with `LayerNorm` | Explicitly **not** BatchNorm — avoids train/eval statistics mismatch. Switch via `cfg.projector.norm_type='batch'`; **required** when `cfg.loss.sigreg_weight > 0` (LeWM paper Section 3). See "Known caveats of switching to BatchNorm projector". |
| T5 variant | `t5-small`, fully frozen | 512-dim hidden, projected to D via trainable `lang_proj` |
| Optimizer | `AdamW`, lr=5e-5, weight_decay=0.05 | `LinearWarmupCosineAnnealingLR` on epoch interval |
| Batch size | 128 | |
| Max epochs | 100 | **Checkpoint selection: pick lowest val loss, not last epoch** — the baseline overfits before reaching `max_epochs`. |
| Precision | `bf16` mixed | |
| Gradient clip | 1.0 | |
| Label smoothing (CE) | 0.1 | Configurable via `cfg.label_smoothing`; 0 disables. CE floor ≈ 1.02 on a 1026-way vocab — read training curves against this floor, not zero. |
| Train / val split | 0.9 / 0.1 | **Demo-level** (see `train.py`: reads `demo_idx` per file and partitions whole demos, not individual chunks) |
| Logging | TensorBoard | `lightning.pytorch.loggers.TensorBoardLogger`, view with `tensorboard --logdir <run_dir>/tb_logs` |
| `loss.pred_weight` | 0.0 | > 0 enables state prediction (3 STATE_QUERY tokens + 3 MSE losses). Paper-default starting point: `1.0`. See "State-Prediction Extension". |
| `loss.sigreg_weight` | 0.0 | > 0 enables SIGReg anti-collapse on encoder outputs. Paper default: `0.1` (Section 3). Forces `projector.norm_type='batch'`. |
| `loss.sigreg.kwargs` | `knots=17, num_proj=1024` | Paper defaults; safe to leave alone (paper Fig. 15: barely sensitive to either). |

## Ablation Modes

All four are wired end-to-end (preprocess / train / eval) and baseline-compatible.

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
above. Three STATE_QUERY tokens are appended at training time, three MSE
losses are added, and SIGReg is enforced on encoder outputs.

```bash
# (One-time) re-preprocess to populate image_*_future + proprio_future
python preprocess_libero.py ...        # see "How to enable" above

# Train with paper defaults
python train.py data=libero \
    loss.pred_weight=1.0 \
    loss.sigreg_weight=0.1 \
    projector.norm_type=batch          # required when sigreg_weight > 0

# Evaluate via the per-epoch object checkpoint (eval_libero.py is unchanged
# but cannot reconstruct the SP architecture from `_weights.ckpt` alone)
python eval_libero.py \
    --checkpoint /path/to/lewm_epoch_{N}_object.ckpt \
    --tokenizer /path/to/fast_tokenizer \
    --processed-dir /path/to/libero_processed/<suite>/ \
    --suite libero_spatial
```

Setting only `pred_weight > 0` (without SIGReg) trains state prediction
without anti-collapse. Setting only `sigreg_weight > 0` (without state
prediction) regularizes the current encoder outputs to isotropic Gaussian
with no future-frame loss; SIGReg's input degenerates to 2 streams in
that case (`z_ag_t`, `z_hd_t`).

## External Libraries

- **`stable-worldmodel` (`swm`)** — env wrappers, `swm.data.utils.get_cache_dir()` for checkpoint locations.
- **`stable-pretraining` (`spt`)** — training orchestration (`spt.Module`, `spt.Manager`, `spt.data.DataModule`), ViT backbone factory (`spt.backbone.utils.vit_hf`).
- **`transformers`** — `AutoProcessor` (FAST tokenizer, `trust_remote_code=True`), `T5EncoderModel` + `T5Tokenizer`.
- **`lightning.pytorch.loggers.TensorBoardLogger`** — scalar logging.

## Key Details

- **Data layout**: training HDF5 under `${STABLEWM_HOME}/libero/`; per-suite preprocessed output under `${DATA_ROOT}/libero_processed/<suite>/` (one `.h5` per task — the **directory** is what `LiberoDataset` consumes for joint training, not individual files). Frozen baseline checkpoint lives at `/Data/lyw/checkpoints/multitask_ln_100ep/lewm_weights.ckpt` (single ckpt joint-trained on the suite). The `${DATA_ROOT}/stable-wm/<suite>/<task>/` per-task layout exists only for the alternative `run_all_suites.sh` pipeline and is **not** how the frozen 65 % baseline was produced.
- **FAST tokens**: variable-length int32 (`h5py.vlen_dtype`) per sample; each preprocessed HDF5 stores its own `action_low` / `action_high` percentile bounds (for inverse normalization at eval time), `chunk_size`, `chunk_stride`, and `language_instruction` in the file attrs.
- **Device handling**: no hardcoded `cuda` — tensor device is inferred from inputs; the caller moves `JEPA` to the target device.
- **Checkpoint formats**:
  - `lewm_weights.ckpt` — Lightning state_dict saved by `spt.Manager`; loaded via `load_checkpoint()` which strips the `model.` prefix and skips `lang_encoder.*` keys (T5 is reloaded from pretrained).
  - `lewm_epoch_{N}_object.ckpt` — `torch.save(model)` pickle from `ModelObjectCallBack`; loaded directly via `torch.load(..., weights_only=False)`.
- **Checkpoint selection convention**: always pick the epoch with the lowest val loss, not `epoch_{max_epochs}` — the baseline consistently overfits before `max_epochs` is reached.
