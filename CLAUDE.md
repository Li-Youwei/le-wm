# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeWorldModel (LeWM) modified for **unified action prediction + world model co-training** on LIBERO. The original LeWM is a JEPA world model that predicts future latent states conditioned on actions. We modify it to: (1) predict actions from visual features via FAST tokenization, and (2) use world model prediction as training-time auxiliary supervision.

## Current Modification Goal

Convert LeWM's Predictor from action-conditioned next-latent prediction into a **unified autoregressive sequence model**.

Training-time input sequence fed to Transformer:
```
[z_t, <BOS>, T_1^GT, T_2^GT, ..., T_k^GT, [STATE_QUERY]]
```

Targets (what we supervise):
- Action positions: next FAST token via shifted teacher forcing (CE loss)
- `[STATE_QUERY]` position: `z_{t+1}` from encoder (MSE loss). Under chunk-level indexing, `t+1` = one full chunk later = H=10 raw steps. See Temporal Indexing Convention below.

Inference:
```
[z_t, <BOS>] → autoregressively generate T_1, ..., T_k, <EOS>
→ FAST decode → continuous action chunk → execute on robot
```

Where:
- `z_t`: visual latent from encoder (192-dim continuous vector, not a discrete token)
- `<BOS>`: beginning-of-action-sequence token (id=1024)
- `T_1...T_k`: FAST action tokens (discrete, vocab=0..1023, variable-length ~20-30 tokens)
- `<EOS>`: end-of-action-sequence token (id=1025)
- `[STATE_QUERY]`: learnable embedding, its output is projected to predict `ẑ_{t+1}` (the latent state one chunk = H raw steps later)

Standard causal mask: each position attends to itself and all positions to its left. Training uses teacher forcing on action token positions.

### Temporal Indexing Convention

**CRITICAL — read this before implementing anything.**

All timestep indices in this document use **chunk-level** indexing, not raw environment steps:
- `t` = chunk index (NOT raw environment step)
- `H` = action chunk length in raw environment steps = 10 (1 second at LIBERO's 10Hz)
- Observation `o_t` = frame at raw step `t * H`
- Action chunk `a_t` = `[a_{t*H}, a_{t*H+1}, ..., a_{t*H+H-1}]` (10 consecutive raw actions)
- FAST tokens: `[T_1...T_k]` = `FAST_encode(a_t)` — tokenization of the entire chunk
- Next observation: `o_{t+1}` = frame at raw step `(t+1) * H` — this is **H=10 raw steps** after `o_t`
- `[STATE_QUERY]` target: `z_{t+1} = Encoder(o_{t+1})`

**In chunk-level indexing, `z_{t+1}` is NOT one raw control step later — it is one full chunk (H=10 raw steps) later.** The data pipeline must pair each `o_t` with `o_{t+1}` (i.e., raw frames that are H=10 steps apart), not consecutive raw frames.

## Development Environment

- **OS**: macOS (development/debugging), Linux GPU server (training)
- **Conda**: miniforge, environment name `vla`, Python 3.10
- **Project path**: `~/Documents/le-wm/`

```bash
# Activate environment
conda activate vla

# Install LeWM dependencies
pip install stable-worldmodel[train,env]

# Install FAST tokenizer
pip install transformers  # for AutoProcessor.from_pretrained("physical-intelligence/fast")

# Train (will be modified)
python train.py data=libero

# Data location
export STABLEWM_HOME=/path/to/storage  # defaults to ~/.stable-wm/
```

**Note on macOS**: `decord` (video decoding library) may not work on macOS. Mac is for development and debugging only — actual training runs on Linux GPU servers.

There are no tests, linting, or CI pipelines in this repository.

## Architecture (Original)

Four Python files implement the core:

- **`jepa.py`** — `JEPA` class: full model (encoder, predictor, projectors, action embedder). Key methods: `encode()`, `predict()`, `rollout()`, `criterion()` (planning-time latent cost, NOT training loss), `get_cost()`.
- **`module.py`** — Neural network building blocks: `SIGReg`, `Attention`, `Block`, `ConditionalBlock` (AdaLN-zero), `Transformer`, `ARPredictor`, `Embedder`, `MLP`.
- **`train.py`** — Training entry point. Contains `lejepa_forward()` (the actual training step and loss computation). Uses Hydra config, PyTorch Lightning, WandB.
- **`eval.py`** — Evaluation/planning. CEM or Adam solver in latent space.
- **`utils.py`** — Image preprocessing, StandardScaler, checkpoint callback.

### Original Data Flow (Training)

```
batch["obs"]    → encoder(ViT) → emb (B, T=4, 192)
batch["action"] → Embedder     → act_emb (B, T=4, 192)

ctx_emb = emb[:, :3]       # frames 0,1,2
ctx_act = act_emb[:, :3]   # actions 0,1,2
tgt_emb = emb[:, 1:]       # frames 1,2,3

pred_emb = predictor(ctx_emb, ctx_act)  # ARPredictor with ConditionalBlock

loss = MSE(pred_emb, tgt_emb) + λ * SIGReg(emb)
```

### Original Action Injection

Action is injected as **external conditioning** via AdaLN-zero in `ConditionalBlock`:
- `action → Embedder → (B,T,192)` passed as `c` to each ConditionalBlock
- Each layer: `c → adaLN_modulation → 6 params (shift/scale/gate × attn/mlp)`
- This modulates LayerNorm outputs, NOT part of the attention sequence

### Original Predictor (ARPredictor)

- Uses `ConditionalBlock` (AdaLN) — action is external conditioning, not in the sequence
- Causal self-attention (`is_causal=True` in `F.scaled_dot_product_attention`)
- Input: `x=(B,T,192)` observation embeddings, `c=(B,T,192)` action embeddings
- Output: `(B,T,192)` predicted next-state embeddings

## Modification Plan

### What Changes

| Component | Original | Modified |
|-----------|----------|----------|
| Predictor block | `ConditionalBlock` (AdaLN) | `Block` (standard self-attention) |
| Action representation | Continuous → `Embedder` → 192d | FAST discrete tokens → `nn.Embedding(1027, 192)` (1024 FAST + BOS + EOS + PAD) |
| Predictor input | Two separate inputs `(x, c)` | One unified sequence `[z_t, <BOS>, T_1...T_k, STATE_QUERY]` |
| Action output | None (actions found via CEM) | Classification head `nn.Linear(192, 1026)` on action positions |
| State output | `pred_proj` on all positions | `pred_proj` on last position only |
| Training loss | `L_pred + λ*L_sigreg` | `L_CE + α*L_pred + λ*L_sigreg` |
| Inference | CEM rollout (300 samples × 30 iters) | Autoregressive token generation → FAST decode |

### Special Tokens

- Vocab 0..1023: FAST action tokens
- Token 1024: `<BOS>` — beginning of action sequence (prepended before T_1)
- Token 1025: `<EOS>` — end of action sequence (appended after T_k as target for last action position)
- Token 1026: `<PAD>` — padding token for variable-length batching

`nn.Embedding` size = 1027 (0..1026). `action_head` output size = 1026 (predict 0..1025; PAD is never a prediction target).

### Temporal Indexing Convention

**CRITICAL: `t` is a chunk-level index, NOT a raw environment step.**

LIBERO runs at 10Hz. One action chunk = H=10 raw steps = 1 second. We index by chunks:

```
Chunk t:
  observation: o_t          = frame at raw step t*H     (e.g., step 0)
  action chunk: a_t         = [a_{t*H}, a_{t*H+1}, ..., a_{t*H+H-1}]  (10 raw actions)
  FAST tokens: T_1...T_k   = FAST(a_t)                 (encodes all 10 raw steps)
  next observation: o_{t+1} = frame at raw step (t+1)*H (e.g., step 10)
```

So `z_t = Enc(o_t)` and `z_{t+1} = Enc(o_{t+1})` are **H=10 raw steps apart**. The `[STATE_QUERY]` predicts the latent of the frame AFTER executing the entire action chunk, not just 1 raw control step.

Dataset construction: from a LIBERO trajectory of N raw steps, extract non-overlapping chunk-aligned samples:
- Chunk t=0: `(frame at raw step 0, actions raw steps 0-9, frame at raw step 10)`
- Chunk t=1: `(frame at raw step 10, actions raw steps 10-19, frame at raw step 20)`
- ...

### Unified Sequence Structure (Training)

```
Position:  0      1      2      ...  k      k+1
Token:    z_t   <BOS>   T_1^GT ...  T_{k-1}^GT  T_k^GT
Target:    -     T_1    T_2    ...  T_k     <EOS>     ← CE loss on these
                                                       
Position: k+2
Token:    [STATE_QUERY]    ← learnable embedding
Target:   z_{t+1}          ← MSE loss (world model: state after executing full chunk)
         (= Enc(o_{t+1}), one chunk = H=10 raw steps after o_t)
```

Note: z_t is a continuous 192-dim vector (from encoder), not a discrete token. It gets a type embedding to distinguish it from action tokens.

Note on history context: Original LeWM uses `history_size=3` (encodes frames t-2, t-1, t and feeds all 3 to the predictor). For the initial implementation we start with **single-frame `z_t`** to keep things simple. Extending to multi-frame `z_{t-h+1:t}` (multiple visual tokens at the start of the sequence) is a natural follow-up — the architecture supports it by just prepending more visual tokens with `type_embedding[0]`.

### Attention Mask (Training)

Standard causal: position i attends to positions 0..i **(including itself)**. No information leak because the supervision target at each position is the *next* token (shifted by 1). Concretely:
- `<BOS>` (pos 1) sees `z_t, <BOS>` → target is `T_1` (next token, not visible)
- `T_1^GT` (pos 2) sees `z_t, <BOS>, T_1^GT` → target is `T_2` (next token, not visible)
- `T_i^GT` (pos i+1) sees `z_t, <BOS>, T_1^GT...T_i^GT` → target is `T_{i+1}`
- `[STATE_QUERY]` (last pos) sees all preceding tokens including all GT action tokens → target is `z_{t+1}`

**Do NOT implement a custom "cannot see self" mask.** Standard causal mask lets each position see itself. When there is NO padding (all sequences same length), use `is_causal=True` in `scaled_dot_product_attention`. When batches have padding, construct an explicit `attn_mask = causal_mask & non_padding_mask` and pass it via the `attn_mask` parameter instead of `is_causal`.

**Padding handling:** Since FAST token sequences are variable-length within a batch, the actual mask is `causal_mask AND non_padding_mask`. Implementation details:
- `pad_token_id = 1026` (as defined in Special Tokens above)
- Padding tokens must be masked out in attention so no real token attends to them (including `[STATE_QUERY]`)
- For CE loss targets, set non-action positions (z_t, STATE_QUERY, padding) to `-100` and use `F.cross_entropy(..., ignore_index=-100)` — this is PyTorch's standard convention

### File-by-File Modification Plan

#### `module.py` — Add new components, keep originals

**Add:**
- `UnifiedPredictor` class: new predictor using `Block` (not `ConditionalBlock`)
  - `action_embedding = nn.Embedding(1027, embed_dim)` (1024 FAST + BOS + EOS + PAD)
  - `type_embedding = nn.Embedding(3, embed_dim)` (0=visual, 1=action, 2=state_query)
  - `pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))` — learnable positional encoding, where `max_seq_len = 1 + 1 + max_action_tokens + 1` (z_t + BOS + tokens + STATE_QUERY). Set `max_action_tokens=35` as safe upper bound (FAST typically produces ~20-30 tokens for LIBERO). Truncate to actual sequence length per sample: `x = x + pos_embedding[:, :L]` (same pattern as original `ARPredictor`)
  - `state_query = nn.Parameter(torch.randn(1, 1, embed_dim))` learnable
  - `action_head = nn.Linear(embed_dim, 1026)` classification over vocab (0..1025, PAD excluded from targets)
  - `forward(z_t, action_tokens, action_lengths)` builds unified sequence, applies causal+padding mask
- Helper: `build_unified_sequence()` — assembles [z_t, BOS, T_1...T_k, STATE_QUERY] with proper embeddings
- Helper: `build_causal_mask()` — standard causal mask for the unified sequence

**Keep unchanged:** `SIGReg`, `Attention`, `Block`, `MLP`, `Transformer`, `Embedder` (may be needed for reference or fallback)

#### `jepa.py` — Modify JEPA class

**Modify `__init__`:**
- Replace `action_encoder` (Embedder) with reference to `UnifiedPredictor`'s internal `action_embedding`
- Add `action_head` reference
- Keep `encoder`, `projector`, `pred_proj`

**Modify `predict()`:**
- New signature: `predict(z_t, action_tokens, action_lengths)`
- Calls `UnifiedPredictor` which returns `(action_logits, state_pred)`

**Add `predict_actions()`:**
- Autoregressive generation for inference
- Loop: start with `[z_t, BOS]`, predict T_1, append, predict T_2, ... until EOS or max_len

**Modify `encode()`:**
- Keep visual encoding unchanged
- Remove action encoding (no longer done here — action tokens are pre-computed by FAST)

**Remove/deprecate:** `rollout()`, `criterion()`, `get_cost()` — these are CEM planning-time components (criterion() computes latent goal-matching cost for MPC, NOT training loss). No longer needed since we use direct autoregressive action generation instead of search-based planning.

#### `train.py` — Modify training loop

**Modify `lejepa_forward()`:**
- Load pre-computed FAST tokens from batch: `batch["fast_tokens"]`, `batch["fast_lengths"]`
- Encode current frame: `z_t = encode(o_t)`, encode next-chunk frame: `z_{t+1} = encode(o_{t+1})` — reminder: `o_{t+1}` is the frame H=10 raw steps after `o_t` (chunk-level indexing, see Temporal Indexing Convention)
- Call unified predictor: `action_logits, state_pred = model.predict(z_t, fast_tokens_gt, fast_lengths)`
- Compute losses:
  - `L_CE = CrossEntropyLoss(action_logits, target_tokens)` (shifted by 1)
  - `L_pred = MSE(state_pred, z_{t+1})` — NO detach, keep end-to-end gradient flow through encoder (this is core to LeWM's design)
  - `L_sigreg = SIGReg(embeddings)`
  - `L_total = L_CE + α * L_pred + λ * L_sigreg`

**Add LIBERO data config:** New Hydra YAML for LIBERO dataset with FAST token fields.

#### `eval.py` — Rewrite for direct inference

**Replace CEM planning with autoregressive generation:**
- Load model, encode observation
- Call `predict_actions(z_t)` → FAST tokens → FAST decode → continuous actions
- Execute actions in environment

#### New file: `preprocess_libero.py`

**Standalone script (not part of model code):**
- Load LIBERO demonstrations
- Normalize actions (1st/99th percentile per dimension)
- Train FAST BPE tokenizer on LIBERO actions (or use FAST+ universal)
- Extract chunk-aligned samples from each trajectory (using raw-step indices here for clarity):
  - `image_current`: observation at raw step `i`
  - `actions`: H=10 consecutive actions `a_{i}, a_{i+1}, ..., a_{i+H-1}`
  - `image_future`: observation at raw step `i + H` (10 steps later)
  - Slide window by H steps: `i = 0, H, 2H, ...` (non-overlapping chunks)
- Tokenize each action chunk with FAST → save alongside observations in HDF5
- Each HDF5 sample contains: `image_current`, `image_future`, `fast_tokens` (variable-length), `fast_length`, `continuous_actions` (for reference/debugging)

### Hyperparameters (Starting Point)

| Parameter | Value | Source |
|-----------|-------|--------|
| Action chunk H | 10 steps (1 sec @ 10Hz) | FAST paper recommendation |
| FAST vocab size | 1024 | FAST paper default |
| FAST rounding scale γ | 10 | FAST paper default |
| BPE vocab size | 1024 | FAST paper default |
| max_action_tokens | 35 | Safe upper bound for FAST on LIBERO (~20-30 typical) |
| max_seq_len | 38 | 1(z_t) + 1(BOS) + 35(tokens) + 1(STATE_QUERY) |
| Embed dim | 192 | LeWM default (ViT-Tiny) |
| Predictor depth | 6 layers | LeWM default |
| Predictor heads | 16 | LeWM default |
| L_sigreg weight λ | 0.09 | LeWM default |
| L_pred weight α | 1.0 | Start here, tune later |
| Learning rate | From LeWM config | Keep original schedule |
| Batch size | 128 | LeWM default |

### Key Constraints

- **Do NOT detach encoder gradients for L_pred.** Original LeWM trains encoder end-to-end through prediction loss. Keep this behavior — it's core to LeWM's design.
- **SIGReg applies to encoder outputs (z_t, z_{t+1}), NOT to action token embeddings.** Action tokens are discrete and don't risk collapse.
- **Action token positions use CE loss; state prediction position uses MSE loss.** These are different heads on the same transformer output.
- **`criterion()` in original `jepa.py` is the planning-time latent cost (for CEM/MPC), NOT the training loss.** The training loss is computed in `lejepa_forward()` in `train.py`. Do not confuse these two.
- **Config naming**: Use `chunk_horizon_raw_steps = 10` in new configs for the action chunk length. Do NOT overload the original `frameskip` variable — in the original LeWM code, `frameskip` means "skip N raw frames between encoder observations", which is a different concept from the FAST action chunk length. Keep these two separate to avoid confusion.

### Scope: Language Conditioning

**This version does NOT include language/task conditioning.** LIBERO is a multi-task benchmark with language instructions (e.g. "pick up the red cup"), and standard VLA/WAM formulations condition on language: `p(a_{1:H} | o, l)`. However:

- LeWM is task-agnostic by design — no language input
- The advisor's requirement is to modify LeWM's predictor, not to build a full VLA
- Adding language conditioning requires a text encoder + cross-attention or prefix tokens, which is out of scope for the initial implementation

**Initial approach**: Train one model per LIBERO task (single-task), same as how LeWM trains one model per environment (PushT, OGBench, etc.). Language integration (`l` as additional conditioning prefix in the unified sequence) is a natural follow-up after the single-task pipeline is validated. Do NOT add any language/text processing in this phase.

## External Libraries

- **`stable-worldmodel` (`swm`)** — Environment wrappers, planning solvers, evaluation APIs
- **`stable-pretraining` (`spt`)** — Training orchestration, dataset loading, ViT backbone
- **`transformers`** — FAST tokenizer via `AutoProcessor.from_pretrained("physical-intelligence/fast")`

## Configuration

Hydra YAML files under `config/`:
- `config/train/lewm.yaml` — Main training config
- `config/train/data/*.yaml` — Dataset configs (add `libero.yaml`)
- `config/eval/*.yaml` — Evaluation configs

## Key Details

- **Data format**: HDF5 files stored under `$STABLEWM_HOME`
- **FAST tokens**: Pre-computed during preprocessing, stored as variable-length integer sequences in HDF5
- **Device handling**: Uses `proj.device` (not hardcoded `cuda`)
- **WandB**: Set `entity` and `project` in config before training
