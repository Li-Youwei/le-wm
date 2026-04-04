import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, attn_mask=None):
        """
        x : (B, T, D)
        attn_mask : optional (B, 1, T, T) bool mask. True = attend, False = mask.
                    When provided, overrides is_causal (cannot use both).
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


# --- Unified Predictor for FAST action tokens + world model ---

# FAST token vocabulary
FAST_VOCAB_SIZE = 1024
BOS_TOKEN_ID = 1024
EOS_TOKEN_ID = 1025
PAD_TOKEN_ID = 1026
TOTAL_VOCAB_SIZE = FAST_VOCAB_SIZE + 3  # 1027: 0..1023 FAST + BOS + EOS + PAD
ACTION_HEAD_SIZE = FAST_VOCAB_SIZE + 2  # 1026: predict 0..1025, PAD excluded


class UnifiedPredictor(nn.Module):
    """Unified autoregressive predictor for action token generation + state prediction.

    Processes a sequence [z_t, BOS, T_1...T_k, PAD..., STATE_QUERY] with causal
    attention and padding mask. Produces action token logits at BOS/action positions
    and a state embedding prediction at the STATE_QUERY position.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        max_action_tokens: int = 40,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_action_tokens = max_action_tokens
        # max_seq_len = 1 (z_t) + 1 (BOS) + max_action_tokens + 1 (STATE_QUERY)
        self.max_seq_len = max_action_tokens + 3

        # Token embeddings
        self.action_embedding = nn.Embedding(TOTAL_VOCAB_SIZE, embed_dim)  # 1027 entries
        self.type_embedding = nn.Embedding(3, embed_dim)  # 0=visual, 1=action, 2=state_query

        # Positional encoding (learnable, same pattern as ARPredictor)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))

        # Learnable state query token
        self.state_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Action classification head: predict vocab 0..1025 (PAD excluded from targets)
        self.action_head = nn.Linear(embed_dim, ACTION_HEAD_SIZE)

        # Transformer blocks (standard Block, not ConditionalBlock)
        self.blocks = nn.ModuleList([
            Block(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def _build_attn_mask(
        self, action_tokens: torch.Tensor, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build combined causal + non-padding attention mask.

        Args:
            action_tokens: (B, max_action_tokens) with PAD_TOKEN_ID for padding.
            seq_len: total sequence length L.
            device: target device.

        Returns:
            (B, 1, L, L) bool mask. True = attend, False = mask out.
        """
        B = action_tokens.size(0)
        L = seq_len

        # Causal: lower-triangular (L, L)
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

        # Non-padding: which key positions are real (not PAD)
        # pos 0 = z_t (always real), pos 1 = BOS (always real), pos L-1 = STATE_QUERY (always real)
        # pos 2..2+max_action_tokens-1 = action tokens or PAD
        is_real = torch.ones(B, L, dtype=torch.bool, device=device)
        is_real[:, 2:2 + self.max_action_tokens] = (action_tokens != PAD_TOKEN_ID)

        # Combine: (B, 1, L, L) — broadcasts over heads in SDPA
        key_mask = is_real.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) & key_mask  # (B, 1, L, L)

        return attn_mask

    def forward(
        self,
        z_t: torch.Tensor,
        action_tokens: torch.Tensor,
        action_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: (B, D) visual latent from encoder (continuous, not a discrete token).
            action_tokens: (B, max_action_tokens) FAST token ids padded with PAD_TOKEN_ID.
            action_lengths: (B,) number of real FAST tokens per sample.

        Returns:
            action_logits: (B, 1+max_action_tokens, 1026) logits at BOS + action positions.
                Position 0 in this tensor = BOS output -> predicts T_1.
                Position j (1..k) = T_j output -> predicts T_{j+1}.
                Position k = T_k output -> predicts EOS.
                Positions k+1.. = PAD output -> ignored via target=-100 in loss.
            state_pred: (B, D) transformer output at STATE_QUERY position
                (not yet projected by pred_proj; that happens in JEPA).
        """
        B = z_t.size(0)
        device = z_t.device

        # 1. Embed visual token: z_t is continuous (B, D), add type=0
        z_emb = z_t.unsqueeze(1) + self.type_embedding.weight[0]  # (B, 1, D)

        # 2. Embed BOS + action tokens (including PAD positions): type=1
        bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        bos_action = torch.cat([bos, action_tokens], dim=1)  # (B, 1+max_action_tokens)
        action_emb = self.action_embedding(bos_action) + self.type_embedding.weight[1]

        # 3. Embed STATE_QUERY: type=2
        sq_emb = self.state_query.expand(B, -1, -1) + self.type_embedding.weight[2]  # (B, 1, D)

        # 4. Concatenate full sequence: [z_t, BOS, T_1...T_k, PAD..., STATE_QUERY]
        x = torch.cat([z_emb, action_emb, sq_emb], dim=1)  # (B, max_seq_len, D)
        L = x.size(1)

        # 5. Positional embedding + dropout
        x = x + self.pos_embedding[:, :L]
        x = self.dropout(x)

        # 6. Build attention mask (causal AND non-padding)
        attn_mask = self._build_attn_mask(action_tokens, L, device)

        # 7. Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.norm(x)

        # 8. Extract outputs
        # Action logits: positions 1..1+max_action_tokens (BOS through last action/PAD)
        action_output = x[:, 1:1 + 1 + self.max_action_tokens]  # (B, 1+max_action_tokens, D)
        action_logits = self.action_head(action_output)  # (B, 1+max_action_tokens, 1026)

        # State prediction: last position (STATE_QUERY)
        state_pred = x[:, -1]  # (B, D)

        return action_logits, state_pred

    # NEW: autoregressive generation for inference (no STATE_QUERY, no padding)
    @torch.no_grad()
    def generate(
        self,
        z_t: torch.Tensor,
        max_len: int = 40,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate FAST action tokens.

        Sequence grows as [z_t, BOS, T_1, T_2, ...] with pure causal attention
        (no STATE_QUERY, no padding). Stops at EOS or max_len.

        Args:
            z_t: (B, D) visual latent from encoder.
            max_len: maximum number of action tokens to generate.
            temperature: 0.0 for greedy (argmax), >0 for sampling.

        Returns:
            tokens: (B, gen_len) generated token ids (may include EOS, followed by PAD).
            lengths: (B,) number of real tokens per sample (EOS exclusive).
        """
        B = z_t.size(0)
        device = z_t.device

        # Initialize sequence: [z_t_emb, bos_emb]
        z_emb = z_t.unsqueeze(1) + self.type_embedding.weight[0]  # (B, 1, D)
        bos_ids = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        bos_emb = self.action_embedding(bos_ids) + self.type_embedding.weight[1]  # (B, 1, D)
        seq = torch.cat([z_emb, bos_emb], dim=1)  # (B, 2, D) — base embeddings, no pos

        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            L = seq.size(1)
            x = seq + self.pos_embedding[:, :L]

            # No padding → pure causal mask (is_causal=True default in Block/Attention)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)

            # Logits at last position → next token prediction
            logits = self.action_head(x[:, -1])  # (B, ACTION_HEAD_SIZE)

            if temperature <= 0:
                next_token = logits.argmax(dim=-1)  # (B,)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)

            # Force PAD for already-finished samples
            next_token = torch.where(finished, torch.tensor(PAD_TOKEN_ID, device=device), next_token)
            generated.append(next_token)

            # Track EOS
            finished = finished | (next_token == EOS_TOKEN_ID)
            if finished.all():
                break

            # Append new token embedding to sequence (base embedding, no pos)
            new_emb = self.action_embedding(next_token.unsqueeze(1)) + self.type_embedding.weight[1]
            seq = torch.cat([seq, new_emb], dim=1)

        raw_tokens = torch.stack(generated, dim=1)  # (B, gen_len)

        # Compute lengths: count tokens before first EOS
        lengths = torch.full((B,), raw_tokens.size(1), dtype=torch.long, device=device)
        for i in range(B):
            eos_pos = (raw_tokens[i] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                lengths[i] = eos_pos[0].item()

        # Strip EOS/PAD: return only real FAST action tokens (vocab 0..1023).
        # Pad output to max generated length so the tensor is rectangular.
        max_len_actual = lengths.max().item() if lengths.numel() > 0 else 0
        max_len_actual = max(max_len_actual, 1)  # at least 1 to avoid empty tensor
        tokens = torch.full((B, max_len_actual), PAD_TOKEN_ID, dtype=torch.long, device=device)
        for i in range(B):
            k = lengths[i].item()
            if k > 0:
                tokens[i, :k] = raw_tokens[i, :k]

        return tokens, lengths
