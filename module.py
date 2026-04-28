import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


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
    """Standard Transformer"""

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

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        for block in self.layers:
            x = block(x)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
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


# --- FAST token vocabulary constants ---

FAST_VOCAB_SIZE = 1024
BOS_TOKEN_ID = 1024
EOS_TOKEN_ID = 1025
PAD_TOKEN_ID = 1026
TOTAL_VOCAB_SIZE = FAST_VOCAB_SIZE + 3  # 1027: 0..1023 FAST + BOS + EOS + PAD
ACTION_HEAD_SIZE = FAST_VOCAB_SIZE + 2  # 1026: predict 0..1025, PAD excluded


class ARPredictor(nn.Module):
    """Autoregressive predictor for VLA baseline + optional state prediction.

    Baseline (use_state_prediction=False) processes the unified sequence:
        [l_1...l_n, z_agent, z_hand, z_proprio, BOS, T_1...T_k, PAD...]

    With state prediction enabled (use_state_prediction=True), three query
    tokens are appended at the end during training:
        [l_1...l_n, z_agent, z_hand, z_proprio, BOS, T_1...T_k, PAD...,
         Q_ag, Q_hd, Q_pr]

    The Q tokens read out predicted future latents (z_ag_{t+H}, z_hd_{t+H})
    and predicted raw future proprio (s_pr_{t+H}, 9d).

    Attention mask is prefix-bidirectional + action-causal + query-isolated:
    - Perception prefix (lang + visual + proprio): bidirectional among real tokens
    - Action tokens (BOS + T_1...T_k): see all real prefix, causal within action group
    - PAD tokens: attend to nothing, no other token attends to them
    - STATE_QUERY tokens (Q_ag, Q_hd, Q_pr): see all real prefix + real action zone
      (BOS + real T) + themselves only. The three queries DO NOT attend to each
      other — this preserves three-way independence of the predictions and
      prevents one head from copying another's hidden state.

    Inference (`generate()`) does NOT construct STATE_QUERY tokens regardless
    of the flag, so eval_libero.py flow `[lang, z_ag, z_hd, z_pr, BOS] →
    autoregressive` is unchanged.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        max_action_tokens: int = 80,
        max_lang_tokens: int = 25,
        proprio_dim: int = 9,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        use_state_prediction: bool = False,
        state_head_norm_type: str = "layer",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_action_tokens = max_action_tokens
        self.max_lang_tokens = max_lang_tokens
        self.proprio_dim = proprio_dim
        self.use_state_prediction = use_state_prediction
        self.n_state_query = 3 if use_state_prediction else 0

        # max_seq_len = lang + z_agent + z_hand + z_proprio + BOS + max_action_tokens
        # (+ Q_ag + Q_hd + Q_pr when use_state_prediction is True)
        self.max_seq_len = (
            max_lang_tokens + 3 + 1 + max_action_tokens + self.n_state_query
        )

        # Proprioception encoder: proprio_dim (9d for LIBERO) → embed_dim
        self.proprio_encoder = MLP(proprio_dim, embed_dim, embed_dim)

        # Token embeddings for action tokens (FAST vocab + BOS/EOS/PAD)
        self.action_embedding = nn.Embedding(TOTAL_VOCAB_SIZE, embed_dim)  # 1027 entries

        # Type embeddings: 0=language, 1=visual, 2=proprioception, 3=action,
        # (4=state_query when use_state_prediction is True).
        # Size depends on the flag so baseline checkpoints (4-row table) load
        # cleanly into a baseline-arch model.
        n_type = 5 if use_state_prediction else 4
        self.type_embedding = nn.Embedding(n_type, embed_dim)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))

        # Action classification head: predict vocab 0..1025 (PAD excluded from targets)
        self.action_head = nn.Linear(embed_dim, ACTION_HEAD_SIZE)

        # State-prediction read-out (only when enabled).
        # Per LeWM paper Section 3: "The predictor is also followed by a
        # projector network with the same implementation as the one used for
        # the encoder" — and the encoder projector is a "1-layer MLP with
        # BatchNorm". Upstream `train.py` confirms this with:
        #     predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
        #                          hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
        # So each per-query state head here is the SAME MLP signature, with
        # the norm type mirroring the encoder-side projector (caller passes
        # `state_head_norm_type` matching `cfg.projector.norm_type`).
        # The proprio head terminates in `proprio_dim` (9 by default) because
        # it predicts the RAW future proprio vector — not an embedding.
        if use_state_prediction:
            if state_head_norm_type == "batch":
                head_norm_fn = nn.BatchNorm1d
            elif state_head_norm_type == "layer":
                head_norm_fn = nn.LayerNorm
            else:
                raise ValueError(
                    f"state_head_norm_type must be 'batch' or 'layer', got "
                    f"'{state_head_norm_type}'"
                )

            # 3 learnable query tokens: Q_ag, Q_hd, Q_pr (one per stream).
            self.state_query_embeddings = nn.Parameter(
                torch.randn(3, embed_dim)
            )
            # Per-stream projector: matches the encoder-side projector
            # signature (LeWM paper Sec. 3 + upstream pattern).
            self.state_pred_head_ag = MLP(
                embed_dim, 2048, embed_dim, norm_fn=head_norm_fn,
            )
            self.state_pred_head_hd = MLP(
                embed_dim, 2048, embed_dim, norm_fn=head_norm_fn,
            )
            self.state_pred_head_pr = MLP(
                embed_dim, 2048, proprio_dim, norm_fn=head_norm_fn,
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def _build_attn_mask(
        self,
        n_lang: int,
        lang_lengths: torch.Tensor | None,
        action_tokens: torch.Tensor,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build hybrid prefix-bidirectional + action-causal attention mask.

        Args:
            n_lang: number of language token positions (0 when language disabled).
            lang_lengths: (B,) real language token count per sample, or None when n_lang==0.
            action_tokens: (B, max_action_tokens) with PAD_TOKEN_ID for padding.
            L: total sequence length.
            device: target device.

        Returns:
            (B, 1, L, L) bool mask. True = attend, False = mask out.
        """
        B = action_tokens.size(0)
        n_prefix = n_lang + 3  # lang + z_agent + z_hand + z_proprio
        action_start = n_prefix + 1  # BOS is at n_prefix, action tokens start at n_prefix+1

        pos = torch.arange(L, device=device)

        # --- Determine which positions are real (not padding) ---
        is_real = torch.zeros(B, L, dtype=torch.bool, device=device)

        # Language: real if position < lang_lengths[b] (skipped when no language)
        if n_lang > 0:
            assert lang_lengths is not None, "lang_lengths required when n_lang > 0"
            lang_pos = pos[:n_lang].unsqueeze(0).expand(B, -1)  # (B, n_lang)
            is_real[:, :n_lang] = lang_pos < lang_lengths.unsqueeze(1)

        # z_agent, z_hand, z_proprio: always real
        is_real[:, n_lang:n_prefix] = True

        # BOS: always real
        is_real[:, n_prefix] = True

        # Action tokens: real if not PAD
        action_end = action_start + self.max_action_tokens
        if action_end <= L:
            is_real[:, action_start:action_end] = (action_tokens != PAD_TOKEN_ID)

        # STATE_QUERY tokens (Q_ag, Q_hd, Q_pr) are always real when present.
        query_start = action_end
        query_end = query_start + self.n_state_query
        if self.n_state_query > 0:
            is_real[:, query_start:query_end] = True

        # --- Zone membership ---
        in_prefix = pos < n_prefix                        # (L,)
        in_action = (pos >= n_prefix) & (pos < action_end)  # (L,) includes BOS
        # in_query is empty when use_state_prediction is False (query_end == query_start).

        # --- Build mask ---
        mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)

        # Real tokens in each zone
        prefix_real = is_real & in_prefix.unsqueeze(0)  # (B, L)
        action_real = is_real & in_action.unsqueeze(0)  # (B, L)

        # Rule 1: Prefix tokens attend bidirectionally to all real prefix tokens
        mask |= prefix_real.unsqueeze(2) & prefix_real.unsqueeze(1)

        # Rule 2: Action tokens attend to all real prefix tokens
        mask |= action_real.unsqueeze(2) & prefix_real.unsqueeze(1)

        # Rule 3: Action tokens attend causally to real action tokens (j <= i)
        causal = pos.unsqueeze(0) <= pos.unsqueeze(1)  # (L, L) lower-triangular
        mask |= action_real.unsqueeze(2) & action_real.unsqueeze(1) & causal.unsqueeze(0)

        # Rule 4: STATE_QUERY rows (Q_ag, Q_hd, Q_pr) — only when enabled.
        # Each Q sees: real prefix + real action zone (BOS + non-PAD T) + itself.
        # Each Q does NOT see: PAD, OTHER queries (preserves 3-way independence
        # so pred_pr cannot peek at pred_ag's hidden state, etc.). No token
        # outside the query group attends to a query (q columns stay all-False).
        if self.n_state_query > 0:
            in_query = (pos >= query_start) & (pos < query_end)  # (L,)
            query_real = is_real & in_query.unsqueeze(0)         # (B, L)

            # Q sees real prefix
            mask |= query_real.unsqueeze(2) & prefix_real.unsqueeze(1)
            # Q sees real action zone (BOS + non-PAD action tokens)
            mask |= query_real.unsqueeze(2) & action_real.unsqueeze(1)
            # Q sees only itself within the query block — diagonal, NOT bidir
            eye_L = torch.eye(L, dtype=torch.bool, device=device)
            mask |= (
                query_real.unsqueeze(2)
                & query_real.unsqueeze(1)
                & eye_L.unsqueeze(0)
            )

        return mask.unsqueeze(1)  # (B, 1, L, L)

    def _build_generate_mask(
        self,
        n_lang: int,
        lang_lengths: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for autoregressive generation (no action PAD).

        During generation the sequence grows as:
            [lang..., z_ag, z_hd, z_pr, BOS, T_1, T_2, ...]
        All action tokens are real (no PAD), but language may still have padding.

        When n_lang == 0 the language block is skipped entirely.

        Returns: (B, 1, L, L) bool mask.
        """
        n_prefix = n_lang + 3
        pos = torch.arange(L, device=device)

        # Real positions: language real + visual/proprio always real + all action real
        is_real = torch.ones(B, L, dtype=torch.bool, device=device)
        # Mask out language padding (skipped when no language)
        if n_lang > 0:
            assert lang_lengths is not None, "lang_lengths required when n_lang > 0"
            lang_pos = pos[:n_lang].unsqueeze(0).expand(B, -1)
            is_real[:, :n_lang] = lang_pos < lang_lengths.unsqueeze(1)

        in_prefix = pos < n_prefix  # (L,)

        mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)

        prefix_real = is_real & in_prefix.unsqueeze(0)
        action_positions = ~in_prefix.unsqueeze(0) & is_real  # (B, L)

        # Prefix: bidirectional among real
        mask |= prefix_real.unsqueeze(2) & prefix_real.unsqueeze(1)
        # Action sees prefix
        mask |= action_positions.unsqueeze(2) & prefix_real.unsqueeze(1)
        # Action: causal within action
        causal = pos.unsqueeze(0) <= pos.unsqueeze(1)
        mask |= action_positions.unsqueeze(2) & action_positions.unsqueeze(1) & causal.unsqueeze(0)

        return mask.unsqueeze(1)

    def forward(
        self,
        z_agent: torch.Tensor,
        z_hand: torch.Tensor,
        z_proprio_raw: torch.Tensor,
        lang_embeds: torch.Tensor | None,
        lang_lengths: torch.Tensor | None,
        action_tokens: torch.Tensor,
        action_lengths: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward: teacher forcing with unified sequence.

        Args:
            z_agent: (B, D) agentview visual latent from encoder.
            z_hand: (B, D) eye-in-hand visual latent from encoder.
            z_proprio_raw: (B, proprio_dim) raw proprioceptive state.
                For LIBERO: (B, 9) = ee_pos(3) + xyzw_quat(4) + gripper_raw(2).
            lang_embeds: (B, max_lang_tokens, D) projected language embeddings,
                or None for language-ablation runs.
            lang_lengths: (B,) real language token count per sample, or None.
            action_tokens: (B, max_action_tokens) FAST token ids padded with PAD_TOKEN_ID.
            action_lengths: (B,) number of real FAST tokens per sample.

        Returns:
            When ``use_state_prediction is False`` (baseline):
                action_logits: (B, 1+max_action_tokens, 1026) logits at BOS + action positions.

            When ``use_state_prediction is True``:
                Tuple ``(action_logits, pred_ag, pred_hd, pred_pr)`` where
                  - action_logits: same shape as baseline,
                  - pred_ag: (B, embed_dim) predicted future agentview latent,
                  - pred_hd: (B, embed_dim) predicted future hand-cam latent,
                  - pred_pr: (B, proprio_dim) predicted RAW future proprio.
        """
        B = z_agent.size(0)
        device = z_agent.device
        n_lang = lang_embeds.size(1) if lang_embeds is not None else 0

        # 1. Encode proprioception
        z_proprio = self.proprio_encoder(z_proprio_raw)  # (B, D)

        # 2. Build prefix embeddings with type embeddings
        vis_agent = z_agent.unsqueeze(1) + self.type_embedding.weight[1]  # (B, 1, D)
        vis_hand = z_hand.unsqueeze(1) + self.type_embedding.weight[1]   # (B, 1, D)
        proprio_emb = z_proprio.unsqueeze(1) + self.type_embedding.weight[2]  # (B, 1, D)

        if lang_embeds is not None:
            lang_prefix = lang_embeds + self.type_embedding.weight[0]  # (B, n_lang, D)
            prefix = torch.cat([lang_prefix, vis_agent, vis_hand, proprio_emb], dim=1)  # (B, n_lang+3, D)
        else:
            prefix = torch.cat([vis_agent, vis_hand, proprio_emb], dim=1)  # (B, 3, D)

        # 3. Build action embeddings (BOS + action tokens) with type embedding
        bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        bos_action = torch.cat([bos, action_tokens], dim=1)  # (B, 1+max_action_tokens)
        action_emb = self.action_embedding(bos_action) + self.type_embedding.weight[3]

        # 4. Concatenate prefix + action zones
        x = torch.cat([prefix, action_emb], dim=1)  # (B, n_prefix + 1 + max_action_tokens, D)

        # 4b. (Optional) Append STATE_QUERY tokens at the very end. Each
        # query is a learnable embedding plus the type embedding (index 4).
        # Index 4 only exists when use_state_prediction is True (n_type=5).
        if self.use_state_prediction:
            # state_query_embeddings: (3, D) → (1, 3, D) → (B, 3, D)
            q_base = self.state_query_embeddings.unsqueeze(0).expand(B, -1, -1)
            q_emb = q_base + self.type_embedding.weight[4]  # broadcast over (B, 3, D)
            x = torch.cat([x, q_emb], dim=1)

        L = x.size(1)

        # 5. Positional embedding + dropout
        x = x + self.pos_embedding[:, :L]
        x = self.dropout(x)

        # 6. Build hybrid attention mask (extended with query rules when SP is on)
        attn_mask = self._build_attn_mask(n_lang, lang_lengths, action_tokens, L, device)

        # 7. Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.norm(x)

        # 8. Extract action zone output: BOS + action token positions
        n_prefix = n_lang + 3
        action_output = x[:, n_prefix:n_prefix + 1 + self.max_action_tokens]  # (B, 1+max_action_tokens, D)
        action_logits = self.action_head(action_output)  # (B, 1+max_action_tokens, ACTION_HEAD_SIZE)

        if not self.use_state_prediction:
            return action_logits

        # 9. Read out the three STATE_QUERY positions and run them through
        #    their respective heads. Q_ag/Q_hd predict latent (D,), Q_pr
        #    predicts the raw 9d proprio vector at t+H (NOT an embedding —
        #    the target is the un-encoded proprio so the loss is in physical
        #    units).
        query_start = n_prefix + 1 + self.max_action_tokens
        q_out = x[:, query_start:query_start + 3]            # (B, 3, D)
        pred_ag = self.state_pred_head_ag(q_out[:, 0])       # (B, D)
        pred_hd = self.state_pred_head_hd(q_out[:, 1])       # (B, D)
        pred_pr = self.state_pred_head_pr(q_out[:, 2])       # (B, proprio_dim)

        return action_logits, pred_ag, pred_hd, pred_pr

    @torch.no_grad()
    def generate(
        self,
        z_agent: torch.Tensor,
        z_hand: torch.Tensor,
        z_proprio_raw: torch.Tensor,
        lang_embeds: torch.Tensor | None,
        lang_lengths: torch.Tensor | None,
        max_len: int = 80,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate FAST action tokens.

        Sequence grows as [lang..., z_ag, z_hd, z_pr, BOS, T_1, T_2, ...]
        (language segment absent when lang_embeds is None).
        Prefix uses bidirectional attention, action tokens use causal.
        Stops at EOS or max_len.

        Returns:
            tokens: (B, gen_len) clean FAST action token ids (EOS/PAD stripped).
            lengths: (B,) number of real tokens per sample.
        """
        B = z_agent.size(0)
        device = z_agent.device
        n_lang = lang_embeds.size(1) if lang_embeds is not None else 0

        # 1. Build prefix embeddings (same as forward)
        z_proprio = self.proprio_encoder(z_proprio_raw)
        vis_agent = z_agent.unsqueeze(1) + self.type_embedding.weight[1]
        vis_hand = z_hand.unsqueeze(1) + self.type_embedding.weight[1]
        proprio_emb = z_proprio.unsqueeze(1) + self.type_embedding.weight[2]

        if lang_embeds is not None:
            lang_prefix = lang_embeds + self.type_embedding.weight[0]
            prefix = torch.cat([lang_prefix, vis_agent, vis_hand, proprio_emb], dim=1)
        else:
            prefix = torch.cat([vis_agent, vis_hand, proprio_emb], dim=1)

        # 2. BOS token
        bos_ids = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        bos_emb = self.action_embedding(bos_ids) + self.type_embedding.weight[3]

        # 3. Initial sequence: prefix + BOS (base embeddings without pos encoding)
        seq = torch.cat([prefix, bos_emb], dim=1)  # (B, n_prefix+1, D)

        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            L = seq.size(1)
            x = seq + self.pos_embedding[:, :L]

            # Build generate-time mask (prefix bidir, action causal, no PAD)
            attn_mask = self._build_generate_mask(n_lang, lang_lengths, B, L, device)

            for block in self.blocks:
                x = block(x, attn_mask=attn_mask)
            x = self.norm(x)

            # Logits at last position
            logits = self.action_head(x[:, -1])  # (B, ACTION_HEAD_SIZE)

            # Prevent BOS from being sampled — it's a start marker, not a valid
            # action token.  Without this mask the model could (rarely) emit 1024
            # which the FAST BPE decoder doesn't expect.
            logits[:, BOS_TOKEN_ID] = -float("inf")

            if temperature <= 0:
                next_token = logits.argmax(dim=-1)
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

            # Append new token embedding (base, without pos encoding)
            new_emb = self.action_embedding(next_token.unsqueeze(1)) + self.type_embedding.weight[3]
            seq = torch.cat([seq, new_emb], dim=1)

        if not generated:
            return (torch.full((B, 1), PAD_TOKEN_ID, dtype=torch.long, device=device),
                    torch.zeros(B, dtype=torch.long, device=device))

        raw_tokens = torch.stack(generated, dim=1)  # (B, gen_len)

        # Compute lengths: count tokens before first EOS
        lengths = torch.full((B,), raw_tokens.size(1), dtype=torch.long, device=device)
        for i in range(B):
            eos_pos = (raw_tokens[i] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                lengths[i] = eos_pos[0].item()

        # Strip EOS/PAD: return only real FAST action tokens
        max_len_actual = lengths.max().item() if lengths.numel() > 0 else 0
        max_len_actual = max(max_len_actual, 1)
        tokens = torch.full((B, max_len_actual), PAD_TOKEN_ID, dtype=torch.long, device=device)
        for i in range(B):
            k = lengths[i].item()
            if k > 0:
                tokens[i, :k] = raw_tokens[i, :k]

        return tokens, lengths
