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
        return statistic.mean()  # average over projections and time


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
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=causal
            )
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


class ModalityLayerNorm(nn.Module):
    """Apply a separate LayerNorm to each token modality."""

    def __init__(self, num_modalities: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
                for _ in range(num_modalities)
            ]
        )

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for modality, norm in enumerate(self.norms):
            mask = modality_ids == modality
            if mask.any():
                out[mask] = norm(x[mask])
        return out


class MoTAttention(nn.Module):
    """Meta MoT-style attention: modality-specific projections, global SDPA."""

    def __init__(
        self,
        dim: int,
        num_modalities: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.ModuleList(
            [nn.Linear(dim, inner_dim * 3, bias=False) for _ in range(num_modalities)]
        )
        self.to_out = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                for _ in range(num_modalities)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        inner_dim = self.to_qkv[0].out_features // 3

        q = torch.empty(B, T, inner_dim, device=device, dtype=x.dtype)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        for modality, proj in enumerate(self.to_qkv):
            mask = modality_ids == modality
            if mask.any():
                q_m, k_m, v_m = proj(x[mask]).chunk(3, dim=-1)
                q[mask] = q_m
                k[mask] = k_m
                v[mask] = v_m

        q = rearrange(q, "b t (h d) -> b h t d", h=self.heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.heads)
        drop = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=drop,
        )
        attn_out = rearrange(attn_out, "b h t d -> b t (h d)")

        out = torch.empty(B, T, self.to_out[0][0].out_features, device=device, dtype=x.dtype)
        for modality, proj in enumerate(self.to_out):
            mask = modality_ids == modality
            if mask.any():
                out[mask] = proj(attn_out[mask])
        return out


class MoTBlock(nn.Module):
    """Mixture-of-Transformers block with modality-specific non-embedding params.

    Mirrors Meta's MoT rule-based routing: each token uses parameters selected
    by its modality for layer normalization, QKV/O attention projections, and
    FFN, while attention is still computed globally over the full sequence.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        num_modalities: int = 4,
    ):
        super().__init__()
        self.norm1 = ModalityLayerNorm(num_modalities, dim, eps=1e-6)
        self.attn = MoTAttention(
            dim,
            num_modalities,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.norm2 = ModalityLayerNorm(num_modalities, dim, eps=1e-6)
        self.mlp = nn.ModuleList(
            [FeedForward(dim, mlp_dim, dropout=dropout) for _ in range(num_modalities)]
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, modality_ids), modality_ids, attn_mask=attn_mask)
        y = self.norm2(x, modality_ids)
        mlp_out = torch.empty_like(x)
        for modality, mlp in enumerate(self.mlp):
            mask = modality_ids == modality
            if mask.any():
                mlp_out[mask] = mlp(y[mask])
        return x + mlp_out


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

MOT_LANG = 0
MOT_VISUAL = 1
MOT_PROPRIO = 2
MOT_ACTION = 3
MOT_NUM_MODALITIES = 4


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

    ``state_prediction_arch="mot"`` enables a Meta MoT-style transformer:
    language, visual, proprio, and action tokens use modality-specific
    layer norms, attention projection matrices, and FFNs, while attention is
    still computed globally over the full sequence. STATE_QUERY tokens are
    routed by target modality: Q_ag/Q_hd use the visual expert, Q_pr uses the
    proprio expert.
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
        n_visual_tokens_per_view: int = 1,
        visual_pool_grid: int = 0,
        use_gripper_aux: bool = False,
        gripper_chunk_size: int = 20,
        state_prediction_arch: str = "shared",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_action_tokens = max_action_tokens
        self.max_lang_tokens = max_lang_tokens
        self.proprio_dim = proprio_dim
        self.use_state_prediction = use_state_prediction
        self.n_state_query = 3 if use_state_prediction else 0
        if state_prediction_arch not in ("shared", "mot"):
            raise ValueError(
                "state_prediction_arch must be 'shared' or 'mot', got "
                f"'{state_prediction_arch}'"
            )
        self.state_prediction_arch = state_prediction_arch
        self.use_mot_transformer = state_prediction_arch == "mot"

        # Visual-token layout (CLS-only by default, opt-in to CLS+pooled patches).
        # If visual_pool_grid > 0 the encoder is expected to deliver CLS + G*G
        # spatially pooled patches per view; n_visual_tokens_per_view tracks the
        # final per-view count used everywhere in the predictor (1, attn mask,
        # pos embedding, prefix length).
        if visual_pool_grid > 0:
            self.n_visual_per_view = 1 + visual_pool_grid * visual_pool_grid
        else:
            self.n_visual_per_view = max(1, int(n_visual_tokens_per_view))
        self.visual_pool_grid = int(visual_pool_grid)
        nv = self.n_visual_per_view

        # max_seq_len = lang + 2*nv (visual prefix) + 1 (proprio) + 1 (BOS) + actions
        # (+ Q_ag + Q_hd + Q_pr when use_state_prediction is True).
        self.max_seq_len = (
            max_lang_tokens + 2 * nv + 1 + 1 + max_action_tokens + self.n_state_query
        )

        # Proprioception encoder: proprio_dim (9d for LIBERO) → embed_dim
        self.proprio_encoder = MLP(proprio_dim, embed_dim, embed_dim)

        # Token embeddings for action tokens (FAST vocab + BOS/EOS/PAD)
        self.action_embedding = nn.Embedding(
            TOTAL_VOCAB_SIZE, embed_dim
        )  # 1027 entries

        # Type embeddings: 0=language, 1=visual, 2=proprioception, 3=action,
        # (4=state_query when use_state_prediction is True).
        # Size depends on the flag so baseline checkpoints (4-row table) load
        # cleanly into a baseline-arch model.
        n_type = 5 if use_state_prediction else 4
        self.type_embedding = nn.Embedding(n_type, embed_dim)

        # View embedding (agent=0, hand=1). Only constructed when nv > 1 so
        # baseline single-CLS checkpoints stay binary-compatible.
        if nv > 1:
            self.view_embedding = nn.Embedding(2, embed_dim)
            # Independent 2D positional grids per view (agent / hand cameras
            # see different image spaces; sharing would underfit). Stored as
            # (G, G, D) so it broadcasts naturally over the patch sequence
            # after a flatten.
            G = self.visual_pool_grid
            self.agent_patch_2d_pos = nn.Parameter(torch.randn(G, G, embed_dim) * 0.02)
            self.hand_patch_2d_pos = nn.Parameter(torch.randn(G, G, embed_dim) * 0.02)

        # Positional encoding (learnable, 1D over the unified sequence)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))

        # Action classification head: predict vocab 0..1025 (PAD excluded from targets)
        self.action_head = nn.Linear(embed_dim, ACTION_HEAD_SIZE)

        # Auxiliary gripper-command head. When enabled, predicts a (H,)
        # gripper sequence directly from the BOS hidden state via a small MLP,
        # bypassing FAST tokenization. Designed to give the gripper dim a clean
        # signal that doesn't get diluted by the 6 spatial dims under FAST's
        # joint BPE. Diagnostic on 4-suite sp_sigreg showed predicted gripper
        # command oscillating ±1 (correct GT is constant -1 for early chunks)
        # → bypass-with-direct-regression head is the surgical fix.
        # The head reads `x[:, n_prefix]` (BOS position): BOS attends only to
        # the prefix (lang + visual + proprio) under our causal mask, so the
        # head's input is identical at train (full sequence) and inference
        # (prefix-only forward) — no exposure-bias mismatch.
        self.use_gripper_aux = use_gripper_aux
        self.gripper_chunk_size = int(gripper_chunk_size)
        if use_gripper_aux:
            # Tanh output → guaranteed [-1, 1] range matching OSC gripper cmd.
            self.gripper_aux_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, self.gripper_chunk_size),
                nn.Tanh(),
            )

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
            self.state_query_embeddings = nn.Parameter(torch.randn(3, embed_dim))
            # Per-stream projector: matches the encoder-side projector
            # signature (LeWM paper Sec. 3 + upstream pattern).
            self.state_pred_head_ag = MLP(
                embed_dim,
                2048,
                embed_dim,
                norm_fn=head_norm_fn,
            )
            self.state_pred_head_hd = MLP(
                embed_dim,
                2048,
                embed_dim,
                norm_fn=head_norm_fn,
            )
            self.state_pred_head_pr = MLP(
                embed_dim,
                2048,
                proprio_dim,
                norm_fn=head_norm_fn,
            )
        # Transformer blocks. The Meta MoT variant keeps global attention over
        # the full sequence but routes non-embedding parameters by token
        # modality inside each block.
        block_cls = MoTBlock if self.use_mot_transformer else Block
        if self.use_mot_transformer:
            self.blocks = nn.ModuleList(
                [
                    block_cls(
                        embed_dim,
                        heads,
                        dim_head,
                        mlp_dim,
                        dropout,
                        num_modalities=MOT_NUM_MODALITIES,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm = ModalityLayerNorm(MOT_NUM_MODALITIES, embed_dim)
        else:
            self.blocks = nn.ModuleList(
                [
                    block_cls(embed_dim, heads, dim_head, mlp_dim, dropout)
                    for _ in range(depth)
                ]
            )
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
        # Prefix layout: lang(n_lang) + agent(nv) + hand(nv) + proprio(1) = n_lang + 2*nv + 1
        # where nv = self.n_visual_per_view (1 by default, 17 for CLS+4x4 pool).
        nv = self.n_visual_per_view
        n_prefix = n_lang + 2 * nv + 1
        action_start = (
            n_prefix + 1
        )  # BOS is at n_prefix, action tokens start at n_prefix+1

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
            is_real[:, action_start:action_end] = action_tokens != PAD_TOKEN_ID

        # STATE_QUERY tokens (Q_ag, Q_hd, Q_pr) are always real when present.
        query_start = action_end
        query_end = query_start + self.n_state_query
        if self.n_state_query > 0:
            is_real[:, query_start:query_end] = True

        # --- Zone membership ---
        in_prefix = pos < n_prefix  # (L,)
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
        mask |= (
            action_real.unsqueeze(2) & action_real.unsqueeze(1) & causal.unsqueeze(0)
        )

        # Rule 4: STATE_QUERY rows (Q_ag, Q_hd, Q_pr) — only when enabled.
        # Each Q sees: real prefix + real action zone (BOS + non-PAD T) + itself.
        # Each Q does NOT see: PAD, OTHER queries (preserves 3-way independence
        # so pred_pr cannot peek at pred_ag's hidden state, etc.). No token
        # outside the query group attends to a query (q columns stay all-False).
        if self.n_state_query > 0:
            in_query = (pos >= query_start) & (pos < query_end)  # (L,)
            query_real = is_real & in_query.unsqueeze(0)  # (B, L)

            # Q sees real prefix
            mask |= query_real.unsqueeze(2) & prefix_real.unsqueeze(1)
            # Q sees real action zone (BOS + non-PAD action tokens)
            mask |= query_real.unsqueeze(2) & action_real.unsqueeze(1)
            # Q sees only itself within the query block — diagonal, NOT bidir
            eye_L = torch.eye(L, dtype=torch.bool, device=device)
            mask |= (
                query_real.unsqueeze(2) & query_real.unsqueeze(1) & eye_L.unsqueeze(0)
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
        nv = self.n_visual_per_view
        n_prefix = n_lang + 2 * nv + 1
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
        mask |= (
            action_positions.unsqueeze(2)
            & action_positions.unsqueeze(1)
            & causal.unsqueeze(0)
        )

        return mask.unsqueeze(1)

    def _build_train_modality_ids(
        self,
        B: int,
        n_lang: int,
        nv: int,
        include_queries: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-token MoT modality ids for teacher-forced training."""
        chunks = []
        if n_lang > 0:
            chunks.append(
                torch.full((B, n_lang), MOT_LANG, device=device, dtype=torch.long)
            )
        chunks.extend(
            [
                torch.full((B, nv), MOT_VISUAL, device=device, dtype=torch.long),
                torch.full((B, nv), MOT_VISUAL, device=device, dtype=torch.long),
                torch.full((B, 1), MOT_PROPRIO, device=device, dtype=torch.long),
                torch.full(
                    (B, 1 + self.max_action_tokens),
                    MOT_ACTION,
                    device=device,
                    dtype=torch.long,
                ),
            ]
        )
        if include_queries:
            query_ids = torch.tensor(
                [MOT_VISUAL, MOT_VISUAL, MOT_PROPRIO],
                device=device,
                dtype=torch.long,
            ).expand(B, -1)
            chunks.append(query_ids)
        return torch.cat(chunks, dim=1)

    def _build_generate_modality_ids(
        self,
        B: int,
        n_lang: int,
        nv: int,
        n_action_positions: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-token MoT modality ids for generation-time sequences."""
        chunks = []
        if n_lang > 0:
            chunks.append(
                torch.full((B, n_lang), MOT_LANG, device=device, dtype=torch.long)
            )
        chunks.extend(
            [
                torch.full((B, nv), MOT_VISUAL, device=device, dtype=torch.long),
                torch.full((B, nv), MOT_VISUAL, device=device, dtype=torch.long),
                torch.full((B, 1), MOT_PROPRIO, device=device, dtype=torch.long),
                torch.full(
                    (B, n_action_positions),
                    MOT_ACTION,
                    device=device,
                    dtype=torch.long,
                ),
            ]
        )
        return torch.cat(chunks, dim=1)

    def _run_blocks(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run shared or MoT-routed transformer blocks."""
        if self.use_mot_transformer:
            assert modality_ids is not None
            for block in self.blocks:
                x = block(x, modality_ids, attn_mask=attn_mask)
            return self.norm(x, modality_ids)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.norm(x)

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

        # Accept either (B, D) — legacy CLS-only encoder output — or (B, N, D)
        # multi-token output from the CLS+pooled-patches encoder. Normalize to
        # the (B, N, D) layout below.
        if z_agent.dim() == 2:
            z_agent = z_agent.unsqueeze(1)  # (B, 1, D)
        if z_hand.dim() == 2:
            z_hand = z_hand.unsqueeze(1)  # (B, 1, D)
        nv = z_agent.size(1)
        if nv != self.n_visual_per_view:
            raise ValueError(
                f"ARPredictor.forward got z_agent with N_visual={nv}, but the "
                f"predictor was constructed with n_visual_per_view="
                f"{self.n_visual_per_view}. Check JEPA.encode visual_pool_grid."
            )

        # 1. Encode proprioception
        z_proprio = self.proprio_encoder(z_proprio_raw)  # (B, D)

        # 2. Build prefix embeddings with type embeddings.
        # Visual: type=1; when nv>1 also add per-view embedding + per-patch 2D
        # positional embedding. CLS (index 0 within the view) only gets the
        # view embedding — the 2D pos grid only covers the G*G patch positions.
        vis_agent = z_agent + self.type_embedding.weight[1]  # (B, nv, D)
        vis_hand = z_hand + self.type_embedding.weight[1]  # (B, nv, D)
        if nv > 1:
            G = self.visual_pool_grid
            assert G * G == nv - 1, (
                f"n_visual_per_view-1 ({nv - 1}) must equal G*G ({G * G})"
            )
            # View embedding: (1, 1, D), broadcast over the (B, nv, D) sequence.
            vis_agent = vis_agent + self.view_embedding.weight[0].view(1, 1, -1)
            vis_hand = vis_hand + self.view_embedding.weight[1].view(1, 1, -1)
            # 2D positional embedding on patch tokens only (skip CLS at idx 0).
            agent_pos_2d = self.agent_patch_2d_pos.view(G * G, -1)  # (G*G, D)
            hand_pos_2d = self.hand_patch_2d_pos.view(G * G, -1)
            vis_agent = vis_agent.clone()
            vis_hand = vis_hand.clone()
            vis_agent[:, 1:] = vis_agent[:, 1:] + agent_pos_2d.unsqueeze(0)
            vis_hand[:, 1:] = vis_hand[:, 1:] + hand_pos_2d.unsqueeze(0)
        proprio_emb = (
            z_proprio.unsqueeze(1) + self.type_embedding.weight[2]
        )  # (B, 1, D)

        if lang_embeds is not None:
            lang_prefix = lang_embeds + self.type_embedding.weight[0]  # (B, n_lang, D)
            prefix = torch.cat(
                [lang_prefix, vis_agent, vis_hand, proprio_emb], dim=1
            )  # (B, n_lang + 2*nv + 1, D)
        else:
            prefix = torch.cat(
                [vis_agent, vis_hand, proprio_emb], dim=1
            )  # (B, 2*nv + 1, D)

        # 3. Build action embeddings (BOS + action tokens) with type embedding
        bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        bos_action = torch.cat([bos, action_tokens], dim=1)  # (B, 1+max_action_tokens)
        action_emb = self.action_embedding(bos_action) + self.type_embedding.weight[3]

        # 4. Concatenate prefix + action zones.
        x = torch.cat(
            [prefix, action_emb], dim=1
        )  # (B, n_prefix + 1 + max_action_tokens, D)

        # 4b. (Optional) Append STATE_QUERY tokens at the very end. Each
        # query is a learnable embedding plus the type embedding (index 4).
        # Index 4 only exists when use_state_prediction is True (n_type=5).
        if self.use_state_prediction:
            # state_query_embeddings: (3, D) → (1, 3, D) → (B, 3, D)
            q_base = self.state_query_embeddings.unsqueeze(0).expand(B, -1, -1)
            q_emb = q_base + self.type_embedding.weight[4]  # broadcast over (B, 3, D)
            x = torch.cat([x, q_emb], dim=1)

        L = x.size(1)
        modality_ids = (
            self._build_train_modality_ids(
                B,
                n_lang,
                nv,
                include_queries=self.use_state_prediction,
                device=device,
            )
            if self.use_mot_transformer
            else None
        )

        # 5. Positional embedding + dropout
        x = x + self.pos_embedding[:, :L]
        x = self.dropout(x)

        # 6. Build hybrid attention mask (extended with query rules when SP is on)
        attn_mask = self._build_attn_mask(
            n_lang, lang_lengths, action_tokens, L, device
        )

        # 7. Transformer blocks
        x = self._run_blocks(x, attn_mask, modality_ids)

        # 8. Extract action zone output: BOS + action token positions
        n_prefix = n_lang + 2 * nv + 1
        action_output = x[
            :, n_prefix : n_prefix + 1 + self.max_action_tokens
        ]  # (B, 1+max_action_tokens, D)
        action_logits = self.action_head(
            action_output
        )  # (B, 1+max_action_tokens, ACTION_HEAD_SIZE)

        # 8b. Optional gripper-aux read-out from BOS hidden state.
        # x[:, n_prefix] is BOS — under our causal mask it only attends to the
        # prefix tokens (lang+visual+proprio), so this is identical to the
        # value the aux head sees during inference where no action tokens have
        # been emitted yet.
        pred_grip = None
        if self.use_gripper_aux:
            pred_grip = self.gripper_aux_head(x[:, n_prefix])  # (B, gripper_chunk_size)

        if not self.use_state_prediction:
            if self.use_gripper_aux:
                return action_logits, pred_grip
            return action_logits

        # 9. Read out the three STATE_QUERY positions and run them through
        #    their respective heads. Q_ag/Q_hd predict latent (D,), Q_pr
        #    predicts the raw 9d proprio vector at t+H (NOT an embedding —
        #    the target is the un-encoded proprio so the loss is in physical
        #    units).
        query_start = n_prefix + 1 + self.max_action_tokens
        q_out = x[:, query_start : query_start + 3]  # (B, 3, D)
        pred_ag = self.state_pred_head_ag(q_out[:, 0])  # (B, D)
        pred_hd = self.state_pred_head_hd(q_out[:, 1])  # (B, D)
        pred_pr = self.state_pred_head_pr(q_out[:, 2])  # (B, proprio_dim)

        if self.use_gripper_aux:
            return action_logits, pred_ag, pred_hd, pred_pr, pred_grip
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

        # Normalize visual input layout (same handling as forward()).
        if z_agent.dim() == 2:
            z_agent = z_agent.unsqueeze(1)
        if z_hand.dim() == 2:
            z_hand = z_hand.unsqueeze(1)
        nv = z_agent.size(1)
        if nv != self.n_visual_per_view:
            raise ValueError(
                f"ARPredictor.generate got z_agent with N_visual={nv}, but the "
                f"predictor was constructed with n_visual_per_view="
                f"{self.n_visual_per_view}."
            )

        # 1. Build prefix embeddings (same as forward — view + 2D pos when nv>1)
        z_proprio = self.proprio_encoder(z_proprio_raw)
        vis_agent = z_agent + self.type_embedding.weight[1]
        vis_hand = z_hand + self.type_embedding.weight[1]
        if nv > 1:
            G = self.visual_pool_grid
            vis_agent = vis_agent + self.view_embedding.weight[0].view(1, 1, -1)
            vis_hand = vis_hand + self.view_embedding.weight[1].view(1, 1, -1)
            agent_pos_2d = self.agent_patch_2d_pos.view(G * G, -1)
            hand_pos_2d = self.hand_patch_2d_pos.view(G * G, -1)
            vis_agent = vis_agent.clone()
            vis_hand = vis_hand.clone()
            vis_agent[:, 1:] = vis_agent[:, 1:] + agent_pos_2d.unsqueeze(0)
            vis_hand[:, 1:] = vis_hand[:, 1:] + hand_pos_2d.unsqueeze(0)
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
            modality_ids = (
                self._build_generate_modality_ids(
                    B,
                    n_lang,
                    nv,
                    n_action_positions=L - (n_lang + 2 * nv + 1),
                    device=device,
                )
                if self.use_mot_transformer
                else None
            )
            x = self._run_blocks(x, attn_mask, modality_ids)

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
            next_token = torch.where(
                finished, torch.tensor(PAD_TOKEN_ID, device=device), next_token
            )
            generated.append(next_token)

            # Track EOS
            finished = finished | (next_token == EOS_TOKEN_ID)
            if finished.all():
                break

            # Append new token embedding (base, without pos encoding)
            new_emb = (
                self.action_embedding(next_token.unsqueeze(1))
                + self.type_embedding.weight[3]
            )
            seq = torch.cat([seq, new_emb], dim=1)

        if not generated:
            return (
                torch.full((B, 1), PAD_TOKEN_ID, dtype=torch.long, device=device),
                torch.zeros(B, dtype=torch.long, device=device),
            )

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
        tokens = torch.full(
            (B, max_len_actual), PAD_TOKEN_ID, dtype=torch.long, device=device
        )
        for i in range(B):
            k = lengths[i].item()
            if k > 0:
                tokens[i, :k] = raw_tokens[i, :k]

        return tokens, lengths

    @torch.no_grad()
    def predict_gripper_aux(
        self,
        z_agent: torch.Tensor,
        z_hand: torch.Tensor,
        z_proprio_raw: torch.Tensor,
        lang_embeds: torch.Tensor | None,
        lang_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inference-time read-out of the gripper aux head.

        Runs a prefix+BOS-only forward (no action tokens emitted yet) and
        reads the BOS hidden state through ``self.gripper_aux_head``. Under
        our causal attention mask BOS only attends to prefix tokens
        (lang + visual + proprio), so this hidden state is identical to the
        value the head sees during teacher-forcing training — no exposure-
        bias mismatch.
        """
        if not self.use_gripper_aux:
            raise RuntimeError(
                "predict_gripper_aux() called but use_gripper_aux=False."
            )

        B = z_agent.size(0)
        device = z_agent.device
        n_lang = lang_embeds.size(1) if lang_embeds is not None else 0

        if z_agent.dim() == 2:
            z_agent = z_agent.unsqueeze(1)
        if z_hand.dim() == 2:
            z_hand = z_hand.unsqueeze(1)
        nv = z_agent.size(1)
        if nv != self.n_visual_per_view:
            raise ValueError(
                f"ARPredictor.predict_gripper_aux got z_agent with N_visual="
                f"{nv}, but the predictor was constructed with "
                f"n_visual_per_view={self.n_visual_per_view}."
            )

        z_proprio = self.proprio_encoder(z_proprio_raw)
        vis_agent = z_agent + self.type_embedding.weight[1]
        vis_hand = z_hand + self.type_embedding.weight[1]
        if nv > 1:
            G = self.visual_pool_grid
            vis_agent = vis_agent + self.view_embedding.weight[0].view(1, 1, -1)
            vis_hand = vis_hand + self.view_embedding.weight[1].view(1, 1, -1)
            agent_pos_2d = self.agent_patch_2d_pos.view(G * G, -1)
            hand_pos_2d = self.hand_patch_2d_pos.view(G * G, -1)
            vis_agent = vis_agent.clone()
            vis_hand = vis_hand.clone()
            vis_agent[:, 1:] = vis_agent[:, 1:] + agent_pos_2d.unsqueeze(0)
            vis_hand[:, 1:] = vis_hand[:, 1:] + hand_pos_2d.unsqueeze(0)
        proprio_emb = z_proprio.unsqueeze(1) + self.type_embedding.weight[2]

        if lang_embeds is not None:
            lang_prefix = lang_embeds + self.type_embedding.weight[0]
            prefix = torch.cat([lang_prefix, vis_agent, vis_hand, proprio_emb], dim=1)
        else:
            prefix = torch.cat([vis_agent, vis_hand, proprio_emb], dim=1)

        bos_ids = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        bos_emb = self.action_embedding(bos_ids) + self.type_embedding.weight[3]
        seq = torch.cat([prefix, bos_emb], dim=1)  # (B, n_prefix+1, D)

        L = seq.size(1)
        x = seq + self.pos_embedding[:, :L]
        attn_mask = self._build_generate_mask(n_lang, lang_lengths, B, L, device)
        modality_ids = (
            self._build_generate_modality_ids(
                B,
                n_lang,
                nv,
                n_action_positions=L - (n_lang + 2 * nv + 1),
                device=device,
            )
            if self.use_mot_transformer
            else None
        )
        x = self._run_blocks(x, attn_mask, modality_ids)

        n_prefix = n_lang + 2 * nv + 1
        return self.gripper_aux_head(x[:, n_prefix])  # (B, gripper_chunk_size)
