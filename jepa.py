"""JEPA Implementation — VLA baseline with dual-view + language + proprioception."""

import torch
from torch import nn


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        projector=None,
        lang_encoder=None,
        lang_proj=None,
    ):
        super().__init__()

        self.encoder = encoder          # Shared ViT encoder (both views)
        self.predictor = predictor      # ARPredictor instance
        self.projector = projector or nn.Identity()
        self.lang_encoder = lang_encoder  # T5EncoderModel, frozen
        self.lang_proj = lang_proj        # nn.Linear(T5_hidden, embed_dim)

    @property
    def use_language(self) -> bool:
        return self.lang_encoder is not None

    def train(self, mode=True):
        """Override to keep T5 encoder frozen in eval mode.

        PyTorch Lightning calls model.train() before each training step,
        which would recursively set all submodules to train mode —
        re-enabling T5's dropout. This override prevents that.
        """
        super().train(mode)
        if self.lang_encoder is not None:
            self.lang_encoder.eval()
        return self

    def encode(
        self,
        pixels_agent,
        pixels_hand,
        lang_input_ids=None,
        lang_attention_mask=None,
    ):
        """Encode both views + (optionally) language.

        Performance note: the two views are concatenated along the batch
        dimension and run through the SHARED ViT in **one** forward pass
        instead of two sequential calls. This roughly halves the visual
        wall-clock time (one CUDA kernel launch + better GPU utilization)
        and gives a BatchNorm projector 2× the samples per call to estimate
        running statistics.

        Args:
            pixels_agent: (B, C, H, W) agentview image.
            pixels_hand: (B, C, H, W) eye-in-hand image.
            lang_input_ids: (B, max_lang_tokens) T5 token IDs, or None for no-language mode.
            lang_attention_mask: (B, max_lang_tokens) attention mask (1=real, 0=pad), or None.

        Returns:
            z_agent: (B, D) agentview CLS token, projected.
            z_hand: (B, D) hand CLS token, projected.
            lang_embeds: (B, max_lang_tokens, D) projected language embeddings,
                or None when language is disabled.
            lang_lengths: (B,) real language token count per sample, or None.
        """
        # Visual: cat both views, single ViT pass, then split.
        visual_cat = torch.cat([pixels_agent, pixels_hand], dim=0)  # (2B, C, H, W)
        visual_out = self.encoder(visual_cat, interpolate_pos_encoding=True)
        visual_z = self.projector(visual_out.last_hidden_state[:, 0])  # (2B, D)
        z_agent, z_hand = visual_z.chunk(2, dim=0)  # (B, D), (B, D)

        # Language: only when encoder is present AND tokens are provided
        if self.lang_encoder is not None and lang_input_ids is not None:
            with torch.no_grad():
                lang_out = self.lang_encoder(
                    input_ids=lang_input_ids,
                    attention_mask=lang_attention_mask,
                )
            lang_embeds = self.lang_proj(lang_out.last_hidden_state)  # (B, seq_len, D)
            lang_lengths = lang_attention_mask.sum(dim=1)  # (B,)
        else:
            lang_embeds = None
            lang_lengths = None

        return z_agent, z_hand, lang_embeds, lang_lengths

    def predict(self, z_agent, z_hand, proprio, lang_embeds, lang_lengths,
                action_tokens, action_lengths):
        """Training: run predictor with teacher forcing.

        Args:
            z_agent: (B, D) agentview visual latent.
            z_hand: (B, D) hand visual latent.
            proprio: (B, 9) raw proprioceptive state
                [ee_pos(3) + xyzw_quat(4) + gripper_raw(2)].
            lang_embeds: (B, max_lang_tokens, D) language embeddings.
            lang_lengths: (B,) real language token count.
            action_tokens: (B, max_action_tokens) padded FAST token ids.
            action_lengths: (B,) real token count per sample.

        Returns:
            When the underlying predictor has ``use_state_prediction=False``:
                action_logits: (B, 1+max_action_tokens, 1026)
            When ``use_state_prediction=True``:
                Tuple ``(action_logits, pred_ag, pred_hd, pred_pr)``;
                see ``ARPredictor.forward`` for shapes.
        """
        return self.predictor(
            z_agent, z_hand, proprio, lang_embeds, lang_lengths,
            action_tokens, action_lengths,
        )

    def encode_future_visual(self, pixels_agent_future, pixels_hand_future):
        """Encode future-frame visual inputs through the SHARED ViT + projector.

        Used by the state-prediction branch (``use_state_prediction=True``).
        Following LeWM paper Section 3 — "We do not employ stop-gradient,
        exponential moving averages, or additional stabilization heuristics.
        Gradients are propagated through all components of the loss" — this
        method does NOT call ``.detach()`` on its outputs. The caller
        (train.py) must also avoid stop-grad. SIGReg on the encoder outputs
        is what prevents collapse, not stop-gradient.

        Performance note: same cat-then-split optimization as ``encode()``;
        the two future views go through the ViT in one forward pass. Total
        ViT calls per training step in SP mode = 2 (down from 4).

        Args:
            pixels_agent_future: (B, 3, H, W) agentview image at raw step t+H.
            pixels_hand_future: (B, 3, H, W) hand-cam image at raw step t+H.

        Returns:
            z_agent_future: (B, D) future agentview latent.
            z_hand_future: (B, D) future hand-cam latent.
        """
        visual_cat = torch.cat(
            [pixels_agent_future, pixels_hand_future], dim=0,
        )  # (2B, C, H, W)
        visual_out = self.encoder(visual_cat, interpolate_pos_encoding=True)
        visual_z = self.projector(visual_out.last_hidden_state[:, 0])  # (2B, D)
        z_agent_future, z_hand_future = visual_z.chunk(2, dim=0)
        return z_agent_future, z_hand_future

    def predict_actions(self, z_agent, z_hand, proprio, lang_embeds, lang_lengths,
                        max_len=80, temperature=0.0):
        """Inference: autoregressively generate FAST action tokens.

        Returns:
            tokens: (B, gen_len) clean FAST token ids.
            lengths: (B,) real token count per sample.
        """
        return self.predictor.generate(
            z_agent, z_hand, proprio, lang_embeds, lang_lengths,
            max_len=max_len, temperature=temperature,
        )
