"""JEPA Implementation — modified for unified action prediction + world model."""

import torch
from einops import rearrange
from torch import nn


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor  # Modified ARPredictor instance
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations into visual embeddings.
        info: dict with pixels key, shape (B, T, C, H, W)
        """
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # CLS token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        return info

    def predict(self, z_t, action_tokens, action_lengths):
        """Training: run predictor with teacher forcing.

        Args:
            z_t: (B, D) visual latent from encoder.
            action_tokens: (B, max_action_tokens) padded FAST token ids.
            action_lengths: (B,) real token count per sample.

        Returns:
            action_logits: (B, 1+max_action_tokens, 1026)
            state_pred: (B, D) projected state prediction.
        """
        action_logits, state_raw = self.predictor(z_t, action_tokens, action_lengths)
        state_pred = self.pred_proj(state_raw)
        return action_logits, state_pred

    def predict_actions(self, z_t, max_len=45, temperature=0.0):
        """Inference: autoregressively generate FAST action tokens.

        Args:
            z_t: (B, D) visual latent from encoder.
            max_len: max tokens to generate.
            temperature: 0.0=greedy, >0=sampling.

        Returns:
            tokens: (B, gen_len) clean FAST token ids.
            lengths: (B,) real token count per sample.
        """
        return self.predictor.generate(z_t, max_len, temperature)

    # NOTE: rollout(), criterion(), get_cost() removed.
    # They were CEM planning methods that referenced the deleted action_encoder
    # and old predict(emb, act_emb) signature. With autoregressive action
    # generation, CEM rollout is no longer needed.
