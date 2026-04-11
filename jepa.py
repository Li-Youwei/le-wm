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

    def encode(self, pixels_agent, pixels_hand, lang_input_ids, lang_attention_mask):
        """Encode both views + language.

        Args:
            pixels_agent: (B, C, H, W) agentview image.
            pixels_hand: (B, C, H, W) eye-in-hand image.
            lang_input_ids: (B, max_lang_tokens) T5 token IDs.
            lang_attention_mask: (B, max_lang_tokens) attention mask (1=real, 0=pad).

        Returns:
            z_agent: (B, D) agentview CLS token, projected.
            z_hand: (B, D) hand CLS token, projected.
            lang_embeds: (B, max_lang_tokens, D) projected language embeddings.
            lang_lengths: (B,) real language token count per sample.
        """
        # Visual: shared ViT encoder for both views
        agent_out = self.encoder(pixels_agent, interpolate_pos_encoding=True)
        z_agent = self.projector(agent_out.last_hidden_state[:, 0])  # CLS token

        hand_out = self.encoder(pixels_hand, interpolate_pos_encoding=True)
        z_hand = self.projector(hand_out.last_hidden_state[:, 0])  # CLS, same encoder

        # Language: frozen T5-small
        with torch.no_grad():
            lang_out = self.lang_encoder(
                input_ids=lang_input_ids,
                attention_mask=lang_attention_mask,
            )
        lang_embeds = self.lang_proj(lang_out.last_hidden_state)  # (B, seq_len, D)

        # Compute lang_lengths from attention_mask
        lang_lengths = lang_attention_mask.sum(dim=1)  # (B,)

        return z_agent, z_hand, lang_embeds, lang_lengths

    def predict(self, z_agent, z_hand, proprio, lang_embeds, lang_lengths,
                action_tokens, action_lengths):
        """Training: run predictor with teacher forcing.

        Args:
            z_agent: (B, D) agentview visual latent.
            z_hand: (B, D) hand visual latent.
            proprio: (B, 8) raw proprioceptive state.
            lang_embeds: (B, max_lang_tokens, D) language embeddings.
            lang_lengths: (B,) real language token count.
            action_tokens: (B, max_action_tokens) padded FAST token ids.
            action_lengths: (B,) real token count per sample.

        Returns:
            action_logits: (B, 1+max_action_tokens, 1026)
        """
        return self.predictor(
            z_agent, z_hand, proprio, lang_embeds, lang_lengths,
            action_tokens, action_lengths,
        )

    def predict_actions(self, z_agent, z_hand, proprio, lang_embeds, lang_lengths,
                        max_len=45, temperature=0.0):
        """Inference: autoregressively generate FAST action tokens.

        Returns:
            tokens: (B, gen_len) clean FAST token ids.
            lengths: (B,) real token count per sample.
        """
        return self.predictor.generate(
            z_agent, z_hand, proprio, lang_embeds, lang_lengths,
            max_len=max_len, temperature=temperature,
        )
