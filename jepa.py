"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    # MODIFIED: removed action_encoder (FAST tokens are pre-computed, not encoded here)
    def __init__(
        self,
        encoder,
        predictor,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor  # UnifiedPredictor instance
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    # MODIFIED: removed action encoding (FAST tokens are pre-computed by preprocessing)
    def encode(self, info):
        """Encode observations into visual embeddings.
        info: dict with pixels key
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        return info

    # MODIFIED: new signature for UnifiedPredictor
    def predict(self, z_t, action_tokens, action_lengths):
        """Run unified predictor: action token logits + state prediction.

        Args:
            z_t: (B, D) single-frame visual latent (after projector).
            action_tokens: (B, max_action_tokens) padded FAST token ids.
            action_lengths: (B,) real token count per sample.

        Returns:
            action_logits: (B, 1+max_action_tokens, 1026)
            state_pred: (B, D) projected state prediction.
        """
        action_logits, state_raw = self.predictor(z_t, action_tokens, action_lengths)
        state_pred = self.pred_proj(state_raw)  # MLP: (B, D) -> (B, D)
        return action_logits, state_pred

    # NEW: autoregressive action generation for inference
    def predict_actions(self, z_t, max_len=40, temperature=0.0):
        """Generate FAST action tokens autoregressively.

        Args:
            z_t: (B, D) visual latent (after projector).
            max_len: max tokens to generate.
            temperature: 0.0=greedy, >0=sampling.

        Returns:
            tokens: (B, gen_len) generated token ids.
            lengths: (B,) real token count (EOS exclusive).
        """
        return self.predictor.generate(z_t, max_len, temperature)

    # NOTE: rollout(), criterion(), get_cost() removed.
    # They were CEM planning methods that called self.action_encoder (now deleted)
    # and the old predict(emb, act_emb) signature (now changed).
    # With direct autoregressive action generation, CEM rollout is no longer needed.
    # See predict_actions() above for the new inference path.
