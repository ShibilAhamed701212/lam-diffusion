"""Dual-Stream Conditioning Pathway (paper §IV-C).

Stream 1, Cross-Attention Conditioning: in every cross-attention layer of the U-Net (all
resolutions, down / mid / up, i.e. "multi-resolution cross-attention with action embeddings") the
character action tokens join the global descriptor tokens in ONE attention:

    out = softmax(Q [K_global ; K_action]^T / sqrt(d) + [0 ; region_bias + g]) [V_global ; V_action]

K/V_action = the layer's text K/V weights + a trainable low-rank update (zero-initialised), so the
action tokens are read like text tokens. Each character's tokens are visible only from latent
positions inside its box (a layout prior, cf. paper Fig. 3), which binds attributes to the right
person; outside every box the layer is exactly the pre-trained one. g is a learned per-layer
logit bias on the action tokens (starts at 0; our addition, logged as the "gate").

A first version ADDED a separate action attention on top of the text attention (gate 1). That
doubled the signal inside boxes and turned images muddy as the tokens grew during training
(reports/milestones/m3_diag); sharing one softmax keeps activations in the pre-trained range.

Stream 2, Temporal Attention Module: self-attention over each character's ordered sub-action
tokens; its context vector is fused into that character's tokens before injection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from lam.encoder import CTX_DIM, LAMOutput, encoder_layers

REGION_MARGIN = 0.04
MASKED = -1e4


class TemporalAttention(nn.Module):
    """Stream 2: self-attention over the time dimension of the sub-action sequence."""

    def __init__(self, d: int, out_dim: int = CTX_DIM, layers: int = 2):
        super().__init__()
        self.enc = encoder_layers(d, layers)
        self.out = nn.Linear(d, out_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def context(self, steps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Temporally coherent context vector before the fusion gate (what stage 2a supervises)."""
        h = self.enc(steps, src_key_padding_mask=~valid)
        m = valid.unsqueeze(-1).to(h.dtype)
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1))

    def forward(self, steps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """steps [N, T, d] (already carry temporal position embeddings), valid [N, T] -> [N, out_dim]"""
        return torch.tanh(self.gate) * self.context(steps, valid)


@dataclass
class ActionState:
    """Per-forward data shared by all cross-attention processors."""

    tokens: torch.Tensor | None = None      # [B, Na, 2048] character tokens, Na = C * TOKENS_PER_CHAR
    boxes: torch.Tensor | None = None       # [B, Na, 4]
    valid: torch.Tensor | None = None       # [B, Na] bool
    latent_hw: tuple[int, int] = (128, 128)
    region_masking: bool = True
    scale: float = 1.0                      # inference-time action strength (0 = off) / ablation switch
    bias_cache: dict = field(default_factory=dict)

    def set(self, lam: LAMOutput, latent_hw: tuple[int, int]) -> None:
        B, C, T = lam.char_token_mask.shape
        self.tokens = lam.char_tokens.reshape(B, C * T, CTX_DIM)
        self.boxes = lam.bbox.repeat_interleave(T, dim=1)
        self.valid = lam.char_token_mask.reshape(B, C * T)
        self.latent_hw = latent_hw
        self.bias_cache = {}

    def bias(self, hw: int, dtype: torch.dtype) -> torch.Tensor:
        """Additive attention bias [B, 1, hw, Na]: 0 where a query may attend a token, MASKED elsewhere."""
        key = (hw, dtype)
        if key not in self.bias_cache:
            H0, W0 = self.latent_hw
            s = math.sqrt(H0 * W0 / hw)
            h, w = round(H0 / s), round(W0 / s)
            assert h * w == hw, f"cannot infer feature map shape for {hw} tokens from latent {H0}x{W0}"
            dev = self.tokens.device
            gy = (torch.arange(h, device=dev) + 0.5) / h
            gx = (torch.arange(w, device=dev) + 0.5) / w
            gy, gx = gy[:, None].expand(h, w).reshape(-1), gx[None, :].expand(h, w).reshape(-1)
            b = self.boxes[:, None]  # [B, 1, Na, 4]
            allowed = self.valid[:, None, :].expand(-1, hw, -1).clone()
            if self.region_masking:
                inside = ((gx[None, :, None] >= b[..., 0] - REGION_MARGIN) & (gx[None, :, None] <= b[..., 2] + REGION_MARGIN)
                          & (gy[None, :, None] >= b[..., 1] - REGION_MARGIN) & (gy[None, :, None] <= b[..., 3] + REGION_MARGIN))
                allowed &= inside
            bias = torch.zeros(allowed.shape, dtype=dtype, device=dev).masked_fill(~allowed, MASKED)
            self.bias_cache[key] = bias.unsqueeze(1)
        return self.bias_cache[key]


class DualStreamAttnProcessor(nn.Module):
    """Replaces a cross-attention (attn2) processor in the SDXL U-Net."""

    def __init__(self, attn, state: ActionState, rank: int = 64):
        super().__init__()
        inner, cross = attn.to_k.out_features, attn.to_k.in_features
        # K/V_action = the layer's frozen text K/V weights + a trainable low-rank update (up
        # projections zero-initialised), so action tokens start out read like text tokens.
        self.k_down, self.k_up = nn.Linear(cross, rank, bias=False), nn.Linear(rank, inner, bias=False)
        self.v_down, self.v_up = nn.Linear(cross, rank, bias=False), nn.Linear(rank, inner, bias=False)
        for down, up in ((self.k_down, self.k_up), (self.v_down, self.v_up)):
            nn.init.normal_(down.weight, std=1.0 / rank)
            nn.init.zeros_(up.weight)
        # Learned logit bias on the action tokens (0 = read exactly like text tokens).
        self.gate = nn.Parameter(torch.zeros(1))
        self.state = state
        self.last_map: torch.Tensor | None = None  # optional attention capture for interpretability
        self.capture = False

    @staticmethod
    def _base(linear) -> torch.Tensor:
        return getattr(linear, "base_layer", linear).weight

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, ch, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b, ch, hh * ww).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        B, HW, _ = hidden_states.shape
        heads = attn.heads

        q = attn.to_q(hidden_states)
        ctx = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attn.norm_cross:
            ctx = attn.norm_encoder_hidden_states(ctx)
        k, v = attn.to_k(ctx), attn.to_v(ctx)
        hd = k.shape[-1] // heads

        def split(x):
            return x.view(B, -1, heads, hd).transpose(1, 2)

        st = self.state
        if st.tokens is None or st.scale <= 0:
            out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=attention_mask)
        else:
            toks = st.tokens.to(q.dtype)
            ka = F.linear(toks, self._base(attn.to_k).to(q.dtype)) + self.k_up(self.k_down(toks.to(self.k_down.weight.dtype))).to(q.dtype)
            va = F.linear(toks, self._base(attn.to_v).to(q.dtype)) + self.v_up(self.v_down(toks.to(self.v_down.weight.dtype))).to(q.dtype)
            region = st.bias(HW, q.dtype) + (self.gate + math.log(st.scale)).to(q.dtype)  # [B, 1, HW, Na]
            bias = torch.cat([torch.zeros(B, 1, HW, k.shape[1], dtype=q.dtype, device=q.device), region], -1)
            if attention_mask is not None:
                bias[..., :k.shape[1]] += attention_mask
            qh, kh = split(q), split(torch.cat([k, ka], 1))
            out = F.scaled_dot_product_attention(qh, kh, split(torch.cat([v, va], 1)), attn_mask=bias)
            if self.capture:
                with torch.no_grad():
                    probs = ((qh @ kh.transpose(-1, -2)) / math.sqrt(hd) + bias).softmax(-1).mean(1)
                    self.last_map = probs[..., k.shape[1]:]  # [B, HW, Na] attention mass on each action token

        out = out.transpose(1, 2).reshape(B, HW, heads * hd).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, ch, hh, ww)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


class DualStreamConditioning(nn.Module):
    """Owns the per-layer processors and the shared state."""

    def __init__(self, unet):
        super().__init__()
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.state = ActionState()
        procs, names = {}, []
        for name, module in unet.named_modules():
            if name.endswith("attn2") and hasattr(module, "to_k"):
                names.append(name)
        modules = dict(unet.named_modules())
        self.processors = nn.ModuleDict()
        for name, proc in unet.attn_processors.items():
            layer = name.rsplit(".processor", 1)[0]
            if layer in names:
                p = DualStreamAttnProcessor(modules[layer], self.state)
                self.processors[layer.replace(".", "__")] = p
                procs[name] = p
            else:
                procs[name] = AttnProcessor2_0()
        unet.set_attn_processor(procs)

    def bind(self, lam: LAMOutput, latent_hw: tuple[int, int]) -> None:
        self.state.set(lam, latent_hw)

    def gates(self) -> torch.Tensor:
        return torch.stack([p.gate.detach().squeeze() for p in self.processors.values()])
