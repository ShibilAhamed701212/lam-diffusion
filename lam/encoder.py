"""LAM Encoder (paper §IV-B): structured action descriptors -> action embeddings B x N x D_action.

- Pre-trained transformer backbone: SDXL's two CLIP text transformers (GPT-style, pre-trained
  on a large image-text corpus; fine-tuned with LoRA on anime action-pose-text triplets). Each
  free-text attribute field is encoded *separately*, so no prompt-tag string is ever built.
- Action Semantic Embedding:
    * Pose Encoding: transformer over 18 keypoint tokens + bbox -> pose tokens
    * Action Dynamics Encoding: sub-action embeddings + kinematic features from the skeleton
    * Style Attribute Fusion: identity + pose + dynamics + attribute-text tokens -> fusion
      transformer -> Perceiver resampler -> fixed tokens per character
- Output: global tokens (scene/style/props, 77 x 2048, the U-Net's native context), per-character
  action tokens (C x TOKENS_PER_CHAR x 2048) with a per-token mask, pooled vectors for SDXL's added
  conditioning and for L_action-img.

Character tokens = 4 identity tokens (gender, age, framing, view) + each attribute field's own token
sequence from the backbone (FIELD_SEQ word tokens per field, not an average: the U-Net reads them at
the scale and in the form of text tokens, so "blonde hair" is legible from step 0) + the learned
Action Semantic Embedding (pose, dynamics, temporal context) added to every token as a residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from lam.descriptor import (ACTIONS, AGES, FRAMINGS, GENDERS, LABEL_NAMES, MAX_ACTIONS, MAX_CHARS,
                            N_KEYPOINTS, TEXT_FIELDS, VIEWS, DescriptorBatch)

CTX_DIM = 2048        # SDXL cross-attention context dim (CLIP-L 768 + OpenCLIP-bigG 1280)
POOLED_DIM = 1280     # SDXL added-condition text embedding
CLIP_IMG_DIM = 768    # CLIP ViT-L/14 projection dim, target space of L_action-img
FIELD_LEN = 16        # tokens per free-text attribute field (backbone input length)
FIELD_SEQ = 6         # word tokens per field passed on to the U-Net (BOS dropped; EOS kept)
N_IDENTITY = 4        # gender, age, framing, view
TOKENS_PER_CHAR = N_IDENTITY + 6 * FIELD_SEQ  # 6 = len(TEXT_FIELDS)
D = 1024              # internal width of the Action Semantic Embedding

# OpenPose-18 limbs used for kinematic features (same layout as server/poses.py).
LIMBS = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0)]
KP_VISIBLE = 0.3


@dataclass
class LAMOutput:
    global_hidden: torch.Tensor   # [B, 77, 2048]  -> U-Net native cross-attention (global descriptor)
    pooled: torch.Tensor          # [B, 1280]      -> SDXL added text embedding
    char_tokens: torch.Tensor     # [B, C, T, 2048] -> action cross-attention stream (T = TOKENS_PER_CHAR)
    char_mask: torch.Tensor       # [B, C]
    bbox: torch.Tensor            # [B, C, 4]
    action_embed: torch.Tensor    # [B, 768] in CLIP image space, for L_action-img
    char_token_mask: torch.Tensor  # [B, C, T] which character tokens the U-Net may attend to


# ------------------------------------------------------------------ backbone

class TextBackbone(nn.Module):
    """SDXL's CLIP-L + OpenCLIP-bigG text transformers, used the way SDXL does (penultimate layers)."""

    def __init__(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2):
        super().__init__()
        self.tok, self.tok2 = tokenizer, tokenizer_2
        self.te, self.te2 = text_encoder, text_encoder_2

    @property
    def device(self) -> torch.device:
        return next(self.te.parameters()).device

    def encode(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> hidden [N, L, 2048], attention mask [N, L], bigG pooled [N, 1280], CLIP-L pooled-projected [N, 768]."""
        ids1 = self.tok(texts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        ids2 = self.tok2(texts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
        o1 = self.te(ids1.input_ids.to(self.device), output_hidden_states=True)
        o2 = self.te2(ids2.input_ids.to(self.device), output_hidden_states=True)
        hidden = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
        # CLIP-L's projection puts its pooled output in CLIP ViT-L/14 image space.
        proj = getattr(self.te, "text_projection", None)
        clip_l = proj(o1.pooler_output) if proj is not None else o1.pooler_output
        return hidden, ids1.attention_mask.to(self.device).bool(), o2.text_embeds, clip_l


# ------------------------------------------------------------------ building blocks

def fourier(x: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """[..., k] in [0, 1] -> [..., k * 2 * n_freq] sinusoidal features."""
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    ang = x.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)


def encoder_layers(d: int, n: int, heads: int = 8) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                       activation="gelu")
    return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)


class AttnPool(nn.Module):
    """Perceiver-style resampler: k learned queries cross-attend to a variable token set."""

    def __init__(self, d: int, k: int, layers: int = 2, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(d), "norm_kv": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, heads, batch_first=True),
                "norm_ff": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None, query_bias: torch.Tensor | None = None) -> torch.Tensor:
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        if query_bias is not None:
            q = q + query_bias
        for blk in self.blocks:
            kv = blk["norm_kv"](x)
            q = q + blk["attn"](blk["norm_q"](q), kv, kv, key_padding_mask=pad_mask, need_weights=False)[0]
            q = q + blk["ff"](blk["norm_ff"](q))
        return q


# ------------------------------------------------------------------ Action Semantic Embedding

class PoseEncoder(nn.Module):
    """Pose Encoding: geometric configuration of the body -> pose-aware tokens."""

    def __init__(self, d: int = D, n_tokens: int = 4):
        super().__init__()
        self.joint_id = nn.Embedding(N_KEYPOINTS + 1, d)  # +1: bbox token
        self.inp = nn.Linear(2 * 2 * 8 + 1, d)
        self.box = nn.Linear(4 * 2 * 8, d)
        self.enc = encoder_layers(d, 2)
        self.pool = AttnPool(d, n_tokens, layers=1)
        self.no_pose = nn.Parameter(torch.zeros(n_tokens, d))

    def forward(self, kps: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """kps [N, 18, 3], bbox [N, 4] -> [N, n_tokens, d]"""
        n = kps.shape[0]
        vis = kps[..., 2] > KP_VISIBLE
        joints = self.inp(torch.cat([fourier(kps[..., :2].clamp(0, 1)), kps[..., 2:3]], -1)) + self.joint_id.weight[:N_KEYPOINTS]
        box = self.box(fourier(bbox.clamp(0, 1))).unsqueeze(1) + self.joint_id.weight[N_KEYPOINTS]
        x = torch.cat([box, joints], 1)
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=kps.device), ~vis], 1)
        tokens = self.pool(self.enc(x, src_key_padding_mask=pad), pad_mask=pad)
        has_pose = vis.any(-1)[:, None, None]
        return torch.where(has_pose, tokens, self.no_pose.to(tokens.dtype).expand(n, -1, -1))


def kinematics(kps: torch.Tensor) -> torch.Tensor:
    """Temporal/kinematic cues readable from one skeleton: limb directions, body extent,
    arm height relative to shoulders, leg spread. [N, 18, 3] -> [N, 3 * len(LIMBS) + 5]."""
    xy, vis = kps[..., :2], kps[..., 2] > KP_VISIBLE
    feats = []
    for a, b in LIMBS:
        v = xy[:, b] - xy[:, a]
        ok = (vis[:, a] & vis[:, b]).float()
        ang = torch.atan2(v[:, 1], v[:, 0])
        feats += [ang.sin() * ok, ang.cos() * ok, ok]
    ys = torch.where(vis, xy[..., 1], torch.full_like(xy[..., 1], float("nan")))
    xs = torch.where(vis, xy[..., 0], torch.full_like(xy[..., 0], float("nan")))
    height = (ys.nan_to_num(-1).amax(-1) - ys.nan_to_num(2).amin(-1)).clamp(0, 1)
    width = (xs.nan_to_num(-1).amax(-1) - xs.nan_to_num(2).amin(-1)).clamp(0, 1)
    shoulder_y = torch.stack([xy[:, 2, 1], xy[:, 5, 1]], -1).mean(-1)
    wrist_y = torch.stack([xy[:, 4, 1], xy[:, 7, 1]], -1).amin(-1)
    arms_raised = ((shoulder_y - wrist_y) * (vis[:, 4] | vis[:, 7]).float()).clamp(-1, 1)
    leg_spread = ((xy[:, 10, 0] - xy[:, 13, 0]).abs() * (vis[:, 10] & vis[:, 13]).float()).clamp(0, 1)
    aspect = width / (height + 1e-3)
    feats += [height, width, arms_raised, leg_spread, aspect.clamp(0, 4) / 4]
    return torch.stack(feats, -1)


class DynamicsEncoder(nn.Module):
    """Action Dynamics Encoding: separates visually similar but dynamically different actions
    (standing vs jumping) from sub-action labels + skeleton kinematics -> one token per sub-action."""

    def __init__(self, d: int = D):
        super().__init__()
        self.action = nn.Embedding(len(ACTIONS), CTX_DIM)  # label-name initialised from the backbone
        self.action_proj = nn.Linear(CTX_DIM, d)
        self.kin = nn.Sequential(nn.Linear(3 * len(LIMBS) + 5, d), nn.GELU(), nn.Linear(d, d))
        self.step = nn.Embedding(MAX_ACTIONS, d)

    def forward(self, actions: torch.Tensor, kps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """actions [N, T], kps [N, 18, 3] -> tokens [N, T, d], valid mask [N, T]"""
        kin = self.kin(kinematics(kps).to(self.step.weight.dtype)).unsqueeze(1)
        tok = self.action_proj(self.action(actions)) + kin + self.step.weight.unsqueeze(0)
        valid = actions > 0
        valid[:, 0] = True  # the first step always exists, carrying kinematics even with no label
        return tok, valid


class StyleAttributeFusion(nn.Module):
    """Fuses identity, pose, dynamics and free-text style attributes into a joint action-style
    embedding: fusion transformer + Perceiver resampler -> TOKENS_PER_CHAR tokens."""

    def __init__(self, d: int = D):
        super().__init__()
        self.text_proj = nn.Linear(CTX_DIM, d)
        self.field_type = nn.Embedding(len(TEXT_FIELDS), d)
        self.kind = nn.Embedding(4, d)  # 0 identity, 1 pose, 2 dynamics, 3 text
        self.char_slot = nn.Embedding(MAX_CHARS, d)
        self.enc = encoder_layers(d, 2)
        self.pool = AttnPool(d, TOKENS_PER_CHAR, layers=2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, CTX_DIM))
        # Zero-initialised: the fused (pose / dynamics / attribute) signal is learned as a residual on
        # top of CLIP-space identity and attribute tokens that the U-Net can read from step 0.
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, identity, pose, dyn, dyn_valid, field_hidden, field_mask, slot):
        """identity [N, 4, d], pose [N, P, d], dyn [N, T, d], field_hidden [N, F, L, 2048],
        field_mask [N, F, L], slot [N] -> [N, TOKENS_PER_CHAR, 2048]"""
        n, f, l, _ = field_hidden.shape
        text = self.text_proj(field_hidden) + self.field_type.weight[None, :, None, :]
        text = text.reshape(n, f * l, -1)
        parts = [identity + self.kind.weight[0], pose + self.kind.weight[1], dyn + self.kind.weight[2],
                 text + self.kind.weight[3]]
        x = torch.cat(parts, 1)
        pad = torch.cat([
            torch.zeros(n, identity.shape[1] + pose.shape[1], dtype=torch.bool, device=x.device),
            ~dyn_valid, ~field_mask.reshape(n, f * l),
        ], 1)
        slot_emb = self.char_slot(slot).unsqueeze(1)
        x = self.enc(x + slot_emb, src_key_padding_mask=pad)
        return self.out(self.pool(x, pad_mask=pad, query_bias=slot_emb))


def field_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over a field's word tokens. Position 0 (CLIP's BOS) is excluded: its hidden state
    has an outsized norm that would otherwise dominate every field's embedding."""
    m = mask.clone()
    m[..., 0] = False
    m = m.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(-2) / m.sum(-2).clamp(min=1)


# ------------------------------------------------------------------ LAM Encoder

class LAMEncoder(nn.Module):
    def __init__(self, backbone: TextBackbone):
        super().__init__()
        from lam.conditioning import TemporalAttention

        self.backbone = backbone
        self.gender = nn.Embedding(len(GENDERS), CTX_DIM)
        self.age = nn.Embedding(len(AGES), CTX_DIM)
        self.framing = nn.Embedding(len(FRAMINGS), CTX_DIM)
        self.view = nn.Embedding(len(VIEWS), CTX_DIM)
        self.label_proj = nn.Linear(CTX_DIM, D)
        self.pose = PoseEncoder()
        self.dynamics = DynamicsEncoder()
        self.temporal = TemporalAttention(D, CTX_DIM)  # Stream 2 of the Dual-Stream pathway (§IV-C)
        self.fusion = StyleAttributeFusion()
        self.empty_field = nn.Parameter(torch.zeros(CTX_DIM))
        # Zero-initialised so L_action-img starts from the global descriptor's CLIP embedding. (No
        # character term is added to SDXL's pooled embedding: the unconditional CFG branch has no
        # characters, so CFG amplified any such shift and washed the colours out.)
        self.embed_from_chars = nn.Linear(CTX_DIM, CLIP_IMG_DIM)
        nn.init.zeros_(self.embed_from_chars.weight)
        nn.init.zeros_(self.embed_from_chars.bias)

    @torch.no_grad()
    def init_label_embeddings(self) -> None:
        """Initialise categorical embeddings from the backbone's encoding of the label names
        (e.g. 'elderly old person'). Only an initialisation; they are trained afterwards."""
        tables = {"gender": self.gender, "age": self.age, "framing": self.framing, "view": self.view,
                  "action": self.dynamics.action}
        for key, emb in tables.items():
            names = LABEL_NAMES[key]
            hidden, mask, _, _ = self.backbone.encode([n or "none" for n in names], FIELD_LEN)
            mean = field_mean(hidden.float(), mask)
            # Averaging shrinks the vector; rescale to the norm of a real word token so identity
            # tokens sit at the same scale as the text tokens the U-Net reads.
            m = mask.clone()
            m[:, 0] = False
            tok_norm = (hidden.float().norm(dim=-1) * m).sum(-1) / m.sum(-1).clamp(min=1)
            emb.weight.copy_((mean * (tok_norm / mean.norm(dim=-1).clamp(min=1e-6))[:, None]).to(emb.weight.dtype))

    def forward(self, d: DescriptorBatch) -> LAMOutput:
        B, C = d.char_mask.shape
        dev = self.backbone.device
        d = d.to(dev)

        # Global descriptor: encoded exactly like SDXL text so the U-Net's native path keeps working.
        global_texts = ["" if drop else t for t, drop in zip(d.global_text, d.drop_all.tolist())]
        g_hidden, _, g_pooled, g_clip = self.backbone.encode(global_texts, 77)

        char_mask = d.char_mask & ~d.drop_all[:, None]
        idx = char_mask.nonzero(as_tuple=False)  # [N, 2] (b, c) of present characters
        char_tokens = g_hidden.new_zeros(B, C, TOKENS_PER_CHAR, CTX_DIM)
        token_mask = torch.zeros(B, C, TOKENS_PER_CHAR, dtype=torch.bool, device=dev)
        if len(idx):
            b, c = idx[:, 0], idx[:, 1]
            # Free-text attribute fields, each encoded separately by the backbone.
            texts = [d.texts[bi][ci][f] for bi, ci in idx.tolist() for f in range(len(TEXT_FIELDS))]
            uniq = sorted({t for t in texts if t})
            n, nf = len(idx), len(TEXT_FIELDS)
            field_hidden = self.empty_field.to(g_hidden.dtype).expand(n * nf, FIELD_LEN, -1).clone()
            field_mask = torch.zeros(n * nf, FIELD_LEN, dtype=torch.bool, device=dev)
            field_mask[:, 0] = True  # empty fields still contribute their learned "empty" token
            if uniq:
                h, m, _, _ = self.backbone.encode(uniq, FIELD_LEN)
                pos = {t: i for i, t in enumerate(uniq)}
                rows = [i for i, t in enumerate(texts) if t]
                src = torch.tensor([pos[texts[i]] for i in rows], device=dev)
                field_hidden[rows] = h[src].to(field_hidden.dtype)
                field_mask[rows] = m[src]
            field_hidden = field_hidden.view(n, nf, FIELD_LEN, -1)
            field_mask = field_mask.view(n, nf, FIELD_LEN)

            identity_ctx = torch.stack([
                self.gender(d.gender[b, c]), self.age(d.age[b, c]), self.framing(d.framing[b, c]), self.view(d.view[b, c]),
            ], 1)
            kps, box = d.keypoints[b, c], d.bbox[b, c]
            pose = self.pose(kps, box)
            dyn, dyn_valid = self.dynamics(d.actions[b, c], kps)
            fused = self.fusion(self.label_proj(identity_ctx), pose, dyn, dyn_valid, field_hidden, field_mask, c)
            # Base tokens live in the backbone's text space, so the U-Net reads them from step 0:
            # 4 identity tokens + each field's word-token sequence (BOS dropped, EOS kept).
            seq = field_hidden[:, :, 1:1 + FIELD_SEQ].reshape(n, nf * FIELD_SEQ, -1)
            seq_mask = field_mask[:, :, 1:1 + FIELD_SEQ].reshape(n, nf * FIELD_SEQ)
            # Unspecified age / framing carry no information: hide them rather than show "none".
            id_mask = torch.stack([torch.ones_like(c, dtype=torch.bool), d.age[b, c] > 0,
                                   d.framing[b, c] > 0, torch.ones_like(c, dtype=torch.bool)], 1)
            base = torch.cat([identity_ctx.to(fused.dtype), seq.to(fused.dtype)], 1)
            tokens = base + fused
            tokens = tokens + self.temporal(dyn, dyn_valid).unsqueeze(1)  # Stream 2 fused before injection
            char_tokens[b, c] = tokens.to(char_tokens.dtype)
            token_mask[b, c] = torch.cat([id_mask, seq_mask], 1)

        tm = token_mask.to(char_tokens.dtype)[..., None]
        per_char = (char_tokens * tm).sum(2) / tm.sum(2).clamp(min=1)  # [B, C, D]
        m = char_mask.to(per_char.dtype)[..., None]
        char_mean = (per_char * m).sum(1) / m.sum(1).clamp(min=1)
        return LAMOutput(
            global_hidden=g_hidden,
            pooled=g_pooled,
            char_tokens=char_tokens,
            char_mask=char_mask,
            bbox=d.bbox,
            action_embed=g_clip + self.embed_from_chars(char_mean).to(g_clip.dtype),
            char_token_mask=token_mask,
        )

    def trainable_parameters(self):
        """Everything except the frozen base weights of the backbone (its LoRA params stay trainable)."""
        return [p for n, p in self.named_parameters() if p.requires_grad]
