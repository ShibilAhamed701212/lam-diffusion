"""Composite multi-objective loss (paper §IV-E, Eq. 3):

    L_total = λ1 L_diff + λ2 L_action-img + λ3 L_perc + λ4 L_style

The three auxiliary terms need an image, so they use the one-step estimate x̂0 from the
predicted noise, decoded by TAESDXL (a tiny differentiable decoder for SDXL latents), and are
only applied at low noise (t < AUX_MAX_T) where x̂0 is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

AUX_MAX_T = 400
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
VGG_LAYERS = (3, 8, 15, 22)  # relu1_2, relu2_2, relu3_3, relu4_3


@dataclass
class LossWeights:
    diff: float = 1.0
    # Lowered 10x after the 2026-10-09 L4 pilots: at (0.05, 0.1, 2000) every sample grid turned into
    # swirly, muddy texture by step 200; diffusion-only and these weights both stayed clean.
    action_img: float = 0.005
    perc: float = 0.02
    style: float = 200.0  # Gram MSE is ~1e-5
    # Attribute-swap consistency (NOT in the paper; see swap_consistency). OFF: on 2026-10-10 it sat at
    # ~1.6e-5 from step 0 (the encoder tokens do not leak) and did not reduce leakage in the CLIP test.
    swap: float = 0.0


def diffusion_loss(eps_pred: torch.Tensor, eps: torch.Tensor, snr: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """L_diff: ε-prediction MSE. gamma > 0 adds min-SNR-γ weighting (per-sample weights, then mean);
    gamma = 0 is the plain "standard denoising loss" the paper specifies (§IV-E)."""
    mse = F.mse_loss(eps_pred.float(), eps.float(), reduction="none").mean(dim=(1, 2, 3))
    if not gamma:
        return mse.mean()
    w = torch.clamp(snr, max=gamma) / snr
    return (w * mse).mean()


def predict_x0(z_t: torch.Tensor, eps_pred: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a = alpha_bar.view(-1, 1, 1, 1).to(z_t.dtype)
    return (z_t - (1 - a).sqrt() * eps_pred) / a.sqrt()


def gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    f = x.reshape(b, c, h * w).float()
    return f @ f.transpose(1, 2) / (c * h * w)


class AuxLosses(nn.Module):
    """L_action-img, L_perc (LPIPS) and L_style (Gram) on decoded x̂0. All networks frozen."""

    def __init__(self, device: str = "cuda", queue_size: int = 512):
        super().__init__()
        import lpips
        from diffusers import AutoencoderTiny
        from torchvision.models import VGG16_Weights, vgg16
        from transformers import CLIPVisionModelWithProjection

        dt = torch.bfloat16
        self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=dt).to(device).eval()
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dt).to(device).eval()
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(VGG_LAYERS) + 1].to(device, dt).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, 768, device=device), dim=-1))
        self.queue_ptr = 0
        self.tau = 0.07

    def decode(self, x0_latents: torch.Tensor) -> torch.Tensor:
        """U-Net-space latents -> RGB in [0, 1]."""
        img = self.taesd.decode(x0_latents.to(torch.bfloat16)).sample
        return ((img.float() + 1) / 2).clamp(0, 1)

    def _norm(self, x: torch.Tensor, mean, std) -> torch.Tensor:
        return (x - x.new_tensor(mean).view(1, 3, 1, 1)) / x.new_tensor(std).view(1, 3, 1, 1)

    def clip_embed(self, img01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img01, size=(224, 224), mode="bicubic", align_corners=False)
        e = self.clip(pixel_values=self._norm(x, CLIP_MEAN, CLIP_STD).to(torch.bfloat16)).image_embeds
        return F.normalize(e.float(), dim=-1)

    def vgg_feats(self, img01: torch.Tensor) -> list[torch.Tensor]:
        x = self._norm(img01, IMNET_MEAN, IMNET_STD).to(torch.bfloat16)
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in VGG_LAYERS:
                feats.append(x)
        return feats

    def forward(self, x0_latents: torch.Tensor, gt01: torch.Tensor, action_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        """x0_latents [B, 4, h, w]; gt01 [B, 3, H, W] in [0, 1]; action_embed [B, 768]."""
        img = self.decode(x0_latents)
        size = (256, round(256 * img.shape[-1] / img.shape[-2]))
        img_s = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        gt_s = F.interpolate(gt01.float(), size=size, mode="bilinear", align_corners=False)

        # L_action-img: generated image vs LAM action embedding, InfoNCE against a queue of past embeddings.
        img_e = self.clip_embed(img)
        act = F.normalize(action_embed.float(), dim=-1)
        pos = (img_e * act).sum(-1, keepdim=True)
        neg = img_e @ self.queue.t()
        logits = torch.cat([pos, neg], 1) / self.tau
        l_act = F.cross_entropy(logits, torch.zeros(len(img_e), dtype=torch.long, device=img_e.device))
        with torch.no_grad():
            n = len(act)
            idx = (torch.arange(n, device=act.device) + self.queue_ptr) % len(self.queue)
            self.queue[idx] = act.detach()
            self.queue_ptr = int((self.queue_ptr + n) % len(self.queue))

        # L_perc: LPIPS distance to the ground-truth image.
        l_perc = self.lpips(img_s * 2 - 1, gt_s * 2 - 1).mean()

        # L_style: Gram-matrix statistics of a pre-trained network, generated vs ground truth.
        l_style = sum(F.mse_loss(gram(a), gram(b)) for a, b in zip(self.vgg_feats(img_s), self.vgg_feats(gt_s)))
        return {"action_img": l_act, "perc": l_perc, "style": l_style,
                "clip_sim": pos.detach().mean()}


def swap_consistency(ref, cf, field: int) -> torch.Tensor:
    """Attribute-swap consistency (our addition, not in the paper). `cf` is the encoder output for a
    descriptor in which only text field `field` (eye or hair colour) was changed. Every character
    token that does not belong to that field must stay put, otherwise the joint fusion leaks the
    changed attribute into the others (seen as eye colour changing hair colour in the Phase A and
    pilot renders). Normalised by the mean square of the reference tokens."""
    from lam.encoder import FIELD_SEQ, N_IDENTITY, TOKENS_PER_CHAR

    keep = torch.ones(TOKENS_PER_CHAR, dtype=torch.bool, device=ref.char_tokens.device)
    keep[N_IDENTITY + field * FIELD_SEQ: N_IDENTITY + (field + 1) * FIELD_SEQ] = False
    m = ref.char_token_mask & cf.char_token_mask & keep[None, None, :]
    if not m.any():
        return ref.char_tokens.sum() * 0.0
    r = ref.char_tokens.detach().float()[m]
    c = cf.char_tokens.float()[m]
    return ((c - r) ** 2).mean() / (r ** 2).mean().clamp(min=1e-6)
