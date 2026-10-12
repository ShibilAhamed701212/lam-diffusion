"""LAM-Diffusion model assembly: LAM Encoder + Dual-Stream Conditioning + latent-diffusion backbone.

Backbone (§IV-D): Animagine XL 4.0 U-Net, trained either fully (unet_mode="full", §IV-F stage 3
as the paper specifies; needs an A100-class GPU) or with LoRA (unet_mode="lora", the L4 fallback).
The VAE is frozen (it only appears at inference since training uses pre-computed latents). Image generation is conditioned on the
action descriptors A only (Eq. 2, I = G_diff(A)): the U-Net's native cross-attention reads the
global descriptor, the action stream reads the per-character tokens.
"""

from __future__ import annotations

import os

from pathlib import Path

import torch
from torch import nn

from lam.conditioning import DualStreamConditioning
from lam.descriptor import DescriptorBatch
from lam.encoder import LAMEncoder, LAMOutput, TextBackbone

BASE_MODEL = "cagliostrolab/animagine-xl-4.0"
CLIP_L = "openai/clip-vit-large-patch14"


# T4-class GPUs have no bf16: LAM_DTYPE=fp16 runs inference in fp16 (training always uses bf16).
COMPUTE_DTYPE = torch.float16 if os.environ.get("LAM_DTYPE") == "fp16" else torch.bfloat16

class LAMDiffusion(nn.Module):
    def __init__(self, base: str = BASE_MODEL, device: str = "cuda", unet_rank: int = 32, te_rank: int = 16,
                 gradient_checkpointing: bool = True, unet_mode: str = "lora",
                 unet_path: str | None = None):
        super().__init__()
        assert unet_mode in ("lora", "full"), unet_mode
        self.unet_mode = unet_mode
        from diffusers import UNet2DConditionModel
        from peft import LoraConfig
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        dt = COMPUTE_DTYPE
        tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
        te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dt)
        te2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", torch_dtype=dt)
        # CLIP-L's text projection (not shipped with SDXL) maps pooled text into CLIP image space,
        # which is where L_action-img compares the action embedding with the generated image.
        te.text_projection = CLIPTextModelWithProjection.from_pretrained(CLIP_L, torch_dtype=dt).text_projection
        # unet_path: a fully-trained U-Net saved by save_unet (inference / evaluation of the full run).
        self.unet = (UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dt) if unet_path
                     else UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dt))

        for m in (te, te2, self.unet):
            m.requires_grad_(False)
        # Dual-stream processors go in before LoRA so they reference the base K/V weights.
        self.conditioning = DualStreamConditioning(self.unet)
        if unet_mode == "lora":
            self.unet.add_adapter(LoraConfig(r=unet_rank, lora_alpha=unet_rank, init_lora_weights="gaussian",
                                             target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        else:
            self.unet.requires_grad_(True)
        # peft freezes every non-LoRA parameter in the U-Net, including the dual-stream
        # processors that live inside it; they are new modules and must train.
        self.conditioning.requires_grad_(True)
        te_lora = LoraConfig(r=te_rank, lora_alpha=te_rank, init_lora_weights="gaussian",
                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        te.add_adapter(te_lora)
        te2.add_adapter(te_lora)
        self.encoder = LAMEncoder(TextBackbone(tok, tok2, te, te2))
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            # Non-reentrant checkpointing keeps gradients flowing to LoRA weights behind frozen embeddings.
            te.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            te2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.to(device)
        # Trainable weights in fp32 (autocast runs the maths in bf16); frozen weights stay bf16.
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.encoder.init_label_embeddings()

    # ------------------------------------------------------------------ forward

    def encode(self, desc: DescriptorBatch) -> LAMOutput:
        return self.encoder(desc)

    def denoise(self, noisy: torch.Tensor, t: torch.Tensor, lam: LAMOutput, time_ids: torch.Tensor) -> torch.Tensor:
        self.conditioning.bind(lam, latent_hw=tuple(noisy.shape[-2:]))
        return self.unet(
            noisy, t, encoder_hidden_states=lam.global_hidden,
            added_cond_kwargs={"text_embeds": lam.pooled, "time_ids": time_ids},
        ).sample

    # ------------------------------------------------------------------ parameters / checkpoints

    def param_groups(self, lr_new: float, lr_unet_lora: float, lr_te_lora: float, lr_gate: float,
                     lr_unet_full: float = 1e-5) -> list[dict]:
        groups = {"new": [], "unet_lora": [], "unet_full": [], "te_lora": [], "gate": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".gate"):
                groups["gate"].append(p)
            elif "lora_" in name and name.startswith("unet."):
                groups["unet_lora"].append(p)
            elif "lora_" in name:
                groups["te_lora"].append(p)
            elif self.is_base_unet_param(name):
                groups["unet_full"].append(p)
            else:
                groups["new"].append(p)
        lrs = {"new": lr_new, "unet_lora": lr_unet_lora, "unet_full": lr_unet_full, "te_lora": lr_te_lora,
               "gate": lr_gate}
        # No weight decay on the pre-trained U-Net weights: decay would pull them away from Animagine.
        return [{"params": ps, "lr": lrs[k], "name": k, **({"weight_decay": 0.0} if k == "unet_full" else {})}
                for k, ps in groups.items() if ps]

    @staticmethod
    def is_base_unet_param(name: str) -> bool:
        """Original U-Net weights (not the dual-stream processors that are registered inside it)."""
        return name.startswith("unet.") and ".processor." not in name and "lora_" not in name

    def trainable_state(self, include_unet: bool = True) -> dict[str, torch.Tensor]:
        """include_unet=False leaves out the full U-Net weights (5 GB in bf16), so the small LAM
        part can be saved and synced often; the U-Net is saved separately (save_unet)."""
        names = {n for n, p in self.named_parameters() if p.requires_grad
                 and (include_unet or not self.is_base_unet_param(n))}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def save_unet(self, path: str | Path) -> None:
        """Fully-trained U-Net as a diffusers folder (bf16), loadable with UNet2DConditionModel.from_pretrained."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.unet.save_config(path)
        sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in self.unet.state_dict().items()
              if ".processor." not in k}
        save_file(sd, str(path / "diffusion_pytorch_model.safetensors"))

    def load_trainable(self, path: str | Path, strict: bool = True) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        missing = {n for n, p in self.named_parameters() if p.requires_grad} - set(sd)
        if strict and missing:
            raise KeyError(f"checkpoint missing {len(missing)} trainable tensors, e.g. {sorted(missing)[:3]}")
        self.load_state_dict(sd, strict=False)

    def count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train}
