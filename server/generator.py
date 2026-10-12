"""Diffusion stage: Animagine XL (anime SDXL) + OpenPose ControlNet.

The pose/layout image from the planner conditions the U-Net through ControlNet, and the
spec's tags condition it through cross-attention. The baseline pipeline shares the same
weights but gets only the raw prompt and no pose image, so comparisons are like-for-like.
"""

from __future__ import annotations

import os

import torch
from PIL import Image

BASE_MODEL = os.environ.get("LAM_BASE_MODEL", "cagliostrolab/animagine-xl-4.0")
CONTROLNET = os.environ.get("LAM_CONTROLNET", "xinsir/controlnet-openpose-sdxl-1.0")
VAE = "madebyollin/sdxl-vae-fp16-fix"


class Generator:
    def __init__(self, device: str = "cuda"):
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerAncestralDiscreteScheduler,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLPipeline,
        )

        dtype = torch.float16
        vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=dtype)
        controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=dtype)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL, controlnet=controlnet, vae=vae, torch_dtype=dtype, add_watermarker=False,
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        # from_pipe re-casts shared components unless given a dtype, which silently
        # doubles VRAM by upcasting the U-Net and text encoders to float32.
        self.baseline = StableDiffusionXLPipeline.from_pipe(self.pipe, torch_dtype=dtype)
        self.baseline.set_progress_bar_config(disable=True)
        self.device = device
        upcast = [n for n, m in self.pipe.components.items()
                  if isinstance(m, torch.nn.Module) and next(m.parameters()).dtype != dtype]
        if upcast:
            raise RuntimeError(f"components not in {dtype}: {upcast}")

    @torch.inference_mode()
    def generate(self, prompt: str, negative: str, pose_image: Image.Image | None, *, width: int,
                 height: int, seed: int, steps: int, cfg: float, control_scale: float) -> Image.Image:
        gen = torch.Generator(self.device).manual_seed(seed)
        common = dict(prompt=prompt, negative_prompt=negative, width=width, height=height,
                      num_inference_steps=steps, guidance_scale=cfg, generator=gen)
        if pose_image is None:
            return self.baseline(**common).images[0]
        return self.pipe(**common, image=pose_image, controlnet_conditioning_scale=control_scale).images[0]
