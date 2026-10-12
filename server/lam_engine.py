"""Serving the trained LAM-Diffusion model (paper §IV) in the app.

plan:     fine-tuned planner (Qwen2.5-3B + LoRA): request -> descriptor. No keyword rules.
generate: descriptor -> LAM Encoder -> dual-stream conditioning -> Animagine U-Net (+LoRA).
baseline: same weights with adapters and action stream off = prompt-only Animagine.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lam.model import LAMDiffusion  # noqa: E402
from lam.pipeline import LAMDiffusionPipeline  # noqa: E402
from lam.planner_ft import parse, plan as planner_generate  # noqa: E402
from lam.spec_bridge import spec_to_descriptor  # noqa: E402

WEIGHTS = os.environ.get("LAM_WEIGHTS", "/content/ckpt/weights.pt")
PLANNER = os.environ.get("LAM_PLANNER_LORA", "/content/ckpt/planner_lora")
POSES = os.environ.get("LAM_POSE_LIBRARY", "/content/data/pose_library.json")
QUALITY = "masterpiece, high score, great score, absurdres"
NEGATIVE = ("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, "
            "worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry")


class LAMEngine:
    def __init__(self):
        # LAM_UNET: folder of the fully-trained U-Net (full run); unset = base U-Net + LoRA weights.
        unet = os.environ.get("LAM_UNET", "")
        self.model = LAMDiffusion(gradient_checkpointing=False, unet_mode="full" if unet else "lora",
                                  unet_path=unet or None)
        self.model.load_trainable(WEIGHTS, strict=False)
        if os.environ.get("LAM_DTYPE") == "fp16":
            # Inference only: trainable weights are fp32 by default (12.7 GB); fp16 fits a 15 GB T4.
            self.model.half().eval().requires_grad_(False)
        self.pipe = LAMDiffusionPipeline(self.model)
        self.pose_lib = json.loads(Path(POSES).read_text())
        self.planner = self.tok = None
        if Path(PLANNER).exists():  # the fine-tuned planner is optional; the app falls back to rules
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base = "Qwen/Qwen2.5-3B-Instruct"
            self.tok = AutoTokenizer.from_pretrained(base)
            llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
            self.planner = PeftModel.from_pretrained(llm, PLANNER).eval()

    def plan(self, prompt: str) -> dict | None:
        if self.planner is None:
            return None
        return planner_generate(self.planner, self.tok, [prompt])[0]

    def generate(self, spec: dict, *, width: int, height: int, seed: int, steps: int, cfg: float,
                 attention: bool = False):
        desc = spec_to_descriptor(spec, self.pose_lib)
        desc["global"]["style"] = ", ".join(x for x in (desc["global"]["style"], QUALITY) if x)
        img = self.pipe([desc], width=width, height=height, steps=steps, cfg=cfg, seed=seed,
                        negative=NEGATIVE, capture_attention=attention)[0]
        return img, desc

    def baseline(self, prompt: str, *, width: int, height: int, seed: int, steps: int, cfg: float):
        if self.model.unet_mode != "full":  # LoRA mode: same weights with adapters + action stream off
            return self.pipe.prompt_only(f"{prompt}, {QUALITY}", NEGATIVE, width=width, height=height,
                                         seed=seed, steps=steps, cfg=cfg)
        # Full mode trained every U-Net weight, so the baseline is a separate stock Animagine pipeline
        # (LAM_BASELINE_DEVICE puts it on a second GPU, e.g. Kaggle's 2x T4).
        if getattr(self, "_stock", None) is None:
            from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

            dev = os.environ.get("LAM_BASELINE_DEVICE", "cuda")
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self._stock = StableDiffusionXLPipeline.from_pretrained(
                "cagliostrolab/animagine-xl-4.0", vae=vae, torch_dtype=torch.float16, add_watermarker=False).to(dev)
            self._stock.scheduler = EulerAncestralDiscreteScheduler.from_config(self._stock.scheduler.config)
            self._stock.set_progress_bar_config(disable=True)
        gen = torch.Generator(self._stock.device).manual_seed(seed)
        return self._stock(f"{prompt}, {QUALITY}", negative_prompt=NEGATIVE, width=width, height=height,
                           num_inference_steps=steps, guidance_scale=cfg, generator=gen).images[0]


__all__ = ["LAMEngine", "parse"]
