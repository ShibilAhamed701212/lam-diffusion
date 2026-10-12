"""Inference: action descriptors -> image (paper §IV-F inference pipeline).

(1) descriptors from the planner, (2) LAM Encoder -> action embeddings, (3) Gaussian latent,
(4) T denoising steps conditioned through the dual-stream pathway with classifier-free guidance
against the dropped-conditioning embedding, (5) VAE decode (the anime-optimized decoder of
§IV-F stage 1 when vae_path is given).

algorithm1(): the paper's Algorithm 1 (Eq. 4-6): image -> h = F_φ(x) -> a = G_θ(h) -> diffusion
refined with the Eq. 5 update -> x̂ = F_φ^{-1}(z0).
"""

from __future__ import annotations

import torch
from PIL import Image

from lam.descriptor import collate
from lam.encoder import LAMOutput
from lam.model import COMPUTE_DTYPE
from lam.model import BASE_MODEL, LAMDiffusion

SIZES = {"portrait": (832, 1216), "square": (1024, 1024), "landscape": (1216, 832)}


def cat_lam(a: LAMOutput, b: LAMOutput) -> LAMOutput:
    return LAMOutput(*(torch.cat([x, y]) for x, y in zip(a.__dict__.values(), b.__dict__.values())))


class LAMDiffusionPipeline:
    def __init__(self, model: LAMDiffusion, device: str = "cuda", vae_path: str | None = None):
        from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler

        self.model = model
        # The fine-tuned anime decoder is no longer guaranteed fp16-safe (the fp16-fix property was
        # tuned for the stock weights), so it runs in bf16.
        self.vae_dtype = COMPUTE_DTYPE if vae_path else torch.float16
        self.vae = AutoencoderKL.from_pretrained(vae_path or "madebyollin/sdxl-vae-fp16-fix",
                                                 torch_dtype=self.vae_dtype).to(device)
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
        self.device = device

    @torch.no_grad()
    def __call__(self, descs: list[dict], *, width: int, height: int, steps: int = 28, cfg: float = 5.0,
                 seed: int = 0, action_scale: float = 1.0, region_masking: bool = True,
                 negative: str = "", capture_attention: bool = False) -> list[Image.Image]:
        m = self.model
        m.eval()
        B = len(descs)
        state = m.conditioning.state
        state.scale, state.region_masking = action_scale, region_masking
        for p in m.conditioning.processors.values():
            p.capture = capture_attention
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            cond = m.encode(collate(descs))
            # Unconditional branch: no characters; the global stream carries the negative prompt
            # (empty string = the dropped-conditioning embedding learned in CFG training).
            uncond_batch = collate(descs)
            uncond_batch.char_mask[:] = False
            uncond_batch.global_text = [negative] * B
            lam = cat_lam(m.encode(uncond_batch), cond)
        gen = torch.Generator(self.device).manual_seed(seed)
        lat = torch.randn(B, 4, height // 8, width // 8, generator=gen, device=self.device)
        self.scheduler.set_timesteps(steps, device=self.device)
        lat = lat * self.scheduler.init_noise_sigma
        time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self.device, dtype=torch.float32).repeat(2 * B, 1)
        for t in self.scheduler.timesteps:
            inp = self.scheduler.scale_model_input(torch.cat([lat, lat]), t)
            with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                eps = m.denoise(inp, t.expand(2 * B), lam, time_ids).float()
            eu, ec = eps.chunk(2)
            lat = self.scheduler.step(eu + cfg * (ec - eu), t, lat, generator=gen).prev_sample
        state.scale, state.region_masking = 1.0, True
        return self._decode(lat)

    def _decode(self, lat: torch.Tensor) -> list[Image.Image]:
        img = self.vae.decode((lat / self.vae.config.scaling_factor).to(self.vae_dtype)).sample
        img = ((img.float() + 1) * 127.5).clamp(0, 255).permute(0, 2, 3, 1).byte().cpu().numpy()
        return [Image.fromarray(x) for x in img]

    @torch.no_grad()
    def encode_image(self, img: Image.Image, width: int, height: int) -> torch.Tensor:
        """Eq. 4: h = F_φ(x), the image's latent under the frozen VAE encoder (U-Net scale)."""
        import numpy as np

        from lam.data.precompute import cover_crop

        crop = cover_crop(img.convert("RGB"), width, height)[0]
        px = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1)[None].to(self.device, self.vae_dtype) / 127.5 - 1
        return self.vae.encode(px).latent_dist.mode().float() * self.vae.config.scaling_factor

    @torch.no_grad()
    def algorithm1(self, images: list[Image.Image], inverse, *, width: int, height: int, eta: float = 0.0,
                   steps: int = 28, cfg: float = 5.0, seed: int = 0) -> tuple[list[Image.Image], list[dict]]:
        """Paper Algorithm 1. For each image x_i:
            h_i = F_φ(x_i)                      (Eq. 4, VAE encoder)
            a_i = G_θ(h_i)                      (LAM image head, lam/inverse.py)
            z ~ N(0, I); for t = T..1: z_{t-1} = z_t - η ∇_{z_t} L_diff(z_t, h_i, a_i)   (Eq. 5)
            x̂_i = F_φ^{-1}(z_0)                 (Eq. 6, VAE decoder)

        Eq. 5 is realised inside the reverse diffusion step. With L_diff(z_t, h, a) =
        ||ε̂(z_t, a) - (z_t - h)/σ_t||² and the stop-gradient on ε̂ used by score-distillation
        methods, ∇_{z_t} L_diff = (2/σ_t²)(ẑ_0 - h) where ẑ_0 = z_t - σ_t ε̂. Taking the step size
        η σ_t²/2 moves the denoiser's clean estimate toward h by a fraction η, i.e.
        ε̂ <- ε̂ + η (ẑ_0 - h)/σ_t before the scheduler step. η = 0 generates purely from the
        predicted actions a_i (a new image with the same characters / pose / attributes);
        η -> 1 reconstructs x_i. Returns the images and the predicted descriptors a_i.
        """
        m = self.model
        m.eval()
        h = torch.cat([self.encode_image(im, width, height) for im in images])
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            descs = inverse.to_descriptors({k: v.float() for k, v in inverse(h).items()})
            cond = m.encode(collate(descs))
            unc = collate(descs)
            unc.char_mask[:] = False
            unc.global_text = [""] * len(descs)
            lam = cat_lam(m.encode(unc), cond)
        B = len(images)
        gen = torch.Generator(self.device).manual_seed(seed)
        lat = torch.randn(B, 4, height // 8, width // 8, generator=gen, device=self.device)
        self.scheduler.set_timesteps(steps, device=self.device)
        lat = lat * self.scheduler.init_noise_sigma
        time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self.device, dtype=torch.float32).repeat(2 * B, 1)
        for i, t in enumerate(self.scheduler.timesteps):
            inp = self.scheduler.scale_model_input(torch.cat([lat, lat]), t)
            with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                eps = m.denoise(inp, t.expand(2 * B), lam, time_ids).float()
            eu, ec = eps.chunk(2)
            eps = eu + cfg * (ec - eu)
            if eta:
                sigma = self.scheduler.sigmas[i].to(lat.device)
                z0 = lat - sigma * eps
                eps = eps + eta * (z0 - h) / sigma  # Eq. 5 step on the clean estimate
            lat = self.scheduler.step(eps, t, lat, generator=gen).prev_sample
        return self._decode(lat), descs

    @torch.no_grad()
    def prompt_only(self, prompt: str, negative: str, *, width: int, height: int, steps: int = 28,
                    cfg: float = 5.0, seed: int = 0) -> Image.Image:
        """Prompt-only baseline on the same weights: LoRA adapters and the action stream off,
        so this is exactly the base Animagine model reading the raw text."""
        m = self.model
        if m.unet_mode == "full":
            raise RuntimeError("full-mode U-Net has no adapters to switch off; build the prompt-only baseline "
                               "from the stock Animagine weights (LAMDiffusion(unet_mode='lora') with no checkpoint)")
        enc = m.encoder.backbone
        m.unet.disable_adapters()
        enc.te.disable_adapters()
        enc.te2.disable_adapters()
        m.conditioning.state.tokens = None
        try:
            with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                h, _, pooled, _ = enc.encode([negative, prompt], 77)
            gen = torch.Generator(self.device).manual_seed(seed)
            lat = torch.randn(1, 4, height // 8, width // 8, generator=gen, device=self.device)
            self.scheduler.set_timesteps(steps, device=self.device)
            lat = lat * self.scheduler.init_noise_sigma
            time_ids = torch.tensor([[height, width, 0, 0, height, width]] * 2, device=self.device, dtype=torch.float32)
            for t in self.scheduler.timesteps:
                inp = self.scheduler.scale_model_input(torch.cat([lat, lat]), t)
                with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                    eps = m.unet(inp, t.expand(2), encoder_hidden_states=h,
                                 added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample.float()
                eu, ec = eps.chunk(2)
                lat = self.scheduler.step(eu + cfg * (ec - eu), t, lat, generator=gen).prev_sample
            return self._decode(lat)[0]
        finally:
            m.unet.enable_adapters()
            enc.te.enable_adapters()
            enc.te2.enable_adapters()
