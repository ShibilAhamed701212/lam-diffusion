"""§IV-F stage 1: the anime-optimized VAE (§IV-D).

The paper pre-trains a VAE on high-resolution (>= 512x512) anime character images and freezes
it for the rest of training. Re-training the *encoder* would move the latent space and break the
pre-trained Animagine U-Net, so the encoder (and quant_conv) stay frozen and only the decoder is
fine-tuned on our anime images. The latents (and every precomputed sample) are unchanged; what
improves is how faithfully latents are turned back into linework, cel shading and flat colour.

Recipe follows Stability's sd-vae-ft-mse decoder fine-tune: MSE + 0.1 * LPIPS, EMA weights.
Gate: the fine-tuned decoder must beat the stock one on held-out anime images (LPIPS and PSNR).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lam.dataset import load_index

VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CROP = 512


class AnimeCrops(Dataset):
    """Random 512 crops of images resized to SDXL scale (short side 1024), i.e. the detail level
    the decoder has to reproduce at generation time."""

    def __init__(self, ids: list[int], images: Path, train: bool):
        self.ids, self.dir, self.train = ids, images, train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.dir / f"{self.ids[i]}.jpg").convert("RGB")
        s = 1024 / min(img.size)
        img = img.resize((max(CROP, round(img.width * s)), max(CROP, round(img.height * s))), Image.BICUBIC)
        if self.train:
            x0, y0 = random.randint(0, img.width - CROP), random.randint(0, img.height - CROP)
        else:
            x0, y0 = (img.width - CROP) // 2, (img.height - CROP) // 2
        crop = img.crop((x0, y0, x0 + CROP, y0 + CROP))
        if self.train and random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float() / 127.5 - 1.0


@torch.no_grad()
def evaluate(vae, decoder_state: dict | None, loader, lpips_fn) -> dict[str, float]:
    """PSNR / LPIPS of encode -> decode on held-out crops. decoder_state=None: current weights."""
    backup = None
    if decoder_state is not None:
        backup = copy.deepcopy(vae.decoder.state_dict()), copy.deepcopy(vae.post_quant_conv.state_dict())
        vae.decoder.load_state_dict(decoder_state["decoder"])
        vae.post_quant_conv.load_state_dict(decoder_state["post_quant_conv"])
    psnr, lp, n = 0.0, 0.0, 0
    for px in loader:
        px = px.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = vae.encode(px).latent_dist.mode()
            rec = vae.decode(z).sample.float().clamp(-1, 1)
        mse = ((rec - px) ** 2).mean(dim=(1, 2, 3))
        psnr += (10 * torch.log10(4.0 / mse)).sum().item()  # range [-1, 1] -> peak-to-peak 2
        lp += lpips_fn(rec, px).sum().item()
        n += len(px)
    if backup is not None:
        vae.decoder.load_state_dict(backup[0])
        vae.post_quant_conv.load_state_dict(backup[1])
    return {"psnr": psnr / n, "lpips": lp / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--images", default="/content/data/images")
    ap.add_argument("--out", default="/content/ckpt/vae_anime")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)  # 8 runs out of memory on a 24 GB L4
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lpips-weight", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-images", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()

    import lpips
    from diffusers import AutoencoderKL

    torch.backends.cuda.matmul.allow_tf32 = True
    images = Path(a.images)
    have = lambda d: (images / f"{d['id']}.jpg").exists()  # noqa: E731
    train_ids = [d["id"] for d in load_index(a.index, "train") if have(d)]
    test_ids = [d["id"] for d in load_index(a.index, "test") if have(d)][: a.eval_images]
    print(f"train images {len(train_ids)} eval images {len(test_ids)}", flush=True)

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float32).cuda()
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    vae.post_quant_conv.requires_grad_(True)
    vae.enable_gradient_checkpointing()
    params = [p for p in vae.parameters() if p.requires_grad]
    ema = {n: p.detach().clone() for n, p in vae.named_parameters() if p.requires_grad}
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).cuda().eval().requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))

    loader = DataLoader(AnimeCrops(train_ids, images, True), batch_size=a.batch, shuffle=True, num_workers=8,
                        drop_last=True, persistent_workers=True, pin_memory=True)
    eval_loader = DataLoader(AnimeCrops(test_ids, images, False), batch_size=16, num_workers=4)

    base = evaluate(vae, None, eval_loader, lpips_fn)
    print("stock SDXL VAE:", base, flush=True)
    log, step, t0 = [], 0, time.time()

    def ema_state() -> dict:
        dec = {k.removeprefix("decoder."): v for k, v in ema.items() if k.startswith("decoder.")}
        pqc = {k.removeprefix("post_quant_conv."): v for k, v in ema.items() if k.startswith("post_quant_conv.")}
        return {"decoder": dec, "post_quant_conv": pqc}

    vae.train()
    while step < a.steps:
        for px in loader:
            px = px.cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = vae.encode(px).latent_dist.sample()  # frozen encoder: the latent space never moves
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = vae.decode(z.float()).sample.float()
            l_mse = F.mse_loss(rec, px)
            l_lp = lpips_fn(rec.clamp(-1, 1), px).mean()
            loss = l_mse + a.lpips_weight * l_lp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for n, p in vae.named_parameters():
                    if n in ema:
                        ema[n].lerp_(p.detach(), 1 - a.ema)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{a.steps} mse {l_mse.item():.4f} lpips {l_lp.item():.4f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                r = evaluate(vae, ema_state(), eval_loader, lpips_fn)
                log.append({"step": step, **r})
                print("eval (EMA):", r, flush=True)
            if step >= a.steps:
                break

    st = ema_state()
    vae.decoder.load_state_dict(st["decoder"])
    vae.post_quant_conv.load_state_dict(st["post_quant_conv"])
    vae.save_pretrained(a.out)
    final = log[-1] if log else {}
    passed = bool(final) and final["lpips"] < base["lpips"] and final["psnr"] > base["psnr"]
    json.dump({"stock": base, "log": log, "gate_passed": passed}, open(Path(a.out) / "stage1.json", "w"), indent=1)
    print("saved", a.out, "| gate", "PASSED" if passed else "FAILED (keep the stock decoder)", flush=True)


if __name__ == "__main__":
    main()
