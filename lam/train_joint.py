"""§IV-F stages 3-5: joint training of the U-Net (frozen VAE) and the LAM Encoder with the
composite loss (Eq. 3), a progressive curriculum (stage 4) and conditioning dropout for
classifier-free guidance (stage 5). Resumable; writes sample grids and checkpoints.

--unet-mode full trains every U-Net weight, as the paper specifies (A100-class GPU);
--unet-mode lora (default) is the L4 fallback. --min-snr 0 gives the paper's plain L_diff.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from lam.dataset import CurriculumSampler, SampleDataset, collate_samples, load_index
from lam.descriptor import collate, swap_text_field
from lam.losses import AUX_MAX_T, AuxLosses, LossWeights, diffusion_loss, predict_x0, swap_consistency
from lam.model import BASE_MODEL, LAMDiffusion


def sample_grid(model: LAMDiffusion, specs: list[dict], out: Path, step: int) -> None:
    from lam.pipeline import LAMDiffusionPipeline

    pipe = sample_grid.pipe = getattr(sample_grid, "pipe", None) or LAMDiffusionPipeline(model)
    tiles = []
    for i, d in enumerate(specs):
        w, h = d["bucket"]
        img = pipe([d], width=w, height=h, steps=20, cfg=5.0, seed=i)[0].resize((w // 4, h // 4))
        draw = ImageDraw.Draw(img)
        for c in d["characters"]:
            x0, y0, x1, y1 = c["bbox"]
            draw.rectangle([x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height], outline=(255, 0, 255))
        tiles.append(img)
    th = max(t.height for t in tiles)
    grid = Image.new("RGB", (sum(t.width for t in tiles), th), "white")
    x = 0
    for t in tiles:
        grid.paste(t, (x, 0))
        x += t.width
    out.mkdir(parents=True, exist_ok=True)
    grid.save(out / f"step{step:06d}.jpg", quality=90)
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--samples", default="/content/data/samples")
    ap.add_argument("--ckpt-dir", default="/content/ckpt")
    ap.add_argument("--init", default="/content/ckpt/stage2.pt", help="LAM Encoder from stage 2")
    ap.add_argument("--resume", default="", help="latest.pt to continue from")
    ap.add_argument("--steps", type=int, default=60000, help="optimizer steps over the whole run (curriculum horizon)")
    ap.add_argument("--hours", type=float, default=5.0, help="stop (and checkpoint) after this long; resume later")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unet-mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--unet-path", default="", help="full mode: start from this saved U-Net folder (resume from Drive)")
    ap.add_argument("--unet-lr", type=float, default=1e-5, help="learning rate of the U-Net weights in full mode")
    ap.add_argument("--min-snr", type=float, default=5.0, help="min-SNR-γ weighting of L_diff; 0 = plain ε-MSE (paper)")
    ap.add_argument("--weights", default="", help='JSON override, e.g. {"action_img":0.05,"perc":0.1,"style":50}')
    ap.add_argument("--no-aux", action="store_true", help="ablation: L_diff only")
    ap.add_argument("--overfit", type=int, default=0, help="train on the first N samples only (sanity check)")
    ap.add_argument("--sample-every", type=int, default=1500)
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    a = ap.parse_args()

    from diffusers import DDPMScheduler

    torch.backends.cuda.matmul.allow_tf32 = True
    # torch 2.11 + H200: the cuDNN fused-attention backward returns NaN gradients for t >= ~500 (found
    # 2026-10-11; flash / mem-efficient / math are fine). Same code was finite on the Colab A100.
    torch.backends.cuda.enable_cudnn_sdp(False)
    ck = Path(a.ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    w = LossWeights(**json.loads(a.weights)) if a.weights else LossWeights()
    train, test = load_index(a.index, "train"), load_index(a.index, "test")
    if a.overfit:
        train = train[: a.overfit]
    specs = (train[:4] if a.overfit else test[:: max(1, len(test) // 8)][:8])

    model = LAMDiffusion(gradient_checkpointing=True, unet_mode=a.unet_mode, unet_path=a.unet_path or None)
    print("parameters:", model.count(), flush=True)
    if a.init and Path(a.init).exists() and not a.resume:
        model.load_trainable(a.init, strict=False)
        print("initialised LAM Encoder from", a.init, flush=True)
    aux = None if a.no_aux else AuxLosses()
    groups = model.param_groups(lr_new=a.lr, lr_unet_lora=a.lr * 0.5, lr_te_lora=a.lr * 0.2, lr_gate=a.lr * 10,
                                lr_unet_full=a.unet_lr)
    import bitsandbytes as bnb

    # Paged 8-bit states for the full U-Net: optimizer memory can spill to CPU instead of OOM.
    opt = (bnb.optim.PagedAdamW8bit if a.unet_mode == "full" else bnb.optim.AdamW8bit)(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    noise_sched = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    alphas = noise_sched.alphas_cumprod.cuda()

    step = 0
    if a.resume:
        st = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in st["model"].items()}, strict=False)
        if "opt" in st:  # a weights-only file (e.g. synced back from the Mac) restarts the optimizer
            opt.load_state_dict(st["opt"])
        step = st["step"]
        if aux is not None and "queue" in st:
            aux.queue.copy_(st["queue"].cuda())
        print(f"resumed at step {step}", flush=True)

    sampler = CurriculumSampler(train, a.steps * a.accum, a.batch, start_step=step * a.accum)
    loader = DataLoader(SampleDataset(train, a.samples), batch_sampler=sampler, num_workers=6,
                        collate_fn=collate_samples, pin_memory=True, persistent_workers=True)
    trainable = [p for g in groups for p in g["params"]]
    rng = random.Random(step)
    t_start = last_ckpt = time.time()
    hist: dict[str, float] = {}
    n_hist = 0
    model.train()

    def save(tag: str) -> None:
        state = {"model": model.trainable_state(), "opt": opt.state_dict(), "step": step,
                 "queue": aux.queue.cpu() if aux is not None else None, "args": vars(a)}
        torch.save(state, ck / "latest.tmp")
        (ck / "latest.tmp").rename(ck / "latest.pt")
        # Weights-only copy in bf16 (~half the size) is what gets synced to the Mac; keep the last 3.
        # In full mode it holds the LAM part only; the U-Net goes to its own diffusers folder.
        small = model.trainable_state(include_unet=False)
        torch.save({"model": {k: v.to(torch.bfloat16) for k, v in small.items()}, "step": step,
                    "unet_mode": a.unet_mode}, ck / f"weights_{tag}.pt")
        for old in sorted(ck.glob("weights_step*.pt"))[:-3]:
            old.unlink()
        if a.unet_mode == "full":
            model.save_unet(ck / f"unet_{tag}")
            for old in sorted(ck.glob("unet_step*"))[:-2]:
                shutil.rmtree(old)
        print(f"checkpoint {tag} at step {step}", flush=True)

    micro = 0
    for batch in loader:
        lat = batch["latents"].cuda(non_blocking=True)
        B = lat.shape[0]
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device="cuda")
        ab = alphas[t]
        noise = torch.randn_like(lat)
        noisy = ab.sqrt().view(-1, 1, 1, 1) * lat + (1 - ab).sqrt().view(-1, 1, 1, 1) * noise
        desc = collate(batch["descs"], train=True, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lam = model.encode(desc)
            eps = model.denoise(noisy, t, lam, batch["time_ids"].cuda())
            l_diff = diffusion_loss(eps, noise, ab / (1 - ab), gamma=a.min_snr)
            loss = w.diff * l_diff
            parts = {"diff": l_diff.detach()}
            if w.swap > 0 and not desc.drop_all.all():
                desc_cf, f_sw = swap_text_field(desc, rng)
                l_swap = swap_consistency(lam, model.encode(desc_cf), f_sw)
                loss = loss + w.swap * l_swap
                parts["swap"] = l_swap.detach()
            low = (t < AUX_MAX_T) & ~desc.drop_all.cuda()
            if aux is not None and low.any():
                x0 = predict_x0(noisy[low], eps[low].float(), ab[low])
                r = aux(x0, batch["gt"].cuda()[low], lam.action_embed[low])
                loss = loss + w.action_img * r["action_img"] + w.perc * r["perc"] + w.style * r["style"]
                parts.update({k: v.detach() for k, v in r.items()})
        (loss / a.accum).backward()
        micro += 1
        for k, v in parts.items():
            hist[k] = hist.get(k, 0.0) + float(v)
        n_hist += 1
        if micro % a.accum:
            continue

        # optimizer step with warmup + cosine decay
        frac = step / a.steps
        scale = min(1.0, (step + 1) / max(1, min(500, a.steps // 20))) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            gates = model.conditioning.gates()
            mean = {k: float(f'{v / n_hist:.4g}') for k, v in hist.items()}
            el = time.time() - t_start
            print(f"step {step}/{a.steps} pool={sampler.pool_fraction(step * a.accum):.1f} {mean} "
                  f"gate mean {gates.mean():.3f} max {gates.max():.3f} | {el / 3600:.2f}h", flush=True)
            hist, n_hist = {}, 0
        if step % a.sample_every == 0:
            sample_grid(model, specs, ck / "samples", step)
        if time.time() - last_ckpt > a.ckpt_minutes * 60:
            save(f"step{step:06d}")
            last_ckpt = time.time()
        if time.time() - t_start > a.hours * 3600 or step >= a.steps:
            break
    save(f"step{step:06d}")
    sample_grid(model, specs, ck / "samples", step)
    print("done at step", step, flush=True)


if __name__ == "__main__":
    main()
