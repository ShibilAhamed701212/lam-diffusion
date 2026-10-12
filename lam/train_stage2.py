"""§IV-F stage 2: pre-train / fine-tune the LAM Encoder on action-pose-text triplets.
(--init stage2a.pt continues from the action-text pre-training of train_stage2a.py.)

Contrastive alignment between the LAM action embedding (descriptor -> CLIP image space) and the
CLIP image embedding of the triplet's image, before the U-Net ever sees the action tokens.
Gate (M2): descriptor->image retrieval on the held-out set must clearly beat the
global-descriptor-only baseline (i.e. the character tokens carry real visual information).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lam.dataset import load_index
from lam.descriptor import collate
from lam.model import LAMDiffusion


class ClipPairs(Dataset):
    def __init__(self, index: list[dict], samples: Path):
        self.items, self.dir = index, samples

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        d = self.items[i]
        with np.load(self.dir / d["file"]) as z:
            return d, torch.from_numpy(z["clip"].astype(np.float32))


def pairs_collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


@torch.no_grad()
def retrieval(model: LAMDiffusion, test: list[dict], samples: Path, chars: bool = True) -> dict[str, float]:
    model.eval()
    embs, imgs = [], []
    for i in range(0, len(test), 64):
        descs = test[i:i + 64]
        if not chars:
            descs = [{**d, "characters": []} for d in descs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embs.append(F.normalize(model.encode(collate(descs)).action_embed.float(), dim=-1))
        for d in test[i:i + 64]:
            with np.load(samples / d["file"]) as z:
                imgs.append(torch.from_numpy(z["clip"].astype(np.float32)))
    e = torch.cat(embs)
    im = F.normalize(torch.stack(imgs).cuda(), dim=-1)
    ranks = (e @ im.t()).argsort(dim=1, descending=True)
    target = torch.arange(len(e), device=e.device)[:, None]
    pos = (ranks == target).float().argmax(1)
    model.train()
    return {"R@1": (pos < 1).float().mean().item(), "R@10": (pos < 10).float().mean().item(),
            "median_rank": pos.float().median().item() + 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--samples", default="/content/data/samples")
    ap.add_argument("--out", default="/content/ckpt/stage2.pt")
    ap.add_argument("--init", default="", help="encoder weights to start from, e.g. /content/ckpt/stage2a.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()

    samples = Path(a.samples)
    train, test = load_index(a.index, "train"), load_index(a.index, "test")
    print(f"train {len(train)} test {len(test)}", flush=True)
    model = LAMDiffusion(gradient_checkpointing=True)
    if a.init:
        model.load_trainable(a.init, strict=False)
        print("initialised LAM Encoder from", a.init, flush=True)
    params = [p for p in model.encoder.parameters() if p.requires_grad]
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device="cuda"))
    opt = torch.optim.AdamW([{"params": params}, {"params": [logit_scale], "lr": 1e-3}], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    loader = DataLoader(ClipPairs(train, samples), batch_size=a.batch, shuffle=True, num_workers=8,
                        collate_fn=pairs_collate, drop_last=True, persistent_workers=True)

    base = retrieval(model, test, samples, chars=False)
    print("baseline (global descriptor only):", base, flush=True)
    rng = random.Random(0)
    step, t0, log = 0, time.time(), []
    while step < a.steps:
        for descs, clip in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = model.encode(collate(descs, train=True, p_drop_all=0.0, rng=rng)).action_embed
            emb = F.normalize(emb.float(), dim=-1)
            img = F.normalize(clip.cuda(), dim=-1)
            logits = logit_scale.exp().clamp(max=100) * emb @ img.t()
            target = torch.arange(len(emb), device=emb.device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.3f} scale {logit_scale.exp().item():.1f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                r = retrieval(model, test, samples)
                log.append({"step": step, **r})
                print("eval:", r, flush=True)
            if step >= a.steps:
                break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    enc = {k: v for k, v in model.trainable_state().items() if k.startswith("encoder.")}
    torch.save({"model": enc, "baseline": base, "log": log}, a.out)
    json.dump({"baseline": base, "log": log}, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
