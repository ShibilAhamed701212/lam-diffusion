"""Algorithm 1 / Eq. 4: train G_θ (h_i = VAE latent -> structured action descriptor a_i).

Uses the precomputed latents and the descriptor annotations; no diffusion involved, so it is
cheap (~15-20 min on any GPU). Reports per-field accuracy on the held-out split.
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
from torch.utils.data import DataLoader, Dataset, Sampler

from lam.dataset import load_index
from lam.inverse import ActionFromLatent, build_vocab, targets


class Latents(Dataset):
    def __init__(self, index: list[dict], samples: Path):
        self.items, self.dir = index, samples

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        d = self.items[i]
        with np.load(self.dir / d["file"]) as z:
            return d, torch.from_numpy(z["latents"].astype(np.float32))


class BucketBatches(Sampler[list[int]]):
    """Batches of one aspect bucket (latents of a batch must share a shape)."""

    def __init__(self, index: list[dict], batch: int, shuffle: bool, seed: int = 0):
        self.groups: dict[tuple, list[int]] = {}
        for i, d in enumerate(index):
            self.groups.setdefault(tuple(d["bucket"]), []).append(i)
        self.batch, self.shuffle, self.rng = batch, shuffle, random.Random(seed)

    def __iter__(self):
        batches = []
        for ids in self.groups.values():
            ids = ids[:]
            if self.shuffle:
                self.rng.shuffle(ids)
            batches += [ids[i:i + self.batch] for i in range(0, len(ids), self.batch)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(math.ceil(len(v) / self.batch) for v in self.groups.values())


def collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


@torch.no_grad()
def evaluate(model: ActionFromLatent, loader, vocab) -> dict[str, float]:
    model.eval()
    acc = {k: [0, 0] for k in ("present", "gender", "age", "framing", "view")}
    bbox_l1, act_tp, act_fp, act_fn = 0.0, 0, 0, 0
    for descs, lat in loader:
        o = model(lat.cuda())
        t = {k: v.cuda() for k, v in targets(descs, vocab).items()}
        pres = t["present"].bool()
        acc["present"][0] += ((o["present"].sigmoid() > 0.5) == pres).sum().item()
        acc["present"][1] += pres.numel()
        for k in ("gender", "age", "framing", "view"):
            acc[k][0] += (o[k].argmax(-1)[pres] == t[k][pres]).sum().item()
            acc[k][1] += pres.sum().item()
        bbox_l1 += (o["bbox"].sigmoid()[pres] - t["bbox"][pres]).abs().mean(-1).sum().item()
        pa, ta = o["actions"].sigmoid()[pres] > 0.4, t["actions"][pres].bool()
        act_tp += (pa & ta).sum().item()
        act_fp += (pa & ~ta).sum().item()
        act_fn += (~pa & ta).sum().item()
    model.train()
    n_char = acc["gender"][1]
    r = {f"{k}_acc": v[0] / max(1, v[1]) for k, v in acc.items()}
    r["bbox_l1"] = bbox_l1 / max(1, n_char)
    r["action_f1"] = 2 * act_tp / max(1, 2 * act_tp + act_fp + act_fn)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--samples", default="/content/data/samples")
    ap.add_argument("--out", default="/content/ckpt/inverse.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    samples = Path(a.samples)
    train, test = load_index(a.index, "train"), load_index(a.index, "test")
    vocab = build_vocab(train)
    model = ActionFromLatent(vocab).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    tl = DataLoader(Latents(train, samples), batch_sampler=BucketBatches(train, a.batch, True), num_workers=8,
                    collate_fn=collate, persistent_workers=True)
    el = DataLoader(Latents(test, samples), batch_sampler=BucketBatches(test, 64, False), num_workers=4,
                    collate_fn=collate)
    total = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * s / total)))
    step, t0, log = 0, time.time(), []
    for ep in range(a.epochs):
        for descs, lat in tl:
            t = {k: v.cuda() for k, v in targets(descs, vocab).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(lat.cuda())
            loss, parts = model.loss({k: v.float() for k, v in o.items()}, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.3f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        r = evaluate(model, el, vocab)
        log.append({"epoch": ep, **r})
        print("eval:", r, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab, "log": log}, a.out)
    json.dump(log, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
