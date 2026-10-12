"""§IV-F stage 2, first half (2a): pre-train the LAM Encoder on an action-text corpus (§IV-B:
"pre-trained on a large-scale action-text corpus and subsequently fine-tuned on an anime-specific
action-pose-text triplet dataset"; the fine-tuning half is train_stage2.py).

Corpus: HumanML3D captions (lam/data/humanml3d.py). Three objectives:
  L_cap   contrastive: two captions of the same motion are positives -> the transformer backbone
          (CLIP text encoders, LoRA) learns action semantics ("jumping while raising both arms").
  L_seq   the Temporal Attention context of the ordered sub-action sequence must retrieve its
          caption -> Action Dynamics embeddings and the temporal module learn what sequences mean.
  L_order true vs shuffled sub-action order from the temporal context -> the Temporal Attention
          Module (§IV-C) is trained on genuinely ordered actions ("running, then jumping, then
          landing"), which still anime images cannot supervise.
  L_keep  keeps the backbone's hidden states close to the frozen ones, so the global text stream
          the pre-trained U-Net reads does not drift.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lam.data.humanml3d import download, load
from lam.descriptor import ACTIONS, MAX_ACTIONS, N_KEYPOINTS
from lam.encoder import CTX_DIM, field_mean
from lam.model import LAMDiffusion


def info_nce(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = scale * F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()
    t = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t)) / 2


class Heads(nn.Module):
    """Training-only projections (discarded after stage 2a)."""

    def __init__(self):
        super().__init__()
        self.text = nn.Linear(CTX_DIM, 512)
        self.seq = nn.Linear(CTX_DIM, 512)
        self.order = nn.Sequential(nn.Linear(CTX_DIM, 512), nn.GELU(), nn.Linear(512, 1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))


def act_ids(seqs: list[list[str]], device) -> torch.Tensor:
    out = torch.zeros(len(seqs), MAX_ACTIONS, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        for t, a in enumerate(s[:MAX_ACTIONS]):
            out[i, t] = ACTIONS.index(a)
    return out


def shuffled(seq: list[str], rng: random.Random) -> list[str] | None:
    """A different ordering of the same sub-actions (None if every ordering is identical)."""
    if len(set(seq)) < 2:
        return None
    while True:
        s = seq[:]
        rng.shuffle(s)
        if s != seq:
            return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/content/data/humanml3d")
    ap.add_argument("--out", default="/content/ckpt/stage2a.pt")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--keep-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()

    root = download(Path(a.corpus))
    train, test = load(root, "train"), load(root, "test")
    print(f"HumanML3D motions: train {len(train)} test {len(test)}", flush=True)

    model = LAMDiffusion(gradient_checkpointing=True)
    enc = model.encoder
    bb = enc.backbone
    heads = Heads().cuda()
    params = [p for p in enc.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    zeros_kps = lambda n: torch.zeros(n, N_KEYPOINTS, 3, device="cuda")  # noqa: E731  text-only corpus: no skeleton

    def caption_embed(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mask, _, _ = bb.encode(texts, 77)
        return field_mean(hidden.float(), mask), hidden

    def temporal_ctx(seqs: list[list[str]]) -> torch.Tensor:
        ids = act_ids(seqs, "cuda")
        tok, valid = enc.dynamics(ids, zeros_kps(len(seqs)))
        return enc.temporal.context(tok, valid).float()

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        enc.eval()
        ms = test[:1000]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ea = torch.cat([caption_embed([m["captions"][0] for m in ms[i:i + 128]])[0] for i in range(0, len(ms), 128)])
            eb = torch.cat([caption_embed([m["captions"][-1] for m in ms[i:i + 128]])[0] for i in range(0, len(ms), 128)])
        sim = F.normalize(heads.text(ea), dim=-1) @ F.normalize(heads.text(eb), dim=-1).t()
        r1 = (sim.argmax(1) == torch.arange(len(ms), device=sim.device)).float().mean().item()
        rng = random.Random(1)
        seqs = [s for m in test for s in m["sequences"] if len(set(s)) >= 2][:2000]
        neg = [shuffled(s, rng) for s in seqs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pt = heads.order(temporal_ctx(seqs)).squeeze(-1)
            pn = heads.order(temporal_ctx(neg)).squeeze(-1)
        order_acc = ((pt > 0).float().mean() + (pn < 0).float().mean()).item() / 2
        enc.train()
        return {"caption_R@1": r1, "order_acc": order_acc, "n_order_test": len(seqs)}

    base = evaluate()
    print("before 2a:", base, flush=True)
    rng = random.Random(0)
    multi = [(m, s, c) for m in train for s, c in zip(m["sequences"], m["captions"]) if len(set(s)) >= 2]
    print(f"ordered multi-step sequences for the temporal module: {len(multi)}", flush=True)
    step, t0, log = 0, time.time(), []
    while step < a.steps:
        batch = rng.sample(train, a.batch)
        cap_a, cap_b = [], []
        for m in batch:
            c = rng.sample(m["captions"], 2) if len(m["captions"]) > 1 else m["captions"] * 2
            cap_a.append(c[0])
            cap_b.append(c[1])
        seq_items = rng.sample(multi, a.batch // 2)
        seqs, seq_caps = [s for _, s, _ in seq_items], [c for _, _, c in seq_items]
        negs = [shuffled(s, rng) for s in seqs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ea, ha = caption_embed(cap_a)
            eb, _ = caption_embed(cap_b)
            l_cap = info_nce(heads.text(ea), heads.text(eb), heads.logit_scale.exp().clamp(max=100))
            ec, _ = caption_embed(seq_caps)
            ctx_t, ctx_n = temporal_ctx(seqs), temporal_ctx(negs)
            l_seq = info_nce(heads.seq(ctx_t), heads.text(ec), heads.logit_scale.exp().clamp(max=100))
            logits = heads.order(torch.cat([ctx_t, ctx_n])).squeeze(-1)
            labels = torch.cat([torch.ones(len(ctx_t)), torch.zeros(len(ctx_n))]).cuda()
            l_order = F.binary_cross_entropy_with_logits(logits.float(), labels)
            with torch.no_grad():
                bb.te.disable_adapters()
                bb.te2.disable_adapters()
                _, h_ref = caption_embed(cap_a)
                bb.te.enable_adapters()
                bb.te2.enable_adapters()
            l_keep = F.mse_loss(ha.float(), h_ref.float()) / h_ref.float().pow(2).mean()
            loss = l_cap + l_seq + l_order + a.keep_weight * l_keep
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{a.steps} cap {l_cap.item():.3f} seq {l_seq.item():.3f} order {l_order.item():.3f} "
                  f"keep {l_keep.item():.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate()
            log.append({"step": step, **r})
            print("eval:", r, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.trainable_state().items() if k.startswith("encoder.")}
    torch.save({"model": state, "before": base, "log": log}, a.out)
    json.dump({"before": base, "log": log}, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
