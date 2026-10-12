"""Diagnosis for the failed M3a overfit: what does the UNTRAINED model draw for the overfit specs?

Rows (same 4 specs, seeds, steps, CFG as the overfit's sample grids):
  gt          ground truth
  fresh       fresh LAM Encoder (no checkpoint), action stream on      <- does the design draw characters at step 0?
  fresh_off   fresh, action stream off (global descriptor only = base Animagine on the scene text)
  fresh_nomask fresh, action stream on, region masks off
  old_ckpt    encoder initialised from the OLD-design stage2.pt         <- is the old checkpoint what breaks it?
Also prints character-token norms vs the U-Net's normal text-token norms for fresh and old_ckpt.

--ckpt PATH instead renders a trained checkpoint, with and without the evaluation's quality tags and
negative prompt (training sample grids have neither, which alone makes Animagine grey and blurry).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from lam.dataset import load_index
from lam.descriptor import collate
from lam.model import LAMDiffusion
from lam.pipeline import LAMDiffusionPipeline


@torch.no_grad()
def token_stats(model: LAMDiffusion, specs: list[dict]) -> dict[str, float]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lam = model.encode(collate(specs))
    g = lam.global_hidden.float()[:, 1:].norm(dim=-1)  # skip BOS (outsized norm)
    c = lam.char_tokens.float().norm(dim=-1)[lam.char_token_mask]  # visible character tokens
    return {"global_token_norm": g.mean().item(), "char_token_norm_mean": c.mean().item(),
            "char_token_norm_max": c.max().item(), "pooled_norm": lam.pooled.float().norm(dim=-1).mean().item()}


QUALITY = "safe, masterpiece, high score, great score, absurdres"  # "safe": Animagine's rating tag
NEGATIVE = ("nsfw, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, "
            "worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry")


def with_quality(d: dict) -> dict:
    g = d["global"]
    return {**d, "global": {**g, "style": ", ".join(x for x in (g.get("style", ""), QUALITY) if x)}}


def render(pipe: LAMDiffusionPipeline, specs: list[dict], quality: bool = False, **kw) -> list[Image.Image]:
    out = []
    if quality:
        specs = [with_quality(d) for d in specs]
        kw.setdefault("negative", NEGATIVE)
    for i, d in enumerate(specs):
        w, h = d["bucket"]
        img = pipe([d], width=w, height=h, steps=20, cfg=5.0, seed=i, **kw)[0].resize((w // 4, h // 4))
        dr = ImageDraw.Draw(img)
        for c in d["characters"]:
            x0, y0, x1, y1 = c["bbox"]
            dr.rectangle([x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height], outline=(255, 0, 255))
        out.append(img)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--samples", default="/content/data/samples")
    ap.add_argument("--old", default="/content/ckpt/stage2.pt", help="'' to skip (the old checkpoint predates the token layout)")
    ap.add_argument("--out", default="/content/diag_step0.jpg")
    ap.add_argument("--ckpt", default="", help="render this trained checkpoint instead of the step-0 variants")
    a = ap.parse_args()

    specs = load_index(a.index, "train")[:4]  # the overfit run's sample specs (train[:16] -> first 4)
    for d in specs:
        print(d["id"], d["bucket"], d["global"], [(c["gender"], c.get("hair_color"), c.get("actions")) for c in d["characters"]])
    rows: list[tuple[str, list[Image.Image]]] = []
    gts = []
    for d in specs:
        w, h = d["bucket"]
        gts.append(Image.open(Path(a.samples) / d["file"].replace(".npz", ".gt.jpg")).convert("RGB").resize((w // 4, h // 4)))
    rows.append(("ground truth", gts))

    model = LAMDiffusion(gradient_checkpointing=False)
    pipe = LAMDiffusionPipeline(model)
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in sd["model"].items()}, strict=False)
        print(f"loaded {a.ckpt} (step {sd.get('step')}); token stats:", token_stats(model, specs), flush=True)
        rows.append(("trained, as in training grids", render(pipe, specs)))
        rows.append(("trained + quality tags", render(pipe, specs, quality=True)))
        rows.append(("trained + quality, action off", render(pipe, specs, quality=True, action_scale=0.0)))
        rows.append(("trained + quality, action x0.5", render(pipe, specs, quality=True, action_scale=0.5)))
    else:
        print("fresh token stats:", token_stats(model, specs), flush=True)
        rows.append(("fresh, step 0", render(pipe, specs)))
        rows.append(("fresh, action off", render(pipe, specs, action_scale=0.0)))
        rows.append(("fresh, no region mask", render(pipe, specs, region_masking=False)))
        rows.append(("fresh + quality tags", render(pipe, specs, quality=True)))
        if a.old:
            model.load_trainable(a.old, strict=False)
            print("old-ckpt token stats:", token_stats(model, specs), flush=True)
            rows.append(("old stage2.pt, step 0", render(pipe, specs)))

    tw = max(sum(t.width for t in r) for _, r in rows) + 170
    th = sum(max(t.height for t in r) + 8 for _, r in rows)
    sheet = Image.new("RGB", (tw, th), "white")
    dr = ImageDraw.Draw(sheet)
    y = 0
    for label, tiles in rows:
        dr.text((6, y + 10), label, fill=(0, 0, 0))
        x = 170
        for t in tiles:
            sheet.paste(t, (x, y))
            x += t.width + 4
        y += max(t.height for t in tiles) + 8
    sheet.save(a.out, quality=90)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
