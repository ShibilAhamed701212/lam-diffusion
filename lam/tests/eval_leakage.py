"""Quantitative attribute-leakage check on held-out test specs.

For single-character specs with a recognisable hair and eye colour, render (a) as specified,
(b) with only the eye colour swapped, (c) with only the hair colour swapped. A CLIP zero-shot
classifier reads the hair / eye colour of the character crop. Leakage = how much the *other*
attribute's accuracy drops when one attribute is swapped:

    hair_acc(orig) - hair_acc(eye swapped)      eye_acc(orig) - eye_acc(hair swapped)

Run for several checkpoints to compare (baseline = stage-2 encoder only, U-Net untrained).

    python -m lam.tests.eval_leakage --ckpt NAME=PATH [--ckpt NAME2=PATH2 ...] --n 60
"""

from __future__ import annotations

import argparse
import copy
import json
import re

import torch
from transformers import CLIPModel, CLIPProcessor

from lam.dataset import load_index
from lam.model import LAMDiffusion
from lam.pipeline import LAMDiffusionPipeline
from lam.tests.diag_step0 import NEGATIVE, with_quality

COLORS = ["red", "blue", "green", "purple", "blonde", "pink", "black", "white", "brown", "silver", "orange"]


def color_of(text: str) -> str | None:
    t = (text or "").lower()
    t = t.replace("yellow", "blonde").replace("gold", "blonde").replace("grey", "silver").replace("gray", "silver")
    hits = [c for c in COLORS if re.search(rf"\b{c}\b", t)]
    return hits[0] if len(hits) == 1 else None


def pick_specs(index: list[dict], n: int) -> list[dict]:
    out = []
    for d in index:
        if len(d["characters"]) != 1:
            continue
        c = d["characters"][0]
        if color_of(c.get("hair_color", "")) and color_of(c.get("eye_color", "")):
            out.append(d)
        if len(out) == n:
            break
    return out


def with_colors(d: dict, hair: str | None = None, eye: str | None = None) -> dict:
    d = copy.deepcopy(d)
    if hair:
        d["characters"][0]["hair_color"] = f"{hair} hair"
    if eye:
        d["characters"][0]["eye_color"] = f"{eye} eyes"
    return d


def other(color: str, i: int) -> str:
    pool = [c for c in COLORS if c != color and c not in ("black", "brown", "silver")]
    return pool[i % len(pool)]


@torch.no_grad()
def classify(clip, proc, imgs, kind: str) -> list[str]:
    prompts = [f"an anime character with {c} {kind}" for c in COLORS]
    inp = proc(text=prompts, images=imgs, return_tensors="pt", padding=True).to("cuda")
    return [COLORS[i] for i in clip(**inp).logits_per_image.argmax(-1).tolist()]


def crop(img, bbox):
    w, h = img.size
    x0, y0, x1, y1 = bbox
    return img.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 8), max(int(y1 * h), int(y0 * h) + 8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH or NAME=PATH@UNET_DIR (full-U-Net run)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="/content/leakage.json")
    a = ap.parse_args()

    specs = pick_specs(load_index(a.index, "test"), a.n)
    print(f"{len(specs)} usable test specs", flush=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    results = {}
    for item in a.ckpt:
        name, path = item.split("=", 1)
        path, _, unet = path.partition("@")  # NAME=WEIGHTS[@FULL_UNET_DIR]
        model = LAMDiffusion(gradient_checkpointing=False, unet_mode="full" if unet else "lora", unet_path=unet or None)
        model.load_trainable(path, strict=False)
        pipe = LAMDiffusionPipeline(model)

        def render(d: dict, seed: int):
            d = with_quality(d)
            w, h = d["bucket"]
            return pipe([d], width=w, height=h, steps=20, cfg=5.0, seed=seed, negative=NEGATIVE)[0]

        n = {"hair_orig": 0, "hair_eyeswap": 0, "eye_orig": 0, "eye_hairswap": 0}
        for i, d in enumerate(specs):
            c = d["characters"][0]
            hair, eye = color_of(c["hair_color"]), color_of(c["eye_color"])
            bb = c["bbox"]
            o = render(d, i)
            e = render(with_colors(d, eye=other(eye, i)), i)
            hs = render(with_colors(d, hair=other(hair, i)), i)
            n["hair_orig"] += classify(clip, proc, [crop(o, bb)], "hair")[0] == hair
            n["hair_eyeswap"] += classify(clip, proc, [crop(e, bb)], "hair")[0] == hair
            n["eye_orig"] += classify(clip, proc, [crop(o, bb)], "eyes")[0] == eye
            n["eye_hairswap"] += classify(clip, proc, [crop(hs, bb)], "eyes")[0] == eye
        N = len(specs)
        r = {k: round(v / N, 3) for k, v in n.items()}
        r["hair_leak"] = round(r["hair_orig"] - r["hair_eyeswap"], 3)
        r["eye_leak"] = round(r["eye_orig"] - r["eye_hairswap"], 3)
        results[name] = r
        print(name, r, flush=True)
        del pipe, model
        torch.cuda.empty_cache()
    json.dump({"n": len(specs), "results": results}, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
