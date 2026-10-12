"""Phase A quality gate: render the stage-2 checkpoint on held-out (test) specs.

Rows: ground truth | as specified | eye colour swapped | hair colour swapped | action off.
Colour leakage shows up as the eye swap changing the hair (or the reverse). Also renders the
test specs with the most actions to check multi-step ordering.

    python -m lam.tests.diag_phase_a --ckpt /content/ckpt/stage2_from2a.pt --out /content/diag_phase_a.jpg
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from PIL import Image, ImageDraw

from lam.dataset import load_index
from lam.model import LAMDiffusion
from lam.pipeline import LAMDiffusionPipeline
from lam.tests.diag_step0 import render

SWAP = {"eye_color": "red eyes", "hair_color": "blonde hair"}


def swapped(d: dict, field: str) -> dict:
    d = copy.deepcopy(d)
    d["characters"][0][field] = SWAP[field]
    return d


def sheet(rows: list[tuple[str, list[Image.Image]]], out: str) -> None:
    tw = max(sum(t.width for t in r) for _, r in rows) + 170
    th = sum(max(t.height for t in r) + 8 for _, r in rows)
    img = Image.new("RGB", (tw, th), "white")
    dr = ImageDraw.Draw(img)
    y = 0
    for label, tiles in rows:
        dr.text((6, y + 10), label, fill=(0, 0, 0))
        x = 170
        for t in tiles:
            img.paste(t, (x, y))
            x += t.width + 4
        y += max(t.height for t in tiles) + 8
    img.save(out, quality=90)
    print("saved", out, flush=True)


def gt_tiles(specs: list[dict], samples: str) -> list[Image.Image]:
    return [Image.open(Path(samples) / d["file"].replace(".npz", ".gt.jpg")).convert("RGB")
            .resize((d["bucket"][0] // 4, d["bucket"][1] // 4)) for d in specs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--samples", default="/content/data/samples")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unet", default="", help="full-U-Net folder from the full run")
    ap.add_argument("--out", default="/content/diag_phase_a.jpg")
    a = ap.parse_args()

    test = load_index(a.index, "test")
    one = [d for d in test if len(d["characters"]) == 1 and d["characters"][0].get("eye_color")
           and d["characters"][0].get("hair_color")][:5]
    multi = sorted(test, key=lambda d: -sum(len(c.get("actions", [])) for c in d["characters"]))[:5]
    for d in one + multi:
        print(d["id"], d["global"], [(c.get("hair_color"), c.get("eye_color"), c.get("actions")) for c in d["characters"]])

    model = LAMDiffusion(gradient_checkpointing=False, unet_mode="full" if a.unet else "lora", unet_path=a.unet or None)
    model.load_trainable(a.ckpt, strict=False)
    pipe = LAMDiffusionPipeline(model)

    rows = [("GT (test)", gt_tiles(one, a.samples)),
            ("as specified", render(pipe, one, quality=True)),
            ("eye -> red", render(pipe, [swapped(d, "eye_color") for d in one], quality=True)),
            ("hair -> blonde", render(pipe, [swapped(d, "hair_color") for d in one], quality=True)),
            ("action off", render(pipe, one, quality=True, action_scale=0.0))]
    sheet(rows, a.out)
    sheet([("GT (most actions)", gt_tiles(multi, a.samples)),
           ("as specified", render(pipe, multi, quality=True))], a.out.replace(".jpg", "_multi.jpg"))


if __name__ == "__main__":
    main()
