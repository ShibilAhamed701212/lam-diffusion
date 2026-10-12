"""Generate evaluation images for one system on the held-out descriptors.

Systems (same descriptors, seeds, steps, CFG and negative prompt for all):
  A          prompt-only Animagine XL: the natural-language request (paper's baseline)
  B          §V-style: planner spec -> hand-written tags + OpenPose ControlNet (ground-truth keypoints)
  C          LAM-Diffusion §IV (trained)
  C_noregion C without the bbox region masks          (ablation)
  C_notemp   C without the Temporal Attention stream  (ablation)
  C_global   C with the action stream switched off    (ablation: global descriptor only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))

NEGATIVE = ("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, "
            "worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry")
QUALITY = "masterpiece, high score, great score, absurdres"

LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def openpose_from_keypoints(d: dict, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), "black")
    dr = ImageDraw.Draw(img)
    r = max(3, w // 200)
    for c in d["characters"]:
        k = c["keypoints"]
        for i, (a, b) in enumerate(LIMBS):
            if k[a][2] > 0.3 and k[b][2] > 0.3:
                dr.line([k[a][0] * w, k[a][1] * h, k[b][0] * w, k[b][1] * h], fill=tuple(int(x * 0.6) for x in COLORS[i]), width=2 * r)
        for i, (x, y, s) in enumerate(k):
            if s > 0.3:
                dr.ellipse([x * w - r, y * h - r, x * w + r, y * h + r], fill=COLORS[i])
    return img


def tag_prompt(d: dict) -> str:
    """§V system: the app's rule-based spec -> tag translation (server/prompting.py)."""
    from prompting import build_prompt
    from spec import normalize_spec

    chars = []
    for c in d["characters"]:
        chars.append({**{k: c.get(k, "") for k in ("gender", "age", "hair_color", "hair_style", "eye_color", "outfit", "expression", "extra")},
                      "action": ", ".join(a.replace("_", " ") for a in c.get("actions", [])),
                      "framing": c["framing"] if c["framing"] != "unspecified" else "full_body",
                      "bbox": c["bbox"]})
    spec = normalize_spec({"scene": d["global"]["scene"], "style": d["global"]["style"], "props": d["global"]["props"],
                           "characters": chars})
    return build_prompt(spec)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--pairs", default="/content/data/planner_pairs.jsonl")
    ap.add_argument("--ckpt", default="/content/ckpt/latest.pt")
    ap.add_argument("--unet", default="", help="trained U-Net folder from a full-U-Net run (unet_stepXXXXXX)")
    ap.add_argument("--vae", default="", help="anime VAE folder from stage 1 (vae_anime)")
    ap.add_argument("--out", default="/content/eval")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    a = ap.parse_args()

    test = [d for d in map(json.loads, open(a.index)) if d["split"] == "test"][: a.n]
    texts = {p["id"]: p["text"] for p in map(json.loads, open(a.pairs)) if p["split"] == "test"}
    out = Path(a.out) / a.system
    out.mkdir(parents=True, exist_ok=True)
    todo = [d for d in test if not (out / f"{d['id']}.png").exists()]
    print(f"{a.system}: {len(todo)} to generate", flush=True)

    if a.system in ("A", "B"):
        from generator import Generator

        gen = Generator()

        def run(d: dict, seed: int) -> Image.Image:
            w, h = d["bucket"]
            if a.system == "A":
                prompt, pose = f"{texts[d['id']]}, {QUALITY}", None
            else:
                prompt, pose = tag_prompt(d), openpose_from_keypoints(d, w, h)
            return gen.generate(prompt, NEGATIVE, pose, width=w, height=h, seed=seed, steps=a.steps, cfg=a.cfg, control_scale=0.8)
    else:
        from lam.model import LAMDiffusion
        from lam.pipeline import LAMDiffusionPipeline

        # Full-U-Net run (§IV-F stage 3 as specified): the trained U-Net folder + LAM weights.
        model = LAMDiffusion(gradient_checkpointing=False, unet_mode="full" if a.unet else "lora", unet_path=a.unet or None)
        model.load_trainable(a.ckpt, strict=False)
        if a.system == "C_notemp":
            with torch.no_grad():
                model.encoder.temporal.gate.zero_()
        pipe = LAMDiffusionPipeline(model, vae_path=a.vae or None)  # anime VAE decoder (§IV-F stage 1)

        def run(d: dict, seed: int) -> Image.Image:
            w, h = d["bucket"]
            d = {**d, "global": {**d["global"], "style": ", ".join(x for x in (d["global"]["style"], QUALITY) if x)}}
            return pipe([d], width=w, height=h, steps=a.steps, cfg=a.cfg, seed=seed, negative=NEGATIVE,
                        region_masking=a.system != "C_noregion", action_scale=0.0 if a.system == "C_global" else 1.0)[0]

    for i, d in enumerate(todo):
        run(d, seed=d["id"] % 100000).save(out / f"{d['id']}.png")
        if (i + 1) % 50 == 0:
            print(f"{a.system}: {i + 1}/{len(todo)}", flush=True)
    print("done", a.system)


if __name__ == "__main__":
    main()
