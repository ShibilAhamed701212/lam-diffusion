"""§V-C metrics for every generated system, computed the same way for all of them.

- CLIP Score: 100 * cos(CLIP ViT-L/14 image, natural-language request), clipped at 0.
- FID / KID: clean-fid, generated vs the held-out real images (KID reported because FID is
  biased at ~1k samples).
- LPIPS / SSIM: against the held-out ground-truth image of the same descriptor.
- Attribute Binding Precision / Recall: WD tagger on each character's box; predicted
  gender / age / hair colour / eye colour tags vs that character's descriptor.
- Pose PCK@0.1: detected skeletons vs the descriptor's keypoints (matched by box overlap).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lam.data import tag_vocab as V


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM over RGB channels, 11x11 Gaussian window (sigma 1.5); inputs [3, H, W] in [0, 1]."""
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    win = (g[:, None] * g[None, :] / g.sum() ** 2).to(a)[None, None].expand(3, 1, 11, 11)
    a, b = a[None], b[None]
    mu_a, mu_b = F.conv2d(a, win, groups=3), F.conv2d(b, win, groups=3)
    va = F.conv2d(a * a, win, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, win, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, win, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean().item()


def spec_tags(c: dict) -> dict[str, set[str]]:
    """Tags the tagger should find on this character, per controlled category."""
    def colors(field: str, suffix: str) -> set[str]:
        return {f"{x.strip().replace(' ', '_')}_{suffix}" for x in c.get(field, "").split(",") if x.strip()}
    out = {"gender": {"1girl"} if c["gender"] == "female" else {"1boy"} if c["gender"] == "male" else set(),
           "hair": colors("hair_color", "hair") & set(V.HAIR_COLOR), "eyes": colors("eye_color", "eyes") & set(V.EYE_COLOR)}
    if c.get("age") == "elderly":
        out["age"] = {"old_man", "old_woman", "old"}
    return out


CATEGORY_TAGS = {"gender": {"1girl", "1boy"}, "hair": set(V.HAIR_COLOR), "eyes": set(V.EYE_COLOR),
                 "age": {"old_man", "old_woman", "old"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="A,B,C")
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--pairs", default="/content/data/planner_pairs.jsonl")
    ap.add_argument("--samples", default="/content/data/samples")
    ap.add_argument("--eval", default="/content/eval")
    ap.add_argument("--out", default="/content/eval/results.json")
    a = ap.parse_args()

    import lpips
    from cleanfid import fid
    from transformers import CLIPModel, CLIPProcessor

    from lam.data.annotate import Tagger, people

    dev = "cuda"
    test = {d["id"]: d for d in map(json.loads, open(a.index)) if d["split"] == "test"}
    texts = {p["id"]: p["text"] for p in map(json.loads, open(a.pairs)) if p["split"] == "test"}
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    tagger = Tagger()

    real_dir = Path(a.eval) / "_real"
    if not real_dir.exists():
        real_dir.mkdir(parents=True)
        for i, d in test.items():
            shutil.copy(Path(a.samples) / d["file"].replace(".npz", ".gt.jpg"), real_dir / f"{i}.jpg")

    results = json.load(open(a.out)) if Path(a.out).exists() else {}
    for system in a.systems.split(","):
        gen_dir = Path(a.eval) / system
        ids = [i for i in test if (gen_dir / f"{i}.png").exists()]
        if not ids:
            print("no images for", system)
            continue
        small = Path(a.eval) / f"_{system}_small"
        small.mkdir(exist_ok=True)
        clip_s, lp_s, ss_s = [], [], []
        tp = fp = fn = 0
        g_ok = g_n = 0
        pck_hit = pck_n = 0
        for i in ids:
            d = test[i]
            img = Image.open(gen_dir / f"{i}.png").convert("RGB")
            gt = Image.open(real_dir / f"{i}.jpg").convert("RGB")
            img_s = img.resize(gt.size, Image.LANCZOS)
            img_s.save(small / f"{i}.jpg", quality=95)
            with torch.no_grad():
                if i in texts:
                    x = proc(text=[texts[i]], images=img, return_tensors="pt", padding=True, truncation=True).to(dev)
                    o = clip(**x)
                    cos = F.cosine_similarity(o.image_embeds, o.text_embeds).item()
                    clip_s.append(100 * max(cos, 0.0))
                ta = torch.from_numpy(np.asarray(img_s)).permute(2, 0, 1).float().div(255).to(dev)
                tb = torch.from_numpy(np.asarray(gt)).permute(2, 0, 1).float().div(255).to(dev)
                lp_s.append(lp(ta[None] * 2 - 1, tb[None] * 2 - 1).item())
                ss_s.append(ssim(ta, tb))
            # attribute binding: tag each character's region of the generated image
            from lam.data.annotate import crop
            probs = tagger([crop(img, c["bbox"], pad=0.05) for c in d["characters"]])
            for c, pr in zip(d["characters"], probs):
                want = spec_tags(c)
                for cat, tags in CATEGORY_TAGS.items():
                    if cat not in want or not want[cat]:
                        continue
                    pred = {t for t in tags if pr.get(t, 0) >= 0.35}
                    tp += len(pred & want[cat])
                    fp += len(pred - want[cat])
                    fn += len(want[cat] - pred) if cat != "age" else int(not (pred & want[cat]))
                if c["gender"] in ("female", "male"):
                    g_n += 1
                    g_ok += (pr.get("1boy", 0) + pr.get("male_focus", 0) > pr.get("1girl", 0)) == (c["gender"] == "male")
            # pose: PCK@0.1 against the descriptor's keypoints
            found = people(img)
            for c in d["characters"]:
                bx = c["bbox"]
                best = max(found, key=lambda p: _iou(p["bbox"], bx), default=None)
                scale = 0.1 * max(bx[2] - bx[0], bx[3] - bx[1])
                for k, (x, y, s) in enumerate(c["keypoints"]):
                    if s <= 0.3:
                        continue
                    pck_n += 1
                    if best is not None and _iou(best["bbox"], bx) > 0.1:
                        px, py, ps = best["keypoints"][k]
                        pck_hit += ps > 0.3 and ((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= scale
        r = {
            "n": len(ids),
            "clip_score": float(np.mean(clip_s)) if clip_s else None,
            "fid": fid.compute_fid(str(small), str(real_dir), mode="clean", verbose=False),
            "kid": fid.compute_kid(str(small), str(real_dir), mode="clean", verbose=False),
            "lpips": float(np.mean(lp_s)),
            "ssim": float(np.mean(ss_s)),
            "attr_binding_precision": tp / max(1, tp + fp),
            "attr_recall": tp / max(1, tp + fn),
            "gender_accuracy": g_ok / max(1, g_n),
            "pose_pck@0.1": pck_hit / max(1, pck_n),
        }
        results[system] = r
        print(system, json.dumps(r), flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    main()
