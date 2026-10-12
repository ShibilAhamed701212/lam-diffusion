"""Stage 4 of the dataset build: tensors the trainer needs, computed once.

Per sample (one .npz in samples/):
- latents: SDXL VAE latents (already * scaling_factor) at the sample's aspect bucket;
- <id>.gt.jpg: the ground-truth image at half bucket size, for L_perc / L_style;
- clip: CLIP ViT-L/14 image embedding (L2-normalized) for LAM pre-training (§IV-F stage 2);
- time_ids: SDXL micro-conditioning (original size, crop offset, target size).
Keypoints/bboxes are re-expressed in the cropped bucket frame. Also writes the train/test split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BUCKETS = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216), (1344, 768), (768, 1344)]
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CLIP_ID = "openai/clip-vit-large-patch14"


def pick_bucket(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    return min(BUCKETS, key=lambda b: abs(np.log(b[0] / b[1]) - np.log(ar)))


def cover_crop(img: Image.Image, bw: int, bh: int) -> tuple[Image.Image, float, float, float]:
    """Resize to cover the bucket then centre-crop. Returns image, scale, crop left/top (px)."""
    s = max(bw / img.width, bh / img.height)
    rw, rh = round(img.width * s), round(img.height * s)
    img = img.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - bw) // 2, (rh - bh) // 2
    return img.crop((left, top, left + bw, top + bh)), s, left, top


def remap(ann: dict, w: int, h: int, s: float, left: float, top: float, bw: int, bh: int) -> dict:
    """Normalized coords in the original image -> normalized coords in the bucket crop."""

    def pt(x: float, y: float) -> tuple[float, float]:
        return (x * w * s - left) / bw, (y * h * s - top) / bh

    for c in ann["characters"]:
        x0, y0 = pt(*c["bbox"][:2])
        x1, y1 = pt(*c["bbox"][2:])
        c["bbox"] = [round(min(1, max(0, v)), 4) for v in (x0, y0, x1, y1)]
        kps = []
        for x, y, sc in c["keypoints"]:
            nx, ny = pt(x, y)
            inside = 0 <= nx <= 1 and 0 <= ny <= 1
            kps.append([round(nx, 4), round(ny, 4), sc if inside else 0.0])
        c["keypoints"] = kps
    return ann


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="/content/data/annotations.jsonl")
    ap.add_argument("--images", default="/content/data/images")
    ap.add_argument("--out", default="/content/data/samples")
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    from diffusers import AutoencoderKL
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(dev).eval()
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(dev).eval()
    clip_proc = CLIPImageProcessor.from_pretrained(CLIP_ID)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    anns = [json.loads(line) for line in open(a.annotations)]
    anns = [x for x in anns if (Path(a.images) / f"{x['id']}.jpg").exists()]  # e.g. a failed re-download
    n_all = len(anns)
    # Tiny, distant characters carry little pose/attribute signal: keep images where every
    # character is at least 20% of the frame tall and the largest at least 35%.
    def big_enough(x: dict) -> bool:
        hs = [c["bbox"][3] - c["bbox"][1] for c in x["characters"]]
        return min(hs) >= 0.20 and max(hs) >= 0.35
    anns = [x for x in anns if big_enough(x)]
    print(f"size filter kept {len(anns)}/{n_all}", flush=True)
    random.Random(0).shuffle(anns)
    # Hold out a test set stratified by composition so every scenario is evaluated.
    by_comp: dict[str, list[dict]] = {}
    for x in anns:
        by_comp.setdefault(x["comp"], []).append(x)
    test_ids = set()
    for comp, xs in by_comp.items():
        k = max(1, round(a.test_size * len(xs) / len(anns)))
        test_ids.update(x["id"] for x in xs[:k])

    # Group by bucket so VAE batches share a shape.
    groups: dict[tuple[int, int], list[dict]] = {}
    for x in anns:
        groups.setdefault(pick_bucket(*x["size"]), []).append(x)

    def prepare(x: dict, bw: int, bh: int) -> tuple[Image.Image | None, dict]:
        """Bucket geometry + remapped annotation; loads/crops pixels only if not yet computed."""
        dest = out / f"{x['id']}.npz"
        path = Path(a.images) / f"{x['id']}.jpg"
        with Image.open(path) as probe:
            w, h = probe.size  # header only
        s = max(bw / w, bh / h)
        left, top = (round(w * s) - bw) // 2, (round(h * s) - bh) // 2
        ann = remap(json.loads(json.dumps(x)), w, h, s, left, top, bw, bh)
        ann.update(split="test" if x["id"] in test_ids else "train", bucket=[bw, bh], file=dest.name)
        # SDXL micro-conditioning: original size, crop top-left (in resized px), target size.
        ann["time_ids"] = [round(h * s), round(w * s), int(top), int(left), bh, bw]
        if dest.exists():
            return None, ann
        return cover_crop(Image.open(path).convert("RGB"), bw, bh)[0], ann

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(8)  # PIL decode/resize release the GIL
    index = open(a.index, "w")
    done = 0
    for (bw, bh), xs in groups.items():
        for j in range(0, len(xs), 8 * a.batch):
            prepared = list(pool.map(lambda x: prepare(x, bw, bh), xs[j:j + 8 * a.batch]))
            for i in range(0, len(prepared), a.batch):
                chunk = prepared[i:i + a.batch]
                imgs, metas = [], []
                for img, ann in chunk:
                    if img is None:  # resumable: tensors already on disk, only re-index
                        index.write(json.dumps(ann) + "\n")
                    else:
                        imgs.append(img)
                        metas.append(ann)
                if not imgs:
                    done += len(chunk)
                    continue
                with torch.no_grad():
                    px = torch.stack([torch.from_numpy(np.asarray(im)).permute(2, 0, 1) for im in imgs]).to(dev, torch.float16)
                    px = px / 127.5 - 1.0
                    lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                    cin = clip_proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, torch.float16)
                    emb = clip(pixel_values=cin).image_embeds
                    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                for im, ann, l, e in zip(imgs, metas, lat, emb):
                    # Ground truth as a JPEG next to the npz: raw uint8 arrays would need ~66 GB.
                    im.resize((bw // 2, bh // 2), Image.LANCZOS).save(out / ann["file"].replace(".npz", ".gt.jpg"), quality=92)
                    tmp = out / (ann["file"] + ".tmp")  # atomic: a killed write never looks finished
                    with open(tmp, "wb") as f:
                        np.savez(f, latents=l.cpu().numpy().astype(np.float16),
                                 clip=e.cpu().numpy().astype(np.float16), time_ids=np.array(ann["time_ids"], dtype=np.int32))
                    tmp.rename(out / ann["file"])
                    index.write(json.dumps(ann) + "\n")
                done += len(chunk)
                if done % 2000 < a.batch:
                    print(f"precomputed {done}/{len(anns)}", flush=True)
                    index.flush()
    index.close()
    print("done", done, "test", len(test_ids))


if __name__ == "__main__":
    main()
