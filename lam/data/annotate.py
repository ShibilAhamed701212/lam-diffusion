"""Stage 3 of the dataset build: images -> structured action descriptors (paper Eq. 1's A).

For each image:
- poses: RTMPose-x + a YOLOX detector trained on HumanArt (illustrations incl. anime), 18
  OpenPose keypoints per person (same layout as server/poses.py);
- characters: the n largest valid skeletons (n from the human count tags), ordered left->right;
- attributes: solo images use the human-written Danbooru tags; multi-character images run the
  WD tagger on each person's crop so hair/eyes/outfit bind to the right character;
- global: scene / style / props from the image tags.
Writes one JSON line per kept image; resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from lam.data import tag_vocab as V

KP_THRESH = 0.3
MIN_VISIBLE = 6  # head + neck + shoulders: keeps upper-body / portrait framings, not just full bodies
TAG_THRESH = 0.35


# ------------------------------------------------------------------ models

class Tagger:
    """SmilingWolf WD EVA02-large v3 (timm). Preprocessing follows the reference wdv3_timm script."""

    REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

    def __init__(self, device: str = "cuda"):
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from timm.data import create_transform, resolve_data_config

        self.torch = torch
        self.model = timm.create_model(f"hf-hub:{self.REPO}", pretrained=True).eval().to(device).half()
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        with open(hf_hub_download(self.REPO, "selected_tags.csv")) as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"] for r in rows]
        self.device = device

    def __call__(self, images: list[Image.Image]) -> list[dict[str, float]]:
        batch = []
        for img in images:
            side = max(img.size)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            batch.append(self.transform(canvas))
        x = self.torch.stack(batch).to(self.device).half()[:, [2, 1, 0]]  # model expects BGR
        with self.torch.no_grad():
            probs = self.torch.sigmoid(self.model(x)).float().cpu().numpy()
        return [{n: float(p) for n, p in zip(self.names, row) if p >= 0.2} for row in probs]


_pose_local = threading.local()


def pose_model():
    if not hasattr(_pose_local, "m"):
        import onnxruntime as ort
        import torch  # noqa: F401  loads the CUDA/cuDNN libraries onnxruntime-gpu links against
        from rtmlib import Body

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        _pose_local.m = Body(mode="performance", to_openpose=True, backend="onnxruntime", device="cuda")
    return _pose_local.m


# ------------------------------------------------------------------ geometry

def people(img: Image.Image) -> list[dict]:
    """Valid skeletons as normalized keypoints + bbox, largest first."""
    arr = np.asarray(img)[:, :, ::-1]  # rtmlib expects BGR
    kps, scores = pose_model()(arr)
    w, h = img.size
    out = []
    for kp, sc in zip(kps, scores):
        vis = sc > KP_THRESH
        if vis.sum() < MIN_VISIBLE:
            continue
        xs, ys = kp[vis, 0], kp[vis, 1]
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0, y1 - y0
        # Skeletons stop at the nose/ears; extend upward for the head and a bit outward.
        x0, x1 = max(0, x0 - 0.12 * bw), min(w, x1 + 0.12 * bw)
        y0, y1 = max(0, y0 - 0.18 * bh), min(h, y1 + 0.05 * bh)
        out.append({
            "bbox": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
            "keypoints": [[round(float(x) / w, 4), round(float(y) / h, 4), round(float(s), 3)] for (x, y), s in zip(kp, sc)],
            "area": float((x1 - x0) * (y1 - y0)),
        })
    return sorted(out, key=lambda p: -p["area"])


def framing_from_keypoints(kps: list[list[float]]) -> str:
    """Most Danbooru posts carry no framing tag; infer it from which joints are in frame.
    OpenPose-18 indices: 10/13 ankles, 9/12 knees, 8/11 hips."""
    vis = [k[2] > KP_THRESH and 0 <= k[0] <= 1 and 0 <= k[1] <= 1 for k in kps]
    if vis[10] or vis[13]:
        return "full_body"
    if vis[9] or vis[12]:
        return "cowboy_shot"
    if vis[8] or vis[11]:
        return "upper_body"
    return "portrait"


def crop(img: Image.Image, bbox: list[float], pad: float = 0.1) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((int(max(0, x0 - dx) * w), int(max(0, y0 - dy) * h), int(min(1, x1 + dx) * w), int(min(1, y1 + dy) * h)))


# ------------------------------------------------------------------ attributes

def _ranked(probs: dict[str, float], table: dict[str, str], limit: int) -> list[str]:
    hits = sorted(((p, t) for t, p in probs.items() if t in table and p >= TAG_THRESH), reverse=True)
    return [table[t] for _, t in hits[:limit]]


def attrs_from_tags(tags: list[str]) -> dict:
    """Solo character: human Danbooru tags, picked in vocabulary order (most important first)."""
    tagset = set(tags)

    def pick(table: dict[str, str], limit: int) -> list[str]:
        return [v for t, v in table.items() if t in tagset][:limit]

    return {
        "age": V.age(tags),
        "hair_color": ", ".join(pick(V.HAIR_COLOR, 2)),
        "hair_style": ", ".join(pick(V.HAIR_STYLE, 3)),
        "eye_color": ", ".join(pick(V.EYE_COLOR, 1)),
        "outfit": ", ".join(pick(V.OUTFIT, 5)),
        "expression": ", ".join(pick(V.EXPRESSION, 2)),
        "extra": ", ".join(pick(V.EXTRA, 3)),
        "actions": V.actions(tags),
        "view": next((v for t, v in V.VIEW.items() if t in tagset), "front"),
    }


def attrs_from_probs(probs: dict[str, float]) -> dict:
    """One character in a multi-character image: WD tagger on that person's crop."""
    tags = [t for t, p in probs.items() if p >= TAG_THRESH]
    return {
        "age": V.age(tags),
        "hair_color": ", ".join(_ranked(probs, V.HAIR_COLOR, 2)),
        "hair_style": ", ".join(_ranked(probs, V.HAIR_STYLE, 3)),
        "eye_color": ", ".join(_ranked(probs, V.EYE_COLOR, 1)),
        "outfit": ", ".join(_ranked(probs, V.OUTFIT, 5)),
        "expression": ", ".join(_ranked(probs, V.EXPRESSION, 2)),
        "extra": ", ".join(_ranked(probs, V.EXTRA, 3)),
        "actions": V.actions(tags),
        "view": next((v for t, v in V.VIEW.items() if probs.get(t, 0) >= TAG_THRESH), "front"),
    }


def male_score(probs: dict[str, float]) -> float:
    return max(probs.get("1boy", 0), probs.get("male_focus", 0)) - probs.get("1girl", 0)


def annotate(sample: dict, img: Image.Image, tagger: Tagger | None, tag_lock: threading.Lock) -> dict | str:
    girls, boys, _ = V.person_counts(sample["tags"])
    n = girls + boys
    found = people(img)
    if len(found) < n:
        return "pose_missing"
    chosen = sorted(found[:n], key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2)
    framing = V.framing(sample["tags"])

    chars = []
    if n == 1:
        a = attrs_from_tags(sample["tags"])
        a["gender"] = "female" if girls else "male"
        chars.append(a)
    else:
        with tag_lock:
            probs = tagger([crop(img, p["bbox"]) for p in chosen])
        order = sorted(range(n), key=lambda i: -male_score(probs[i]))
        males = set(order[:boys])
        for i, pr in enumerate(probs):
            a = attrs_from_probs(pr)
            a["gender"] = "male" if i in males else "female"
            chars.append(a)
    for a, p in zip(chars, chosen):
        a["framing"] = framing if framing != "unspecified" else framing_from_keypoints(p["keypoints"])
        a["bbox"] = p["bbox"]
        a["keypoints"] = p["keypoints"]

    tagset = set(sample["tags"])
    return {
        "id": sample["id"],
        "comp": sample["comp"],
        "size": list(img.size),
        "global": {
            "scene": ", ".join(v for t, v in V.SCENE.items() if t in tagset)[:200],
            "style": ", ".join(v for t, v in V.STYLE.items() if t in tagset),
            "props": V.props(sample["tags"]),
        },
        "characters": chars,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="/content/data/selection.jsonl")
    ap.add_argument("--images", default="/content/data/images")
    ap.add_argument("--out", default="/content/data/annotations.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0, help="this process's shard (threads are GIL-bound, so run several processes)")
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    done = set()
    for f in Path(a.out).parent.glob(Path(a.out).name + "*"):  # main file + per-shard part files
        done |= {json.loads(line)["id"] for line in open(f) if line.strip()}
    sel = [json.loads(line) for line in open(a.selection)]
    todo = [s for s in sel if s["id"] not in done and s["id"] % a.num_shards == a.shard
            and (Path(a.images) / f"{s['id']}.jpg").exists()]
    if a.num_shards > 1:
        a.out = f"{a.out}.part{a.shard}"
    print(f"{len(done)} already annotated, {len(todo)} to go", flush=True)

    tagger = Tagger()
    tag_lock = threading.Lock()
    write_lock = threading.Lock()
    out = open(a.out, "a")
    stats: dict[str, int] = {}

    def work(s: dict) -> str:
        try:
            img = Image.open(Path(a.images) / f"{s['id']}.jpg").convert("RGB")
            res = annotate(s, img, tagger, tag_lock)
        except Exception as e:  # one bad image shouldn't stop the build
            return f"error:{type(e).__name__}"
        if isinstance(res, str):
            return res
        with write_lock:
            out.write(json.dumps(res) + "\n")
        return "ok"

    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(work, todo)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 1000 == 0:
                out.flush()
                print(f"annotated {i + 1}/{len(todo)} {stats}", flush=True)
    out.close()
    print("done", stats)


if __name__ == "__main__":
    main()
