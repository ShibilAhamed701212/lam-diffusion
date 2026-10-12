"""Data-mined pose library: typical skeletons per action, from the annotated dataset.

Replaces the hand-drawn templates in server/poses.py for the learned path: when the planner
asks for "running", the keypoints come from k-means centres of real running skeletons.
Skeletons are stored in a box-relative frame (x, y in [0, 1] of the character's bbox).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_KP = 18
MIN_SAMPLES = 40


def to_box_frame(kps: list[list[float]], bbox: list[float]) -> np.ndarray | None:
    x0, y0, x1, y1 = bbox
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    k = np.array(kps, dtype=np.float32)
    vis = k[:, 2] > 0.3
    if vis.sum() < 12:
        return None
    out = np.zeros((N_KP, 3), dtype=np.float32)
    out[:, 0] = (k[:, 0] - x0) / w
    out[:, 1] = (k[:, 1] - y0) / h
    out[:, 2] = vis.astype(np.float32)
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = ((x[:, None] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        c = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c


def build(index_path: str, out_path: str, k: int = 4) -> dict:
    by_action: dict[str, list[np.ndarray]] = defaultdict(list)
    for line in open(index_path):
        d = json.loads(line)
        if d.get("split") != "train":
            continue
        for c in d["characters"]:
            s = to_box_frame(c["keypoints"], c["bbox"])
            if s is None:
                continue
            key = (c["actions"] or ["standing"])[0]
            by_action[f"{key}|{c['framing']}"].append(s)
            by_action[key].append(s)
    lib = {}
    for key, skels in by_action.items():
        if len(skels) < MIN_SAMPLES:
            continue
        arr = np.stack(skels)
        full = arr[arr[:, :, 2].min(1) > 0]  # fully visible skeletons give clean centres
        if len(full) < MIN_SAMPLES // 2:
            continue
        flat = full[:, :, :2].reshape(len(full), -1)
        centres = kmeans(flat, min(k, len(full) // 10 or 1)).reshape(-1, N_KP, 2)
        lib[key] = {"count": len(skels), "poses": np.round(centres, 4).tolist()}
    Path(out_path).write_text(json.dumps(lib))
    return lib


def keypoints_for(lib: dict, action: str, framing: str, bbox: list[float], variant: int = 0) -> list[list[float]]:
    """Library skeleton placed into an image-space bbox; falls back to 'standing'."""
    entry = lib.get(f"{action}|{framing}") or lib.get(action) or lib.get(f"standing|{framing}") or lib["standing"]
    pose = entry["poses"][variant % len(entry["poses"])]
    x0, y0, x1, y1 = bbox
    return [[round(x0 + px * (x1 - x0), 4), round(y0 + py * (y1 - y0), 4), 1.0] for px, py in pose]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--out", default="/content/data/pose_library.json")
    a = ap.parse_args()
    lib = build(a.index, a.out)
    print(len(lib), "pose entries;", sorted(((v["count"], k) for k, v in lib.items() if "|" not in k), reverse=True)[:15])
