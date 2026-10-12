"""Training data: pre-computed samples + descriptors, with the paper's progressive curriculum."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from lam.descriptor import TEXT_FIELDS


def load_index(path: str | Path, split: str) -> list[dict]:
    return [d for d in (json.loads(line) for line in open(path)) if d["split"] == split]


def complexity(d: dict) -> float:
    """Curriculum score (§IV-F stage 4): basic poses -> multi-action sequences -> fine-grained style."""
    chars = d["characters"]
    n_actions = sum(len(c.get("actions", [])) for c in chars)
    n_attrs = sum(bool(c.get(f)) for c in chars for f in TEXT_FIELDS)
    return 1.0 * len(chars) + 0.5 * n_actions + 0.15 * n_attrs


class SampleDataset(Dataset):
    def __init__(self, index: list[dict], samples_dir: str | Path):
        self.items = index
        self.dir = Path(samples_dir)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        d = self.items[i]
        z = np.load(self.dir / d["file"])
        gt = np.asarray(Image.open(self.dir / d["file"].replace(".npz", ".gt.jpg")).convert("RGB"))
        return {
            "desc": d,
            "latents": torch.from_numpy(z["latents"].astype(np.float32)),
            "gt": torch.from_numpy(gt.copy()).permute(2, 0, 1).float() / 255.0,
            "clip": torch.from_numpy(z["clip"].astype(np.float32)),
            "time_ids": torch.from_numpy(z["time_ids"].astype(np.float32)),
        }


def collate_samples(items: list[dict]) -> dict:
    return {
        "descs": [x["desc"] for x in items],
        "latents": torch.stack([x["latents"] for x in items]),
        "gt": torch.stack([x["gt"] for x in items]),
        "clip": torch.stack([x["clip"] for x in items]),
        "time_ids": torch.stack([x["time_ids"] for x in items]),
    }


class CurriculumSampler(Sampler[list[int]]):
    """Yields single-bucket batches. The eligible pool grows with training progress:
    the simplest 40% of samples first, then 80%, then everything (30/40/30% of steps)."""

    def __init__(self, index: list[dict], total_steps: int, batch_size: int, start_step: int = 0, seed: int = 0):
        self.order = sorted(range(len(index)), key=lambda i: complexity(index[i]))
        self.bucket = [tuple(d["bucket"]) for d in index]
        self.total, self.bs, self.step, self.rng = total_steps, batch_size, start_step, random.Random(seed + start_step)

    def pool_fraction(self, step: int) -> float:
        p = step / max(1, self.total)
        return 0.4 if p < 0.3 else 0.8 if p < 0.7 else 1.0

    def __iter__(self):
        while self.step < self.total:
            k = max(self.bs, int(len(self.order) * self.pool_fraction(self.step)))
            first = self.order[self.rng.randrange(k)]
            same = [first]
            while len(same) < self.bs:  # fill the batch from the same aspect bucket
                j = self.order[self.rng.randrange(k)]
                if self.bucket[j] == self.bucket[first]:
                    same.append(j)
            self.step += 1
            yield same

    def __len__(self) -> int:
        return self.total - self.step
