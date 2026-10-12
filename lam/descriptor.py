"""Structured action descriptors (paper Eq. 1: A = {a_1, ..., a_n}) as model inputs.

One descriptor per character plus a global one (scene / style / props). Categorical fields
(gender, age, framing, view, sub-actions) become ids for learned embeddings; free-text style
attributes stay text so the transformer backbone encodes them; pose stays numeric keypoints.
Nothing here translates attributes into prompt tags.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from lam.data.tag_vocab import ACTION_ORDER

MAX_CHARS = 3
MAX_ACTIONS = 4
N_KEYPOINTS = 18

GENDERS = ["female", "male", "other"]
AGES = ["unspecified", "child", "teen", "young_adult", "adult", "elderly"]
FRAMINGS = ["unspecified", "full_body", "cowboy_shot", "upper_body", "portrait"]
VIEWS = ["front", "side", "behind", "above", "below"]
ACTIONS = ["<none>"] + ACTION_ORDER
TEXT_FIELDS = ["hair_color", "hair_style", "eye_color", "outfit", "expression", "extra"]

# Human-readable names used only to *initialise* the learned label embeddings from the
# backbone (label-name initialisation); the embeddings are trained afterwards.
LABEL_NAMES = {
    "gender": ["girl, female", "boy, male", "person"],
    "age": ["", "child", "teenager", "young adult", "mature adult", "elderly old person"],
    "framing": ["", "full body", "cowboy shot", "upper body", "portrait close-up"],
    "view": ["from front", "from side", "from behind", "from above", "from below"],
    "action": [""] + [a.replace("_", " ") for a in ACTION_ORDER],
}


def _idx(table: list[str], value: Any, default: int = 0) -> int:
    try:
        return table.index(value)
    except ValueError:
        return default


@dataclass
class DescriptorBatch:
    """Batched descriptor tensors. C = MAX_CHARS, T = MAX_ACTIONS."""

    char_mask: torch.Tensor    # [B, C] bool, character present
    gender: torch.Tensor       # [B, C] long
    age: torch.Tensor          # [B, C] long
    framing: torch.Tensor      # [B, C] long
    view: torch.Tensor         # [B, C] long
    actions: torch.Tensor      # [B, C, T] long (0 = none)
    keypoints: torch.Tensor    # [B, C, 18, 3] (x, y, confidence) normalized to the image
    bbox: torch.Tensor         # [B, C, 4] normalized x0, y0, x1, y1
    texts: list[list[list[str]]]  # [B][C][len(TEXT_FIELDS)] free-text attributes
    global_text: list[str]     # [B] scene / style / props
    drop_all: torch.Tensor     # [B] bool, classifier-free-guidance drop (§IV-F stage 5)

    def to(self, device: torch.device | str) -> "DescriptorBatch":
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def global_text(g: dict[str, Any]) -> str:
    parts = [g.get("scene", ""), g.get("style", ""), ", ".join(g.get("props", []))]
    return ", ".join(p for p in parts if p)


def collate(descs: list[dict[str, Any]], *, train: bool = False, p_drop_all: float = 0.1,
            p_drop_field: float = 0.1, rng: random.Random | None = None) -> DescriptorBatch:
    """Descriptors (annotation-format dicts) -> DescriptorBatch.

    In training, applies the paper's CFG dropout (drop all conditioning) plus per-field
    dropout so partially specified descriptors still work at inference.
    """
    rng = rng or random.Random()
    B, C, T = len(descs), MAX_CHARS, MAX_ACTIONS
    out = DescriptorBatch(
        char_mask=torch.zeros(B, C, dtype=torch.bool),
        gender=torch.zeros(B, C, dtype=torch.long),
        age=torch.zeros(B, C, dtype=torch.long),
        framing=torch.zeros(B, C, dtype=torch.long),
        view=torch.zeros(B, C, dtype=torch.long),
        actions=torch.zeros(B, C, T, dtype=torch.long),
        keypoints=torch.zeros(B, C, N_KEYPOINTS, 3),
        bbox=torch.zeros(B, C, 4),
        texts=[[[""] * len(TEXT_FIELDS) for _ in range(C)] for _ in range(B)],
        global_text=[""] * B,
        drop_all=torch.zeros(B, dtype=torch.bool),
    )
    drop = lambda: train and rng.random() < p_drop_field  # noqa: E731
    for b, d in enumerate(descs):
        out.drop_all[b] = train and rng.random() < p_drop_all
        out.global_text[b] = "" if drop() else global_text(d.get("global", {}))
        for c, ch in enumerate(d.get("characters", [])[:C]):
            out.char_mask[b, c] = True
            out.gender[b, c] = _idx(GENDERS, ch.get("gender"), 2)
            out.age[b, c] = 0 if drop() else _idx(AGES, ch.get("age"))
            out.framing[b, c] = 0 if drop() else _idx(FRAMINGS, ch.get("framing"))
            out.view[b, c] = _idx(VIEWS, ch.get("view"))
            acts = [] if drop() else ch.get("actions", [])
            for t, act in enumerate(acts[:T]):
                out.actions[b, c, t] = _idx(ACTIONS, act)
            kps = ch.get("keypoints")
            if kps and not drop():
                out.keypoints[b, c] = torch.tensor(kps, dtype=torch.float32)
            out.bbox[b, c] = torch.tensor(ch.get("bbox") or [0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            for f, field in enumerate(TEXT_FIELDS):
                out.texts[b][c][f] = "" if drop() else str(ch.get(field, "") or "")
    return out


SWAP_FIELDS = {
    "eye_color": ["red eyes", "blue eyes", "green eyes", "purple eyes", "yellow eyes", "pink eyes", "brown eyes"],
    "hair_color": ["red hair", "blue hair", "green hair", "purple hair", "blonde hair", "pink hair", "black hair",
                   "white hair"],
}


def swap_text_field(d: DescriptorBatch, rng: random.Random) -> tuple[DescriptorBatch, int]:
    """Counterfactual descriptor: replace ONE colour field (eye or hair) of every character that has it
    with a different colour. Returns (batch, index of the swapped field in TEXT_FIELDS).

    Used by the swap-consistency loss (a deviation from the paper, see losses.swap_consistency): tokens
    of all the OTHER fields must not change, which trains the fusion not to leak one attribute into
    another (eye colour -> hair colour). Tensors are shared with `d`; only the text lists are copied."""
    import copy

    name = rng.choice(list(SWAP_FIELDS))
    f = TEXT_FIELDS.index(name)
    texts = copy.deepcopy(d.texts)
    for b in range(len(texts)):
        for c in range(len(texts[b])):
            cur = texts[b][c][f]
            if cur:
                texts[b][c][f] = rng.choice([v for v in SWAP_FIELDS[name] if v != cur])
    out = copy.copy(d)
    out.texts = texts
    return out, f
