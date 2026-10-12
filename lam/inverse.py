"""Algorithm 1 / Eq. 4: h_i = F_φ(x_i), a_i = G_θ(h_i).

F_φ is the (frozen, §IV-D) VAE encoder: its inverse F_φ^{-1} is the decoder of Eq. 6, so h_i is the
image's latent. G_θ is the LAM's image-side head: it predicts the structured action descriptor
A = {a_1..a_n} (the same schema the planner produces from text) from h_i. The predicted
descriptor then conditions generation exactly like a planned one (Eq. 2).

Characters are predicted in slots ordered left to right, matching the annotation order. Free-text
attribute fields are predicted as multi-label sets over the tag vocabulary seen in training.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

from lam.descriptor import ACTIONS, AGES, FRAMINGS, GENDERS, MAX_ACTIONS, MAX_CHARS, N_KEYPOINTS, TEXT_FIELDS, VIEWS
from lam.encoder import AttnPool, encoder_layers, fourier

GLOBAL_FIELDS = ["scene", "style", "props"]
VOCAB_SIZE = 256


def split_tags(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [t.strip() for t in str(v or "").split(",") if t.strip()]


def build_vocab(index: list[dict], k: int = VOCAB_SIZE) -> dict[str, list[str]]:
    """Most frequent tags per free-text field (per character) and per global field."""
    counts = {f: Counter() for f in TEXT_FIELDS + GLOBAL_FIELDS}
    for d in index:
        for c in d["characters"]:
            for f in TEXT_FIELDS:
                counts[f].update(split_tags(c.get(f)))
        for f in GLOBAL_FIELDS:
            counts[f].update(split_tags(d.get("global", {}).get(f)))
    return {f: [t for t, _ in cnt.most_common(k)] for f, cnt in counts.items()}


def targets(descs: list[dict], vocab: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    """Annotation dicts -> supervision tensors for G_θ."""
    B, C = len(descs), MAX_CHARS
    pos = {f: {t: i for i, t in enumerate(v)} for f, v in vocab.items()}
    out = {
        "present": torch.zeros(B, C), "gender": torch.zeros(B, C, dtype=torch.long),
        "age": torch.zeros(B, C, dtype=torch.long), "framing": torch.zeros(B, C, dtype=torch.long),
        "view": torch.zeros(B, C, dtype=torch.long), "actions": torch.zeros(B, C, len(ACTIONS) - 1),
        "bbox": torch.zeros(B, C, 4), "kps": torch.zeros(B, C, N_KEYPOINTS, 2), "kvis": torch.zeros(B, C, N_KEYPOINTS),
        **{f"t_{f}": torch.zeros(B, C, len(vocab[f])) for f in TEXT_FIELDS},
        **{f"g_{f}": torch.zeros(B, len(vocab[f])) for f in GLOBAL_FIELDS},
    }
    for b, d in enumerate(descs):
        for f in GLOBAL_FIELDS:
            for t in split_tags(d.get("global", {}).get(f)):
                if t in pos[f]:
                    out[f"g_{f}"][b, pos[f][t]] = 1
        for c, ch in enumerate(d["characters"][:C]):
            out["present"][b, c] = 1
            out["gender"][b, c] = GENDERS.index(ch["gender"]) if ch.get("gender") in GENDERS else 2
            out["age"][b, c] = AGES.index(ch["age"]) if ch.get("age") in AGES else 0
            out["framing"][b, c] = FRAMINGS.index(ch["framing"]) if ch.get("framing") in FRAMINGS else 0
            out["view"][b, c] = VIEWS.index(ch["view"]) if ch.get("view") in VIEWS else 0
            for act in ch.get("actions", []):
                if act in ACTIONS[1:]:
                    out["actions"][b, c, ACTIONS.index(act) - 1] = 1
            out["bbox"][b, c] = torch.tensor(ch.get("bbox") or [0, 0, 1, 1], dtype=torch.float32)
            if ch.get("keypoints"):
                k = torch.tensor(ch["keypoints"], dtype=torch.float32)
                out["kps"][b, c], out["kvis"][b, c] = k[:, :2].clamp(0, 1), (k[:, 2] > 0.3).float()
            for f in TEXT_FIELDS:
                for t in split_tags(ch.get(f)):
                    if t in pos[f]:
                        out[f"t_{f}"][b, c, pos[f][t]] = 1
    return out


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, stride, 1),
                                  nn.GroupNorm(8, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1))
        self.skip = nn.Conv2d(cin, cout, 1, stride) if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class ActionFromLatent(nn.Module):
    """G_θ: VAE latent h [B, 4, H/8, W/8] (any aspect bucket) -> descriptor logits."""

    def __init__(self, vocab: dict[str, list[str]], d: int = 512):
        super().__init__()
        self.vocab = vocab
        chans = [128, 256, 384, d]
        layers, cin = [nn.Conv2d(4, chans[0], 3, 1, 1)], chans[0]
        for c in chans:
            layers += [ResBlock(cin, c, 2), ResBlock(c, c, 1)]
            cin = c
        self.cnn = nn.Sequential(*layers)  # latent /16 -> 8x8 tokens at 1024px
        self.pos = nn.Linear(2 * 2 * 8, d)
        self.enc = encoder_layers(d, 4)
        self.slots = AttnPool(d, MAX_CHARS + 1, layers=2)  # 3 character slots + 1 global
        n_t = {f: len(vocab[f]) for f in TEXT_FIELDS}
        self.char_heads = nn.ModuleDict({
            "present": nn.Linear(d, 1), "gender": nn.Linear(d, len(GENDERS)), "age": nn.Linear(d, len(AGES)),
            "framing": nn.Linear(d, len(FRAMINGS)), "view": nn.Linear(d, len(VIEWS)),
            "actions": nn.Linear(d, len(ACTIONS) - 1), "bbox": nn.Linear(d, 4),
            "kps": nn.Linear(d, N_KEYPOINTS * 2), "kvis": nn.Linear(d, N_KEYPOINTS),
            **{f"t_{f}": nn.Linear(d, n) for f, n in n_t.items()},
        })
        self.global_heads = nn.ModuleDict({f"g_{f}": nn.Linear(d, len(vocab[f])) for f in GLOBAL_FIELDS})

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.cnn(h)
        B, D, hh, ww = x.shape
        gy, gx = torch.meshgrid((torch.arange(hh, device=x.device) + 0.5) / hh,
                                (torch.arange(ww, device=x.device) + 0.5) / ww, indexing="ij")
        tok = x.flatten(2).transpose(1, 2) + self.pos(fourier(torch.stack([gx, gy], -1).reshape(-1, 2).to(x.dtype)))
        s = self.slots(self.enc(tok))
        chars, glob = s[:, :MAX_CHARS], s[:, MAX_CHARS]
        out = {k: head(chars) for k, head in self.char_heads.items()}
        out["present"] = out["present"].squeeze(-1)
        out["kps"] = out["kps"].view(B, MAX_CHARS, N_KEYPOINTS, 2)
        out.update({k: head(glob) for k, head in self.global_heads.items()})
        return out

    @staticmethod
    def loss(o: dict[str, torch.Tensor], t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        pres = t["present"]
        m = pres.bool()
        parts = {"present": F.binary_cross_entropy_with_logits(o["present"], pres)}
        if m.any():
            for k in ("gender", "age", "framing", "view"):
                parts[k] = F.cross_entropy(o[k][m].float(), t[k][m])
            parts["actions"] = F.binary_cross_entropy_with_logits(o["actions"][m], t["actions"][m])
            parts["bbox"] = F.l1_loss(o["bbox"][m].sigmoid(), t["bbox"][m])
            vis = t["kvis"][m].unsqueeze(-1)
            parts["kps"] = (F.l1_loss(o["kps"][m].sigmoid(), t["kps"][m], reduction="none") * vis).sum() / vis.sum().clamp(min=1) / 2
            parts["kvis"] = F.binary_cross_entropy_with_logits(o["kvis"][m], t["kvis"][m])
            for f in TEXT_FIELDS:
                parts[f"t_{f}"] = F.binary_cross_entropy_with_logits(o[f"t_{f}"][m], t[f"t_{f}"][m]) * 10
        for f in GLOBAL_FIELDS:
            parts[f"g_{f}"] = F.binary_cross_entropy_with_logits(o[f"g_{f}"], t[f"g_{f}"]) * 10
        total = sum(parts.values())
        return total, {k: float(v) for k, v in parts.items()}

    @torch.no_grad()
    def to_descriptors(self, o: dict[str, torch.Tensor], thresh: float = 0.4) -> list[dict]:
        """Logits -> annotation-format descriptor dicts (the input format of the LAM Encoder)."""
        B = o["present"].shape[0]
        descs = []
        for b in range(B):
            g = {f: [self.vocab[f][i] for i in (o[f"g_{f}"][b].sigmoid() > thresh).nonzero().flatten().tolist()]
                 for f in GLOBAL_FIELDS}
            chars = []
            p = o["present"][b].sigmoid()
            keep = [c for c in range(MAX_CHARS) if p[c] > 0.5] or [int(p.argmax())]
            for c in keep:
                acts = o["actions"][b, c].sigmoid()
                top = [i for i in acts.argsort(descending=True).tolist() if acts[i] > thresh][:MAX_ACTIONS]
                kps = o["kps"][b, c].sigmoid()
                kv = o["kvis"][b, c].sigmoid()
                ch = {
                    "gender": GENDERS[int(o["gender"][b, c].argmax())], "age": AGES[int(o["age"][b, c].argmax())],
                    "framing": FRAMINGS[int(o["framing"][b, c].argmax())], "view": VIEWS[int(o["view"][b, c].argmax())],
                    "actions": [ACTIONS[i + 1] for i in sorted(top)],  # canonical order (posture first)
                    "bbox": [round(float(v), 4) for v in o["bbox"][b, c].sigmoid()],
                    "keypoints": [[round(float(kps[j, 0]), 4), round(float(kps[j, 1]), 4), round(float(kv[j]), 3)]
                                  for j in range(N_KEYPOINTS)],
                }
                for f in TEXT_FIELDS:
                    probs = o[f"t_{f}"][b, c].sigmoid()
                    ch[f] = ", ".join(self.vocab[f][i] for i in probs.argsort(descending=True).tolist()[:2] if probs[i] > thresh)
                chars.append(ch)
            chars.sort(key=lambda ch: (ch["bbox"][0] + ch["bbox"][2]) / 2)
            descs.append({"global": {"scene": ", ".join(g["scene"]), "style": ", ".join(g["style"]), "props": g["props"]},
                          "characters": chars})
        return descs
