"""Action-text corpus for §IV-B ("pre-trained on a large-scale action-text corpus").

HumanML3D (Guo et al., CVPR 2022) has ~44k English descriptions of ~14.6k human motions, about
three independent captions per motion, many of them ordered ("a person walks forward, then
turns around and sits down"). Only the caption text is used (texts.zip from the public repo);
the AMASS motion data is not needed.

Each caption is also split into ordered sub-action phrases and mapped onto the LAM's action
vocabulary. Like the Danbooru tag mapping in tag_vocab.py this is data annotation for training
the Action Dynamics / Temporal Attention modules; it is never used at generation time.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://github.com/EricGuo5513/HumanML3D/raw/main/HumanML3D"

# phrase pattern -> action label (labels from lam.data.tag_vocab.ACTION_ORDER). First match wins
# within a phrase, so specific patterns come before general ones.
ACTION_PATTERNS = [
    (r"\b(raises?|lifts?|puts?|holds?)\b.*\b(both|two)\b.*\b(arms|hands)\b.*\b(up|above|over)\b|\barms (up|above)\b|\bhands up\b", "arms_up"),
    (r"\b(raises?|lifts?)\b.*\b(left|right|one)\b.*\b(arm|hand)\b", "arm_up"),
    (r"\bhands? on (his|her|their|the)? ?hips?\b", "hands_on_hips"),
    (r"\b(cross(es|ed)?|folds?) (his |her |their )?arms\b", "crossed_arms"),
    (r"\b(spreads?|stretch(es)? out|opens?) (his |her |their )?arms\b|\bt[- ]pose\b", "spread_arms"),
    (r"\bwav(e|es|ing)\b", "waving"),
    (r"\bsalut(e|es|ing)\b", "salute"),
    (r"\bclap(s|ping)?\b", "hands_together"),
    (r"\bpoint(s|ing)?\b", "pointing"),
    (r"\breach(es|ing)?\b", "reaching"),
    (r"\bstretch(es|ing)?\b", "stretching"),
    (r"\bkick(s|ing)?\b", "kicking"),
    (r"\bpunch(es|ing)?\b|\bbox(es|ing)\b|\bjabs?\b", "punching"),
    (r"\bswing(s|ing)?\b", "swinging"),
    (r"\b(fight(s|ing)?|spar(s|ring)?)\b", "fighting"),
    (r"\bdanc(e|es|ing)\b", "dancing"),
    (r"\b(jump(s|ing)?|hop(s|ping)?|leap(s|ing)?|skip(s|ping)?)\b", "jumping"),
    (r"\b(run(s|ning)?|jog(s|ging)?|sprint(s|ing)?)\b", "running"),
    (r"\b(walk(s|ing)?|steps? (forward|back|backward|to)|side[- ]?step(s|ping)?|takes? (a |one |two |three )?steps?|pac(e|es|ing)|strolls?|march(es|ing)?)\b", "walking"),
    (r"\b(sits?|sitting|seated)\b", "sitting"),
    (r"\b(lies|lying|lays? down|on (his|her|their) back)\b", "lying"),
    (r"\bkneel(s|ing)?|knelt\b", "kneeling"),
    (r"\bsquat(s|ting)?\b", "squatting"),
    (r"\bcrouch(es|ing)?\b|\bbends? down\b", "crouching"),
    (r"\blean(s|ing)? (forward|over)\b|\bbends? (forward|over)\b|\bbows?\b", "leaning_forward"),
    (r"\blean(s|ing)? back(ward)?s?\b", "leaning_back"),
    (r"\bfall(s|ing)?\b|\bfell\b|\bstumbl(e|es|ing)\b", "falling"),
    (r"\bcrawl(s|ing)?\b", "on_stomach"),
    (r"\b(eat(s|ing)?)\b", "eating"),
    (r"\b(drink(s|ing)?)\b", "drinking"),
    (r"\b(carr(y|ies|ying)|lift(s|ing)?|pick(s|ing)? up)\b", "carrying"),
    (r"\bhug(s|ging)?\b", "hug"),
    (r"\b(turns? around|looks? (back|behind))\b", "looking_back"),
    (r"\blooks? up\b", "looking_up"),
    (r"\blooks? down\b", "looking_down"),
    (r"\bplay(s|ing)? (the )?(guitar|violin|piano|drums|flute)\b", "playing_instrument"),
    (r"\bsing(s|ing)?\b", "singing"),
    (r"\btalk(s|ing)?\b", "talking"),
    (r"\bthrow(s|ing)?\b", "swinging"),
    (r"\b(stands?|standing|gets? up|rises?)\b", "standing"),
]
_COMPILED = [(re.compile(p), lab) for p, lab in ACTION_PATTERNS]
# Sequence connectives: the caption order of the parts is the temporal order.
_SPLIT = re.compile(r"\b(?:and then|then|after that|afterwards|followed by|and finally|finally|before|next)\b|[,;.]")


def sub_actions(caption: str, max_steps: int = 4) -> list[str]:
    """'a person walks forward, then sits down and waves' -> ['walking', 'sitting', 'waving']"""
    out: list[str] = []
    for part in _SPLIT.split(caption.lower()):
        # "and" inside a part ("sits down and waves") is also sequential in these captions
        for phrase in re.split(r"\band\b", part):
            for pat, lab in _COMPILED:
                if pat.search(phrase):
                    if not out or out[-1] != lab:
                        out.append(lab)
                    break
    return out[:max_steps]


def download(dest: Path) -> Path:
    """Fetch texts.zip and the split lists once; returns the folder with one caption file per motion."""
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "texts").exists():
        data = urllib.request.urlopen(f"{REPO}/texts.zip", timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    for split in ("train", "val", "test"):
        f = dest / f"{split}.txt"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(f"{REPO}/{split}.txt", timeout=60).read())
    return dest


def load(dest: Path, split: str) -> list[dict]:
    """-> [{'id', 'captions': [...], 'sequences': [[action, ...] per caption]}]"""
    ids = [line.strip() for line in open(dest / f"{split}.txt") if line.strip()]
    out = []
    for mid in ids:
        f = dest / "texts" / f"{mid}.txt"
        if not f.exists():
            continue
        caps = [line.split("#")[0].strip() for line in open(f, encoding="utf-8") if line.strip()]
        caps = [c for c in caps if c]
        if caps:
            out.append({"id": mid, "captions": caps, "sequences": [sub_actions(c) for c in caps]})
    return out


if __name__ == "__main__":
    import sys

    root = download(Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data/humanml3d"))
    tr = load(root, "train")
    seqs = [s for m in tr for s in m["sequences"]]
    print(json.dumps({"motions": len(tr), "captions": len(seqs),
                      "with_action": sum(bool(s) for s in seqs) / max(1, len(seqs)),
                      "multi_step": sum(len(s) >= 2 for s in seqs) / max(1, len(seqs)),
                      "example": [(m["captions"][0], m["sequences"][0]) for m in tr[:5]]}, indent=1))
