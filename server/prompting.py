"""Turn a normalized spec into Animagine-style tag prompts, plus the safety gate."""

from __future__ import annotations

import re
from typing import Any

QUALITY = "masterpiece, high score, great score, absurdres"
NEGATIVE = (
    "nsfw, explicit, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
    "fewer digits, cropped, worst quality, low quality, low score, bad score, average score, "
    "signature, watermark, username, blurry"
)

POSE_TAGS = {
    "standing": "standing",
    "waving": "waving",
    "arms_up": "arms up",
    "hands_on_hips": "hands on hips",
    "peace_sign": "v",
    "pointing": "pointing",
    "holding_object": "holding",
    "walking": "walking",
    "running": "running",
    "jumping": "jumping",
    "sitting": "sitting",
    "kneeling": "kneeling",
}
FRAMING_TAGS = {"full_body": "full body", "cowboy_shot": "cowboy shot", "upper_body": "upper body", "portrait": "portrait"}

# Demo is SFW-only. Sexual content is refused outright rather than filtered, which also
# rules out sexualised depictions of young-looking characters.
_BLOCKED = re.compile(
    r"\b(nsfw|nude|nudity|naked|explicit|sex|sexual|porn|hentai|lewd|ecchi|topless|"
    r"underwear|lingerie|erotic|fetish|genitals?|nipples?)\b",
    re.IGNORECASE,
)


def check_safe(*texts: str) -> str | None:
    """Return a reason string if any text is disallowed, else None."""
    for t in texts:
        if t and _BLOCKED.search(t):
            return "This demo only generates safe-for-work images."
    return None


def spec_texts(spec: dict[str, Any]) -> list[str]:
    out = [spec.get("scene", ""), spec.get("style", ""), *spec.get("props", [])]
    for c in spec.get("characters", []):
        out += [str(v) for v in c.values() if isinstance(v, str)]
    return out


def _count_tags(chars: list[dict[str, Any]]) -> list[str]:
    # Danbooru counts adults and children alike as 1girl/1boy.
    girls = sum(c["gender"] == "female" for c in chars)
    boys = sum(c["gender"] == "male" for c in chars)
    others = len(chars) - girls - boys
    tags = []
    if girls:
        tags.append("1girl" if girls == 1 else f"{girls}girls")
    if boys:
        tags.append("1boy" if boys == 1 else f"{boys}boys")
    if others:
        tags.append("1other" if others == 1 else f"{others}others")
    if len(chars) == 1:
        tags.append("solo")
    elif girls > 1:
        tags.append("multiple girls")
    elif boys > 1:
        tags.append("multiple boys")
    return tags


AGE_TAGS = {
    ("elderly", "male"): ["old man", "old", "wrinkled skin"],
    ("elderly", "female"): ["old woman", "old", "wrinkled skin"],
    ("elderly", "other"): ["old", "wrinkled skin"],
    ("adult", "male"): ["mature male"],
    ("adult", "female"): ["mature female"],
    ("child", "male"): ["child"],
    ("child", "female"): ["child"],
    ("child", "other"): ["child"],
}


def character_tags(c: dict[str, Any]) -> list[str]:
    tags = list(AGE_TAGS.get((c["age"], c["gender"]), []))
    if c["hair_color"]:
        tags.append(c["hair_color"] if "hair" in c["hair_color"] else f"{c['hair_color']} hair")
    if c["hair_style"]:
        tags.append(c["hair_style"])
    if c["eye_color"]:
        tags.append(c["eye_color"] if "eye" in c["eye_color"] else f"{c['eye_color']} eyes")
    for f in ("outfit", "expression", "action", "extra"):
        if c[f]:
            tags.append(c[f])
    tags.append(POSE_TAGS[c["pose"]])
    return tags


def _gender_tags(chars: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Extra positive/negative tags that pin down gender and age.

    Anime models lean strongly toward young girls: attributes like long hair, silver hair or
    school uniform pull a "1boy" prompt toward a girl. Danbooru's "male focus" tag plus a
    negative on the girl count tags counteracts that. Mixed casts get the explicit counts only.
    """
    girls = sum(c["gender"] == "female" for c in chars)
    boys = sum(c["gender"] == "male" for c in chars)
    pos: list[str] = []
    neg: list[str] = []
    if boys and not girls:
        pos, neg = ["male focus"], ["1girl", "multiple girls", "female focus"]
    elif girls and not boys:
        neg = ["1boy", "male focus"]
    if all(c["age"] == "elderly" for c in chars):
        neg += ["young", "child", "teenage"]
    return pos, neg


def build_prompt(spec: dict[str, Any]) -> tuple[str, str]:
    chars = spec["characters"]
    gender_pos, gender_neg = _gender_tags(chars)
    tags = _count_tags(chars) + gender_pos + ["safe"]
    for c in chars:
        tags += character_tags(c)
    # Use the widest framing among characters for the global framing tag.
    widest = max(chars, key=lambda c: {"portrait": 0, "upper_body": 1, "cowboy_shot": 2, "full_body": 3}[c["framing"]])
    tags.append(FRAMING_TAGS[widest["framing"]])
    tags += spec["props"]
    if spec["scene"]:
        tags.append(spec["scene"])
    if spec["style"]:
        tags.append(spec["style"])
    tags.append(QUALITY)
    return _dedupe(tags), ", ".join(gender_neg + [NEGATIVE])


def baseline_prompt(user_prompt: str) -> tuple[str, str]:
    """Prompt-only baseline: the raw user text with the same quality tags, no planning."""
    return f"{user_prompt.strip()}, safe, {QUALITY}", NEGATIVE


def _dedupe(tags: list[str]) -> str:
    seen, out = set(), []
    for t in tags:
        t = t.strip().strip(",")
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)
