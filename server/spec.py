"""The character spec: the structured "action representation" the planner produces.

Everything downstream (pose image, prompt tags, UI editor) reads this one shape, so
`normalize_spec` is strict: it fills defaults, clamps values and computes layout
boxes, whatever the planner (or a user edit) sent in.
"""

from __future__ import annotations

from typing import Any

from poses import FRAMING_CROP, POSE_NAMES

MAX_CHARACTERS = 3
ASPECTS = {"portrait": (832, 1216), "square": (1024, 1024), "landscape": (1216, 832)}
POSITIONS = ("left", "center", "right")
GENDERS = ("female", "male", "other")
AGES = ("unspecified", "child", "teen", "young_adult", "adult", "elderly")

# Planners (and older specs) use everyday words; map them onto the canonical values.
_GENDER_ALIASES = {
    "girl": "female", "woman": "female", "lady": "female", "f": "female",
    "boy": "male", "man": "male", "guy": "male", "m": "male",
}
_AGE_ALIASES = {"old": "elderly", "senior": "elderly", "young adult": "young_adult", "kid": "child", "teenager": "teen"}

CHARACTER_FIELDS = ("hair_color", "hair_style", "eye_color", "outfit", "expression", "action", "extra")


def default_character() -> dict[str, Any]:
    return {
        "gender": "female",
        "age": "unspecified",
        "hair_color": "",
        "hair_style": "",
        "eye_color": "",
        "outfit": "",
        "expression": "",
        "action": "",
        "extra": "",
        "pose": "standing",
        "framing": "full_body",
        "position": "center",
        "mirror": False,
        "bbox": None,
    }


def normalize_spec(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    chars_in = raw.get("characters") or [{}]
    if not isinstance(chars_in, list):
        chars_in = [chars_in]
    characters = [_normalize_character(c) for c in chars_in[:MAX_CHARACTERS] if isinstance(c, dict)]
    if not characters:
        characters = [default_character()]

    aspect = raw.get("aspect")
    if aspect not in ASPECTS:
        aspect = "landscape" if len(characters) > 1 else "portrait"

    props = raw.get("props") or []
    if isinstance(props, str):
        props = [p.strip() for p in props.split(",")]
    spec = {
        "scene": _text(raw.get("scene")),
        "style": _text(raw.get("style")),
        "props": [_text(p) for p in props if _text(p)][:8],
        "aspect": aspect,
        "characters": characters,
    }
    _assign_layout(spec)
    return spec


def _normalize_character(c: dict[str, Any]) -> dict[str, Any]:
    out = default_character()
    for f in CHARACTER_FIELDS:
        out[f] = _text(c.get(f))
    gender = str(c.get("gender") or "").strip().lower()
    gender = _GENDER_ALIASES.get(gender, gender)
    out["gender"] = gender if gender in GENDERS else "other"
    age = str(c.get("age") or "").strip().lower()
    age = _AGE_ALIASES.get(age, age)
    out["age"] = age if age in AGES else "unspecified"
    out["pose"] = c.get("pose") if c.get("pose") in POSE_NAMES else "standing"
    out["framing"] = c.get("framing") if c.get("framing") in FRAMING_CROP else "full_body"
    out["position"] = c.get("position") if c.get("position") in POSITIONS else "center"
    out["mirror"] = bool(c.get("mirror", False))
    bbox = c.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in bbox)
            if x1 - x0 > 0.05 and y1 - y0 > 0.05:
                out["bbox"] = [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]
        except (TypeError, ValueError):
            pass
    return out


def _assign_layout(spec: dict[str, Any]) -> None:
    """Give every character without a user-set bbox a box from its position slot."""
    chars = spec["characters"]
    n = len(chars)
    w, h = ASPECTS[spec["aspect"]]
    # Characters that share a position get spread across that slot's width.
    slots = {"left": 0.0, "center": 1.0, "right": 2.0}
    order = sorted(range(n), key=lambda i: (slots[chars[i]["position"]], i))
    for rank, i in enumerate(order):
        ch = chars[i]
        if ch["bbox"] is not None:
            continue
        if n == 1:
            cx = {"left": 0.33, "center": 0.5, "right": 0.67}[ch["position"]]
        else:
            cx = (rank + 0.5) / n
        top, bottom = 0.05, 0.97
        if ch["framing"] in ("upper_body", "portrait"):
            bottom = 1.0
        box_h = bottom - top
        # A full body is roughly 0.45 as wide as it is tall; keep that in pixel space.
        box_w = min(0.95 / n, box_h * h / w * 0.45 / FRAMING_CROP[ch["framing"]])
        x0 = max(0.0, cx - box_w / 2)
        x1 = min(1.0, cx + box_w / 2)
        ch["bbox"] = [round(x0, 3), round(top, 3), round(x1, 3), round(bottom, 3)]


def _text(v: Any) -> str:
    if v is None:
        return ""
    return " ".join(str(v).split())[:120]
