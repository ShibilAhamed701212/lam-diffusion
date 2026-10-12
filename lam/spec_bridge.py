"""App spec (what the planner and the editor produce) -> LAM action descriptor.

No prompt tags are built: fields map one-to-one onto the descriptor, and keypoints come from
the data-mined pose library placed in each character's box.
"""

from __future__ import annotations

from typing import Any

from lam.descriptor import ACTIONS
from lam.pose_library import keypoints_for

# The app's pose names -> the dataset's action vocabulary (same concepts, different labels).
POSE_TO_ACTION = {
    "standing": "standing", "waving": "waving", "arms_up": "arms_up", "hands_on_hips": "hands_on_hips",
    "peace_sign": "v", "pointing": "pointing", "holding_object": "holding", "walking": "walking",
    "running": "running", "jumping": "jumping", "sitting": "sitting", "kneeling": "kneeling",
}


def spec_to_descriptor(spec: dict[str, Any], pose_lib: dict) -> dict[str, Any]:
    chars = []
    for i, c in enumerate(spec.get("characters", [])):
        pose_action = POSE_TO_ACTION.get(c.get("pose", "standing"), "standing")
        actions = [pose_action]
        for extra in (c.get("actions") or []):
            if extra in ACTIONS and extra not in actions:
                actions.append(extra)
        bbox = c.get("bbox") or [0.2, 0.05, 0.8, 0.97]
        framing = c.get("framing", "full_body")
        kps = c.get("keypoints") or keypoints_for(pose_lib, pose_action, framing, bbox, variant=i)
        if c.get("mirror"):
            kps = _mirror(kps, bbox)
        chars.append({
            "gender": c.get("gender", "other"), "age": c.get("age", "unspecified"),
            "hair_color": c.get("hair_color", ""), "hair_style": c.get("hair_style", ""),
            "eye_color": c.get("eye_color", ""), "outfit": c.get("outfit", ""),
            "expression": c.get("expression", ""), "extra": ", ".join(x for x in (c.get("extra", ""), c.get("action", "")) if x),
            "actions": actions[:4], "framing": framing, "view": c.get("view", "front"),
            "bbox": bbox, "keypoints": kps,
        })
    return {
        "global": {"scene": spec.get("scene", ""), "style": spec.get("style", ""), "props": spec.get("props", [])},
        "characters": chars,
    }


_SWAP = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
_SWAP.update({v: k for k, v in list(_SWAP.items())})


def _mirror(kps: list[list[float]], bbox: list[float]) -> list[list[float]]:
    x0, _, x1, _ = bbox
    out = [None] * len(kps)
    for i, (x, y, s) in enumerate(kps):
        out[_SWAP.get(i, i)] = [round(x0 + x1 - x, 4), y, s]
    return out
