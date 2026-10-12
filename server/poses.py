"""Pose library and OpenPose renderer.

The planner picks a named pose and a bounding box for each character. This module
turns that into an OpenPose skeleton image, which conditions the diffusion model
through ControlNet. That gives explicit, editable control over both *what* each
character is doing and *where* they are in the frame.

Keypoints follow the 18-point COCO/OpenPose layout used by OpenPose ControlNets:
0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist, 5 l_shoulder, 6 l_elbow,
7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee, 13 l_ankle,
14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear.
The character faces the viewer, so their right side is on the image's left.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

# Template coordinates: unit body height (y=0 top of head, y=1 feet), x centred on 0.5.
_HEAD = {
    0: (0.50, 0.100),
    14: (0.47, 0.085),
    15: (0.53, 0.085),
    16: (0.43, 0.100),
    17: (0.57, 0.100),
    1: (0.50, 0.200),
    2: (0.36, 0.210),
    5: (0.64, 0.210),
}
_STAND_LEGS = {
    8: (0.42, 0.52), 9: (0.42, 0.73), 10: (0.42, 0.94),
    11: (0.58, 0.52), 12: (0.58, 0.73), 13: (0.58, 0.94),
}
_ARMS_DOWN = {3: (0.32, 0.36), 4: (0.30, 0.50), 6: (0.68, 0.36), 7: (0.70, 0.50)}


def _pose(*parts: dict) -> dict:
    out: dict = {}
    for p in parts:
        out.update(p)
    return out


POSES: dict[str, dict[int, tuple[float, float]]] = {
    "standing": _pose(_HEAD, _ARMS_DOWN, _STAND_LEGS),
    "waving": _pose(_HEAD, _ARMS_DOWN, _STAND_LEGS, {6: (0.76, 0.15), 7: (0.73, 0.03)}),
    "arms_up": _pose(_HEAD, _STAND_LEGS, {3: (0.27, 0.12), 4: (0.25, 0.01), 6: (0.73, 0.12), 7: (0.75, 0.01)}),
    "hands_on_hips": _pose(_HEAD, _STAND_LEGS, {3: (0.24, 0.36), 4: (0.40, 0.50), 6: (0.76, 0.36), 7: (0.60, 0.50)}),
    "peace_sign": _pose(_HEAD, _STAND_LEGS, _ARMS_DOWN, {6: (0.73, 0.30), 7: (0.62, 0.13)}),
    "pointing": _pose(_HEAD, _STAND_LEGS, _ARMS_DOWN, {6: (0.80, 0.22), 7: (0.96, 0.22)}),
    "holding_object": _pose(_HEAD, _STAND_LEGS, {3: (0.30, 0.38), 4: (0.44, 0.33), 6: (0.70, 0.38), 7: (0.56, 0.33)}),
    "walking": _pose(
        _HEAD,
        {3: (0.33, 0.36), 4: (0.36, 0.49), 6: (0.67, 0.36), 7: (0.64, 0.49)},
        {8: (0.43, 0.52), 9: (0.44, 0.73), 10: (0.40, 0.94), 11: (0.57, 0.52), 12: (0.56, 0.72), 13: (0.62, 0.92)},
    ),
    "running": _pose(
        _HEAD,
        {3: (0.30, 0.34), 4: (0.40, 0.27), 6: (0.72, 0.34), 7: (0.75, 0.47)},
        {8: (0.42, 0.52), 9: (0.39, 0.66), 10: (0.44, 0.80), 11: (0.58, 0.52), 12: (0.60, 0.73), 13: (0.63, 0.94)},
    ),
    "jumping": _pose(
        _HEAD,
        {3: (0.27, 0.12), 4: (0.25, 0.01), 6: (0.73, 0.12), 7: (0.75, 0.01)},
        {8: (0.42, 0.52), 9: (0.34, 0.68), 10: (0.40, 0.86), 11: (0.58, 0.52), 12: (0.66, 0.68), 13: (0.60, 0.86)},
    ),
    "sitting": _pose(
        _HEAD,
        {3: (0.32, 0.38), 4: (0.42, 0.54), 6: (0.68, 0.38), 7: (0.58, 0.54)},
        {8: (0.42, 0.52), 9: (0.37, 0.59), 10: (0.38, 0.80), 11: (0.58, 0.52), 12: (0.63, 0.59), 13: (0.62, 0.80)},
    ),
    "kneeling": _pose(
        _HEAD,
        _ARMS_DOWN,
        {8: (0.42, 0.52), 9: (0.40, 0.76), 10: (0.40, 0.92), 11: (0.58, 0.52), 12: (0.60, 0.76), 13: (0.60, 0.92)},
    ),
}

POSE_NAMES = sorted(POSES)

# How much of the body (from the top of the head) is visible for each framing.
FRAMING_CROP = {"full_body": 1.0, "cowboy_shot": 0.75, "upper_body": 0.55, "portrait": 0.32}

_SWAP_LR = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
_SWAP_LR.update({v: k for k, v in list(_SWAP_LR.items())})

# Limb connections and colours used by the reference OpenPose renderer (controlnet_aux).
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
          (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
           (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
           (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def place_keypoints(pose: str, bbox: list[float], width: int, height: int,
                    framing: str = "full_body", mirror: bool = False) -> dict[int, tuple[float, float]]:
    """Scale a pose template into a normalized [x0, y0, x1, y1] box on a width x height canvas."""
    template = POSES.get(pose, POSES["standing"])
    if mirror:
        template = {_SWAP_LR.get(k, k): (1.0 - x, y) for k, (x, y) in template.items()}
    crop = FRAMING_CROP.get(framing, 1.0)
    x0, y0, x1, y1 = bbox
    box_h = (y1 - y0) * height
    scale = box_h / crop
    cx = (x0 + x1) / 2 * width
    top = y0 * height
    return {k: (cx + (x - 0.5) * scale, top + y * scale) for k, (x, y) in template.items()}


def render_openpose(characters: list[dict], width: int, height: int) -> Image.Image:
    """Draw every character's skeleton on a black canvas in OpenPose style."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    stick = max(3, round(min(width, height) / 160))
    for ch in characters:
        pts = place_keypoints(ch.get("pose", "standing"), ch["bbox"], width, height,
                              ch.get("framing", "full_body"), ch.get("mirror", False))
        for i, (a, b) in enumerate(_LIMBS):
            if a not in pts or b not in pts:
                continue
            _draw_limb(draw, pts[a], pts[b], stick, _COLORS[i])
        for k, (x, y) in pts.items():
            r = stick
            draw.ellipse([x - r, y - r, x + r, y + r], fill=_COLORS[k])
    return canvas


def _draw_limb(draw: ImageDraw.ImageDraw, p: tuple[float, float], q: tuple[float, float],
               width: int, color: tuple[int, int, int]) -> None:
    (x1, y1), (x2, y2) = p, q
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    angle = math.atan2(y2 - y1, x2 - x1)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    t = np.linspace(0, 2 * math.pi, 36)
    ex, ey = length / 2 * np.cos(t), width * np.sin(t)
    xs = mx + ex * math.cos(angle) - ey * math.sin(angle)
    ys = my + ex * math.sin(angle) + ey * math.cos(angle)
    draw.polygon(list(zip(xs, ys)), fill=(*(int(c * 0.6) for c in color), 255))
