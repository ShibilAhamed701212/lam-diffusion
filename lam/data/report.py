"""M1 report: dataset statistics + a grid of random annotated triplets for visual inspection."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170)]
BOX = [(255, 64, 160), (64, 160, 255), (255, 200, 0)]


def tile(ann: dict, images: Path, height: int = 360, panel: int = 260) -> Image.Image:
    img = Image.open(images / f"{ann['id']}.jpg").convert("RGB")
    img = img.resize((round(img.width * height / img.height), height))
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for ci, c in enumerate(ann["characters"]):
        x0, y0, x1, y1 = c["bbox"]
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=BOX[ci % 3], width=3)
        k = c["keypoints"]
        for li, (a, b) in enumerate(LIMBS):
            if k[a][2] > 0.3 and k[b][2] > 0.3:
                draw.line([k[a][0] * w, k[a][1] * h, k[b][0] * w, k[b][1] * h], fill=COLORS[li], width=3)
    out = Image.new("RGB", (w + panel, height), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    y = 4
    d.text((w + 6, y), f"#{ann['id']} {ann['comp']}", fill="black", font=font)
    y += 14
    g = ann["global"]
    for line in (f"scene: {g['scene'][:40]}", f"props: {', '.join(g['props'])[:40]}"):
        d.text((w + 6, y), line, fill=(90, 90, 90), font=font)
        y += 12
    for ci, c in enumerate(ann["characters"]):
        y += 4
        d.text((w + 6, y), f"char {ci + 1}: {c['gender']}, {c['age']}, {c['framing']}", fill=BOX[ci % 3], font=font)
        y += 12
        for f in ("hair_color", "hair_style", "eye_color", "outfit", "expression", "extra"):
            if c.get(f):
                d.text((w + 12, y), f"{f}: {c[f]}"[:42], fill="black", font=font)
                y += 11
        if c.get("actions"):
            d.text((w + 12, y), f"actions: {' > '.join(c['actions'])}"[:42], fill=(0, 110, 0), font=font)
            y += 11
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="/content/data/annotations.jsonl")
    ap.add_argument("--images", default="/content/data/images")
    ap.add_argument("--out", default="/content/data/report")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    anns = [json.loads(line) for line in open(a.annotations)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chars = [c for x in anns for c in x["characters"]]
    stats = {
        "images": len(anns),
        "characters": len(chars),
        "composition": Counter(x["comp"] for x in anns).most_common(),
        "gender": Counter(c["gender"] for c in chars).most_common(),
        "age": Counter(c["age"] for c in chars).most_common(),
        "framing": Counter(c["framing"] for c in chars).most_common(),
        "first_action": Counter((c["actions"] or ["<none>"])[0] for c in chars).most_common(20),
        "field_coverage": {f: round(sum(bool(c.get(f)) for c in chars) / max(1, len(chars)), 3)
                           for f in ("hair_color", "hair_style", "eye_color", "outfit", "expression", "extra")},
        "with_actions": round(sum(bool(c["actions"]) for c in chars) / max(1, len(chars)), 3),
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

    pick = random.Random(1).sample(anns, min(a.n, len(anns)))
    tiles = [tile(x, Path(a.images)) for x in pick]
    cols = 3
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    width = max(sum(t.width for t in r) for r in rows)
    sheet = Image.new("RGB", (width, sum(max(t.height for t in r) for r in rows)), (235, 235, 235))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y))
            x += t.width
        y += max(t.height for t in r)
    sheet.save(out / "samples.jpg", quality=88)
    print("wrote", out / "samples.jpg")


if __name__ == "__main__":
    main()
