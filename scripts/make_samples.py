"""Generate sample images from the running LAM Studio server (planner -> pose -> Animagine + ControlNet,
with the prompt-only baseline on the same seed) and save them to lam-studio/output/samples/.

    python3 scripts/make_samples.py            # needs PIL for the side-by-side sheets
Reads LAM_API_URL / LAM_TOKEN from web/.env.local (written by scripts/colab_up.sh).
"""

import base64
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "samples"

# (slug, prompt, seed) - fixed seeds so the pairs can be regenerated. 2-3 are the paper's Table 5 prompts.
PROMPTS = [
    ("girl_reading_cherry_tree", "an anime girl reading a book under a cherry blossom tree", 1201),
    ("two_girls_beach", "two girls jumping happily on a beach, summer, bright sunlight", 1202),
    ("knight_sword_castle", "a knight girl in silver armor pointing a sword forward, castle at sunset", 1203),
    ("silver_hair_waving", "an anime girl with long silver hair and blue eyes waving, school uniform, cherry blossom park", 1204),
    ("couple_walking_park", "a boy and a girl holding hands walking in a park at sunset", 1205),
    ("old_man_bench", "an elderly man with a white beard sitting on a bench reading a book", 1206),
    ("classroom_boy_girl", "a boy with black hair and a girl with pink twintails sitting side by side in a classroom, smiling", 1207),
    ("three_friends_dancing", "three girls dancing together in a cozy living room", 1208),
]


def env() -> tuple[str, str]:
    kv = dict(line.strip().split("=", 1) for line in (ROOT / "web" / ".env.local").read_text().splitlines() if "=" in line)
    return kv["LAM_API_URL"].rstrip("/"), kv["LAM_TOKEN"]


def call(url: str, token: str, path: str, body: dict, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-LAM-Token": token})
            return json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as e:  # tunnel hiccups: retry
            last = e
            print(f"   retry {attempt + 1} after {type(e).__name__}: {e}", flush=True)
            time.sleep(8)
    raise RuntimeError(f"{path} failed: {last}")


def to_img(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", b64)))).convert("RGB")


def caption(img: Image.Image, text: str, h: int = 44) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + h), "white")
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((10, img.height + 12), text, fill=(0, 0, 0), font=ImageFont.load_default())
    return out


def main() -> None:
    url, token = env()
    OUT.mkdir(parents=True, exist_ok=True)
    health = json.load(urllib.request.urlopen(f"{url}/health", timeout=30))
    if health["status"] != "ready":
        sys.exit(f"server not ready: {health}")
    rows = []
    for slug, prompt, seed in PROMPTS:
        d = OUT / slug
        if (d / "side_by_side.png").exists():
            print("skip (done):", slug, flush=True)
            rows.append((slug, prompt, Image.open(d / "side_by_side.png")))
            continue
        print("==", slug, "-", prompt, flush=True)
        t0 = time.time()
        plan = call(url, token, "/plan", {"prompt": prompt, "planner": "llm"})
        res = call(url, token, "/generate", {"spec": plan["spec"], "user_prompt": prompt, "seed": seed,
                                            "steps": 28, "cfg": 5.0, "control_scale": 0.8, "with_baseline": True})
        d.mkdir(exist_ok=True)
        lam, base, pose = to_img(res["image"]), to_img(res["baseline"]["image"]), to_img(res["pose_image"])
        lam.save(d / "lam.png")
        base.save(d / "baseline.png")
        pose.save(d / "pose.png")
        (d / "spec.json").write_text(json.dumps({"prompt": prompt, "seed": seed, "spec": res["spec"],
                                                  "conditioning_prompt": res["prompt"],
                                                  "baseline_prompt": res["baseline"]["prompt"]}, indent=2))
        h = 512
        tiles = [im.resize((round(im.width * h / im.height), h)) for im in (pose, lam, base)]
        tiles = [caption(t, label) for t, label in zip(tiles, ("planned pose", "planner + pose + diffusion (paper §V)", "prompt-only baseline"))]
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 20 * 2, tiles[0].height + 50), "white")
        ImageDraw.Draw(sheet).text((10, 8), f'"{prompt}"  (seed {seed})', fill=(0, 0, 0), font=ImageFont.load_default())
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 50))
            x += t.width + 20
        sheet.save(d / "side_by_side.png")
        rows.append((slug, prompt, sheet))
        print(f"   done in {time.time() - t0:.0f}s", flush=True)

    # one contact sheet with every pair
    w = max(s.width for _, _, s in rows)
    contact = Image.new("RGB", (w, sum(s.height + 10 for _, _, s in rows)), "white")
    y = 0
    for _, _, s in rows:
        contact.paste(s, (0, y))
        y += s.height + 10
    contact.save(OUT / "all_samples.png")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
