"""Stage 2 of the dataset build: download only the selected images.

danbooru2024-sfw stores images in 7.5 GB tar shards, but each shard has a JSON index with
the byte offset of every file, so single images can be fetched with HTTP range requests
instead of pulling whole shards. Resumable: already-saved ids are skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

REPO = "https://huggingface.co/datasets/deepghs/danbooru2024-sfw/resolve/main/images"
N_SHARDS = 1000  # images/0000 .. images/0999 (each a .tar plus its .json index)
MAX_SIDE = 1536

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16, max_retries=3))
    return _local.s


def build_index(wanted: set[int], cache: Path, workers: int = 32) -> dict[int, tuple[int, int, int]]:
    """id -> (shard, offset, size) for the wanted ids, from the shard indexes."""
    cache.mkdir(parents=True, exist_ok=True)

    def one(shard: int) -> dict[int, tuple[int, int, int]]:
        path = cache / f"{shard:04d}.json"
        if not path.exists():
            r = _session().get(f"{REPO}/{shard:04d}.json", timeout=120)
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            path.write_bytes(r.content)
        files = json.loads(path.read_text())["files"]
        out = {}
        for name, meta in files.items():
            stem = name.split(".")[0]
            if stem.isdigit() and int(stem) in wanted:
                out[int(stem)] = (shard, meta["offset"], meta["size"])
        return out

    index: dict[int, tuple[int, int, int]] = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, part in enumerate(ex.map(one, range(N_SHARDS))):
            index.update(part)
            if (i + 1) % 100 == 0:
                print(f"indexed {i + 1}/{N_SHARDS} shards, {len(index)} wanted ids found", flush=True)
    return index


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        return bg.convert("RGB")
    return img.convert("RGB")


_cdn: dict[int, tuple[str, float]] = {}
_cdn_lock = threading.Lock()
CDN_TTL = 20 * 60


def resolve_urls(shards: set[int], workers: int = 16) -> dict[int, str]:
    """Signed CDN URL per shard, resolved once in the parent process."""
    def one(sh: int) -> tuple[int, str]:
        for attempt in range(6):
            r = _session().head(f"{REPO}/{sh:04d}.tar", allow_redirects=False, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt * 5)))
                continue
            url = r.headers.get("location") or f"{REPO}/{sh:04d}.tar"
            return sh, ("https://huggingface.co" + url) if url.startswith("/") else url
        return sh, f"{REPO}/{sh:04d}.tar"
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, sorted(shards)))


def shard_url(shard: int, refresh: bool = False) -> str:
    """Resolve the shard's signed CDN URL once and reuse it, instead of paying a huggingface.co
    redirect round trip (and its anonymous rate limit) on every range request."""
    now = time.time()
    with _cdn_lock:
        hit = _cdn.get(shard)
        if hit and not refresh and now - hit[1] < CDN_TTL:
            return hit[0]
    r = _session().head(f"{REPO}/{shard:04d}.tar", allow_redirects=False, timeout=60)
    url = r.headers.get("location") or f"{REPO}/{shard:04d}.tar"
    if url.startswith("/"):
        url = "https://huggingface.co" + url
    with _cdn_lock:
        _cdn[shard] = (url, now)
    return url


def fetch_one(post_id: int, loc: tuple[int, int, int], out_dir: Path) -> str:
    dest = out_dir / f"{post_id}.jpg"
    if dest.exists():
        return "skip"
    shard, offset, size = loc
    rng = {"Range": f"bytes={offset}-{offset + size - 1}"}
    for attempt in range(8):
        r = _session().get(shard_url(shard), headers=rng, timeout=120)
        if r.status_code in (401, 403):  # signed URL expired: resolve again
            r = _session().get(shard_url(shard, refresh=True), headers=rng, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(float(r.headers.get("retry-after", min(120, 5 * 2 ** attempt))))
    if r.status_code != 206 or len(r.content) != size:
        return f"http {r.status_code}"
    img = Image.open(io.BytesIO(r.content))
    img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode at a reduced scale directly (much cheaper)
    img = to_rgb(img)
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    tmp = dest.with_suffix(".tmp")
    img.save(tmp, "JPEG", quality=93)
    tmp.rename(dest)
    return "ok"


def fetch_chunk(items: list[tuple[int, tuple[int, int, int]]], out_dir: str, threads: int,
                urls: dict[int, str]) -> dict[str, int]:
    now = time.time()
    _cdn.update({sh: (u, now) for sh, u in urls.items()})
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(threads) as ex:
        for res in ex.map(lambda it: _safe_fetch(it[0], it[1], Path(out_dir)), items):
            stats[res] = stats.get(res, 0) + 1
    return stats


def _safe_fetch(pid: int, loc: tuple[int, int, int], out: Path) -> str:
    try:
        return fetch_one(pid, loc, out)
    except Exception as e:  # corrupt image / network: skip, it's one sample of many
        return type(e).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="/content/data/selection.jsonl")
    ap.add_argument("--out", default="/content/data/images")
    ap.add_argument("--index-cache", default="/content/data/shard_index")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    sel = [json.loads(line) for line in open(a.selection)]
    wanted = {s["id"] for s in sel}
    index = build_index(wanted, Path(a.index_cache))
    print(f"{len(index)}/{len(wanted)} selected ids present in danbooru2024-sfw", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Decoding/resizing originals is CPU-bound and threads serialise on the GIL, so work is
    # split across processes (one per core), each downloading with a few threads.
    items = [(pid, loc) for pid, loc in index.items() if not (out / f"{pid}.jpg").exists()]
    print(f"{len(items)} images still to download", flush=True)
    urls = resolve_urls({loc[0] for _, loc in items})
    print(f"resolved {len(urls)} shard URLs", flush=True)
    n_proc = max(1, (os.cpu_count() or 4) - 2)
    chunks = [items[i::n_proc * 8] for i in range(n_proc * 8)]
    stats: dict[str, int] = {}
    with ProcessPoolExecutor(n_proc) as ex:
        futs = [ex.submit(fetch_chunk, c, str(out), max(2, a.workers // n_proc), urls) for c in chunks]
        for i, fut in enumerate(as_completed(futs)):
            for k, v in fut.result().items():
                stats[k] = stats.get(k, 0) + v
            print(f"chunk {i + 1}/{len(futs)} {stats}", flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
