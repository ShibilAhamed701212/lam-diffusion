"""Stage 1 of the dataset build: choose a balanced set of Danbooru posts from metadata.

Anime data is heavily skewed (solo young girls standing). Training on it as-is reproduces the
exact bias seen in the app (boys and old men rendered as girls), so sampling is stratified:
composition (solo/duo/trio, gender mix) gets fixed shares, rare ages are boosted, and
postures are flattened with inverse-sqrt frequency weights.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lam.data import tag_vocab as V

COMPOSITION_SHARE = {"solo_f": 0.30, "solo_m": 0.25, "duo_mf": 0.15, "duo_ff": 0.10, "duo_mm": 0.08, "trio": 0.12}
AGE_BOOST = {"elderly": 25.0, "adult": 3.0, "child": 2.0, "teen": 1.0, "unspecified": 1.0}
COLUMNS = ["id", "rating", "score", "image_width", "image_height", "file_ext", "file_size",
           "tag_string_general", "tag_string_meta", "is_deleted", "is_banned"]


def composition(girls: int, boys: int) -> str | None:
    n = girls + boys
    if n == 1:
        return "solo_f" if girls else "solo_m"
    if n == 2:
        return {2: "duo_ff", 0: "duo_mm"}.get(girls, "duo_mf")
    if n == 3:
        return "trio"
    return None


def load_candidates(parquet: Path, max_id: int) -> list[dict]:
    f = pq.ParquetFile(parquet)
    out = []
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg, columns=COLUMNS)
        keep = pc.and_(pc.equal(t["rating"], "g"), pc.invert(t["is_deleted"].fill_null(False)))
        keep = pc.and_(keep, pc.invert(t["is_banned"].fill_null(False)))
        keep = pc.and_(keep, pc.less_equal(t["id"], max_id))
        keep = pc.and_(keep, pc.greater_equal(pc.min_element_wise(t["image_width"], t["image_height"]), 768))
        keep = pc.and_(keep, pc.is_in(t["file_ext"], value_set=pa.array(["jpg", "png", "webp"])))
        keep = pc.and_(keep, pc.less_equal(t["file_size"], 12_000_000))
        t = t.filter(keep)
        for r in t.to_pylist():
            w, h = r["image_width"], r["image_height"]
            if max(w, h) / min(w, h) > 2.0:
                continue
            tags = V.split_tags(r["tag_string_general"])
            tagset = set(tags)
            if tagset & V.EXCLUDE or set(V.split_tags(r["tag_string_meta"])) & V.EXCLUDE_META:
                continue
            girls, boys, crowd = V.person_counts(tagset)
            comp = composition(girls, boys)
            if crowd or comp is None:
                continue
            acts = V.actions(tags)
            out.append({
                "id": r["id"], "w": w, "h": h, "score": r["score"] or 0, "tags": tags,
                "comp": comp, "age": V.age(tags), "posture": acts[0] if acts else "none",
                "framing": V.framing(tags),
            })
        print(f"row group {rg + 1}/{f.metadata.num_row_groups}: {len(out)} candidates", flush=True)
    return out


def stratified_sample(cands: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    comp_count = Counter(c["comp"] for c in cands)
    posture_count = Counter((c["comp"], c["posture"]) for c in cands)
    w = np.empty(len(cands))
    for i, c in enumerate(cands):
        share = COMPOSITION_SHARE[c["comp"]] / comp_count[c["comp"]]
        posture = 1.0 / math.sqrt(posture_count[(c["comp"], c["posture"])])
        quality = math.log1p(max(0, c["score"])) + 1.0
        framing = 1.3 if c["framing"] == "full_body" else 1.0  # full bodies carry the most pose signal
        w[i] = share * posture * AGE_BOOST[c["age"]] * quality * framing
    # Efraimidis-Spirakis weighted sampling without replacement, in log space: u ** (1/w)
    # underflows to 0 for every candidate when weights are ~1e-7, which silently turns the
    # stratified sample into an unweighted one.
    keys = np.log(rng.random(len(cands))) / w
    idx = np.argsort(-keys)[:n]
    return [cands[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/content/data/meta/metadata/posts-snapshot.parquet")
    ap.add_argument("--out", default="/content/data/selection.jsonl")
    ap.add_argument("--n", type=int, default=70_000)
    ap.add_argument("--max-id", type=int, default=8_360_000, help="danbooru2024-sfw covers ids up to ~8.36M")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cands = load_candidates(Path(a.parquet), a.max_id)
    print("candidates:", len(cands), Counter(c["comp"] for c in cands))
    picked = stratified_sample(cands, a.n, a.seed)
    with open(a.out, "w") as f:
        for c in picked:
            f.write(json.dumps(c) + "\n")
    for key in ("comp", "age", "posture", "framing"):
        print(key, Counter(c[key] for c in picked).most_common(25))


if __name__ == "__main__":
    main()
