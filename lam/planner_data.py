"""Training pairs for the learned planner (Eq. 1, A = F_LAM(N)).

For real dataset descriptors, Qwen writes the natural-language request a user might type
(varied phrasing: everyday words like "grandpa", "old man", "a guy"), and a fraction get
character-level typos ("grandfater"). Fine-tuning the planner on (request -> descriptor)
teaches it gender/age/pose from data, which is what replaces the keyword rules.
The same descriptions are the text inputs for the prompt-only baseline and CLIP Score.
"""

from __future__ import annotations

import argparse
import json
import random

import torch

AGE_WORDS = {"elderly": "elderly", "adult": "adult", "child": "young child", "teen": "teenage"}
STYLES = [
    "a short casual request, like someone typing quickly",
    "one descriptive sentence",
    "a detailed request listing the look of each character",
    "a request using everyday words (grandpa, old lady, kid, guy, girl) instead of formal terms",
]


def facts(d: dict) -> str:
    lines = []
    for i, c in enumerate(d["characters"], 1):
        who = ("old " if c["age"] == "elderly" else "") + {"female": "woman/girl", "male": "man/boy"}.get(c["gender"], "person")
        if c["age"] in AGE_WORDS and c["age"] != "elderly":
            who = f"{AGE_WORDS[c['age']]} {who}"
        parts = [f"character {i}: {who}"]
        for f in ("hair_color", "hair_style", "eye_color", "outfit", "expression", "extra"):
            if c.get(f):
                parts.append(f"{f.replace('_', ' ')}: {c[f]}")
        if c.get("actions"):
            parts.append("doing: " + ", ".join(a.replace("_", " ") for a in c["actions"]))
        parts.append(f"framing: {c['framing'].replace('_', ' ')}")
        pos = (c["bbox"][0] + c["bbox"][2]) / 2
        if len(d["characters"]) > 1:
            parts.append("on the " + ("left" if pos < 0.4 else "right" if pos > 0.6 else "center"))
        lines.append("; ".join(parts))
    g = d["global"]
    if g.get("scene"):
        lines.append(f"setting: {g['scene']}")
    if g.get("props"):
        lines.append(f"objects: {', '.join(g['props'])}")
    return "\n".join(lines)


def typo(text: str, rng: random.Random) -> str:
    words = text.split()
    for _ in range(rng.randint(1, 2)):
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 4:
            j = rng.randrange(1, len(w) - 1)
            op = rng.choice(["drop", "swap", "dup"])
            w = w[:j] + w[j + 1:] if op == "drop" else w[:j] + w[j + 1] + w[j] + w[j + 2:] if op == "swap" else w[:j] + w[j] + w[j:]
            words[i] = w
    return " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--out", default="/content/data/planner_pairs.jsonl")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--typo-rate", type=float, default=0.15)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(0)
    items = [json.loads(line) for line in open(a.index)]
    test = [d for d in items if d["split"] == "test"]
    train = [d for d in items if d["split"] == "train"]
    rng.shuffle(train)
    todo = test + train[: a.n_train]

    mid = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).cuda().eval()
    out = open(a.out, "w")
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        prompts = []
        for d in chunk:
            style = rng.choice(STYLES)
            msg = [{"role": "system", "content": "You write requests that a user would type into an anime illustration generator. "
                                                 "Reply with the request only."},
                   {"role": "user", "content": f"Write {style} for an anime illustration with exactly these facts:\n{facts(d)}"}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=90, do_sample=True, temperature=0.8, top_p=0.9)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for d, t in zip(chunk, texts):
            t = " ".join(t.strip().strip('"').split())
            if d["split"] == "train" and rng.random() < a.typo_rate:
                t = typo(t, rng)
            out.write(json.dumps({"id": d["id"], "split": d["split"], "text": t}) + "\n")
        out.flush()
        if (i // a.batch) % 20 == 0:
            print(f"{i + len(chunk)}/{len(todo)}: {texts[0][:120]!r}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
