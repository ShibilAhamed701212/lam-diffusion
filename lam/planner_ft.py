"""Fine-tune the planner (Eq. 1, A = F_LAM(N)): natural-language request -> action descriptor.

Qwen2.5-3B-Instruct + LoRA, trained on (request, descriptor) pairs from planner_data.py, so
gender / age / actions are learned from data instead of keyword rules. Output is the
descriptor without geometry; layout boxes come from positions and keypoints from the pose
library (lam/spec_bridge.py).
"""

from __future__ import annotations

import argparse
import json
import random

import torch

from lam.descriptor import ACTIONS, AGES, FRAMINGS, GENDERS

SYSTEM = (
    "Convert the request into an anime illustration descriptor. Reply with JSON only: "
    '{"scene": str, "style": str, "props": [str], "aspect": "portrait|square|landscape", '
    '"characters": [{"gender": ' + "|".join(GENDERS) + ', "age": ' + "|".join(AGES) + ", "
    '"hair_color": str, "hair_style": str, "eye_color": str, "outfit": str, "expression": str, "extra": str, '
    '"actions": [action tags], "framing": ' + "|".join(FRAMINGS) + ', "position": "left|center|right"}]}'
)
CASES = ["an anime grandfater with long silver hair and blue eyes waving, school uniform, cherry blossom park",
         "an old man with a white beard sitting on a bench", "a boy with black hair running in the rain",
         "two girls, one with pink twintails and one with short blue hair, standing in a classroom",
         "a grandma knitting by the window, smiling", "a guy in a hoodie with headphones, upper body"]


def target(d: dict) -> dict:
    """Ground-truth descriptor (no geometry) the planner should produce."""
    chars = []
    for c in d["characters"]:
        cx = (c["bbox"][0] + c["bbox"][2]) / 2
        chars.append({k: c.get(k, "") for k in ("gender", "age", "hair_color", "hair_style", "eye_color", "outfit",
                                                 "expression", "extra")}
                     | {"actions": c.get("actions", []), "framing": c["framing"],
                        "position": "left" if cx < 0.4 else "right" if cx > 0.6 else "center"})
    w, h = d["bucket"]
    aspect = "square" if abs(w - h) < 150 else "portrait" if h > w else "landscape"
    return {"scene": d["global"]["scene"], "style": d["global"]["style"], "props": d["global"]["props"],
            "aspect": aspect, "characters": chars}


def chat(tok, text: str, answer: str | None = None) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    if answer is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)


def parse(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[s:e + 1]) if s >= 0 else None
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def plan(model, tok, texts: list[str], max_new: int = 400) -> list[dict | None]:
    tok.padding_side = "left"
    enc = tok([chat(tok, t) for t in texts], return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return [parse(x) for x in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


def evaluate(model, tok, pairs: list[tuple[str, dict]]) -> dict:
    model.eval()
    n = ok_json = count_ok = 0
    g_ok = a_ok = act_ok = chars = 0
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        for (_, gt), pred in zip(chunk, plan(model, tok, [t for t, _ in chunk])):
            n += 1
            if not pred or not isinstance(pred.get("characters"), list):
                continue
            ok_json += 1
            pc, gc = pred["characters"], gt["characters"]
            count_ok += len(pc) == len(gc)
            for p, g in zip(pc, gc):
                chars += 1
                g_ok += p.get("gender") == g["gender"]
                a_ok += p.get("age") == g["age"]
                act_ok += bool(set(p.get("actions") or []) & set(g["actions"])) or not g["actions"]
    model.train()
    return {"valid_json": ok_json / n, "char_count_acc": count_ok / n, "gender_acc": g_ok / max(1, chars),
            "age_acc": a_ok / max(1, chars), "action_overlap": act_ok / max(1, chars)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/content/data/index.jsonl")
    ap.add_argument("--pairs", default="/content/data/planner_pairs.jsonl")
    ap.add_argument("--out", default="/content/ckpt/planner_lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    descs = {d["id"]: d for d in map(json.loads, open(a.index))}
    pairs = [(p["text"], target(descs[p["id"]]), p["split"]) for p in map(json.loads, open(a.pairs)) if p["id"] in descs]
    train = [(t, g) for t, g, s in pairs if s == "train"]
    test = [(t, g) for t, g, s in pairs if s == "test"][:400]
    print(f"train {len(train)} test {len(test)}", flush=True)

    mid = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    before = evaluate(model, tok, test)
    print("before fine-tuning:", before, flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = int(len(train) * a.epochs / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    rng = random.Random(0)
    tok.padding_side = "right"
    model.train()
    for step in range(steps):
        batch = rng.sample(train, a.batch)
        full = [chat(tok, t, json.dumps(g)) for t, g in batch]
        prompt_lens = [len(tok(chat(tok, t)).input_ids) for t, _ in batch]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_lens):
            labels[i, :n] = -100  # learn the descriptor, not the request
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
    tok.padding_side = "left"
    after = evaluate(model, tok, test)
    print("after fine-tuning:", after, flush=True)
    model.save_pretrained(a.out)
    cases = dict(zip(CASES, plan(model, tok, CASES)))
    for k, v in cases.items():
        print(k, "->", [(c.get("gender"), c.get("age"), c.get("actions")) for c in (v or {}).get("characters", [])], flush=True)
    json.dump({"before": before, "after": after, "cases": cases}, open(f"{a.out}/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
