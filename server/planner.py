"""The planner ("Large Action Model" stage): natural language -> character spec.

Two implementations share one output shape:
- LLMPlanner: an instruction-tuned open model (Qwen2.5) emits the spec as JSON.
- rule_plan: keyword matching, close to the paper's rule-assisted planner. Used as a
  fallback when the LLM output can't be parsed, and as an ablation baseline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from poses import FRAMING_CROP, POSE_NAMES
from spec import MAX_CHARACTERS, normalize_spec

SYSTEM_PROMPT = f"""You convert a request for an anime illustration into a JSON character spec.
Reply with JSON only, no prose. Schema:
{{
  "scene": "background/setting as short tags, e.g. 'cherry blossoms, park, daytime'",
  "style": "art style or lighting tags, e.g. 'soft lighting, watercolor'",
  "props": ["objects that are not characters, e.g. 'book', 'cat'"],
  "aspect": "portrait | square | landscape",
  "characters": [
    {{
      "gender": "female | male | other",
      "age": "unspecified | child | teen | young_adult | adult | elderly",
      "hair_color": "e.g. 'silver'",
      "hair_style": "e.g. 'long hair, twintails'",
      "eye_color": "e.g. 'blue'",
      "outfit": "e.g. 'school uniform, red ribbon'",
      "expression": "facial expression only, e.g. 'smile', 'serious'",
      "action": "short tag for what they do, e.g. 'reading book'",
      "pose": one of {POSE_NAMES},
      "framing": one of {sorted(FRAMING_CROP)},
      "position": "left | center | right"
    }}
  ]
}}
Rules: at most {MAX_CHARACTERS} characters (animals and objects go in props). Leave a field ""
if the request doesn't say. Gender and age come from words like man/woman/boy/girl/grandfather/
grandmother/old/father/mother; grandparents and old people are "elderly". Beards, glasses and
accessories go in "extra". Pick the pose closest to the described action. Use Danbooru-style
short tags. Give characters different positions when there are several."""

FEW_SHOTS = [
    ("a girl with long silver hair reading a book under a cherry blossom tree", {
        "scene": "cherry blossoms, tree, outdoors, petals",
        "style": "soft lighting",
        "props": ["book"],
        "aspect": "portrait",
        "characters": [{
            "gender": "female", "age": "unspecified", "hair_color": "silver", "hair_style": "long hair",
            "eye_color": "", "outfit": "", "expression": "calm", "action": "reading book", "extra": "",
            "pose": "holding_object", "framing": "full_body", "position": "center",
        }],
    }),
    ("an old grandpa with a white beard sitting on a bench in a garden, smiling", {
        "scene": "garden, bench, outdoors",
        "style": "",
        "props": ["bench"],
        "aspect": "portrait",
        "characters": [{
            "gender": "male", "age": "elderly", "hair_color": "white", "hair_style": "short hair",
            "eye_color": "", "outfit": "", "expression": "smile", "action": "sitting on bench",
            "extra": "white beard", "pose": "sitting", "framing": "full_body", "position": "center",
        }],
    }),
]


class LLMPlanner:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.model.eval()
        self.device = device

    def plan(self, prompt: str) -> tuple[dict[str, Any], str]:
        import torch

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user, reply in FEW_SHOTS:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(reply)}]
        messages.append({"role": "user", "content": prompt})
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=700, do_sample=False)
        reply = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = _extract_json(reply)
        if parsed is None:
            return rule_plan(prompt), "rules (LLM output was not valid JSON)"
        spec = normalize_spec(parsed)
        fixed = apply_identity_hints(spec, prompt)
        return spec, "llm + keyword fix" if fixed else "llm"


# ---------------------------------------------------------------- identity keywords

# Small models sometimes miss gender/age (e.g. "grandfater" -> gender "other"), and the image
# model then falls back to its default of a young girl. When the prompt names exactly one
# gender, or an age, unambiguously, these keywords override the planner for solo characters.
_MALE = re.compile(r"\b(boys?|man|men|male|guys?|gentleman|grand\s*f\w*|grand\s*pa\w*|grandad|granddad|"
                   r"father|dad|papa|husband|king|prince|monk|uncle|brother|son|he|him|his|old man)\b")
_FEMALE = re.compile(r"\b(girls?|wom[ae]n|female|lady|ladies|grand\s*m\w*|granny|mother|mom|mama|"
                     r"wife|queen|princess|witch|nun|aunt|sister|daughter|she|her|old woman)\b")
_AGE_PATTERNS = [
    ("elderly", re.compile(r"\b(elderly|elder|aged|senior|grand\s*(f|m|p)\w*|granny|grandad|granddad|"
                           r"old\s+(man|woman|lady|men|women|person|people|guy|gentleman|wizard|monk))\b")),
    ("child", re.compile(r"\b(child|children|kids?|toddler|little\s+(boy|girl))\b")),
    ("adult", re.compile(r"\b(mature|adult|middle[- ]aged|mother|father|mom|dad|businessman|businesswoman)\b")),
    ("teen", re.compile(r"\b(teen\w*|high\s*school\w*|schoolgirl|schoolboy)\b")),
]


def infer_identity(prompt: str) -> tuple[str | None, str | None]:
    p = prompt.lower()
    male, female = bool(_MALE.search(p)), bool(_FEMALE.search(p))
    gender = "male" if male and not female else "female" if female and not male else None
    age = next((a for a, rx in _AGE_PATTERNS if rx.search(p)), None)
    return gender, age


def apply_identity_hints(spec: dict[str, Any], prompt: str) -> bool:
    """Correct a solo character's gender/age from unambiguous prompt keywords. Returns True if changed."""
    if len(spec["characters"]) != 1:
        return False
    ch = spec["characters"][0]
    gender, age = infer_identity(prompt)
    changed = False
    if gender and ch["gender"] != gender:
        ch["gender"], changed = gender, True
    if age and ch["age"] != age and (ch["age"] == "unspecified" or age == "elderly"):
        ch["age"], changed = age, True
    return changed


def _extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- rule-based planner

_COLORS = ["silver", "white", "black", "blonde", "blue", "red", "pink", "green", "purple",
           "brown", "orange", "grey", "aqua", "light blue", "dark blue"]
_HAIR_STYLES = ["long hair", "short hair", "medium hair", "very long hair", "twintails", "ponytail",
                "braid", "bob cut", "messy hair", "hime cut", "side ponytail", "ahoge"]
_EXPRESSIONS = ["smile", "smiling", "happy", "sad", "angry", "crying", "blush", "surprised",
                "serious", "laughing", "calm", "embarrassed", "determined"]
_OUTFITS = ["school uniform", "serafuku", "kimono", "yukata", "maid outfit", "hoodie", "armor",
            "dress", "suit", "jacket", "sweater", "witch hat", "cape", "uniform", "t-shirt"]
_POSE_WORDS = {
    "waving": "waving", "wave": "waving", "cheer": "arms_up", "arms up": "arms_up",
    "hands on hips": "hands_on_hips", "peace": "peace_sign", "v sign": "peace_sign",
    "point": "pointing", "holding": "holding_object", "reading": "holding_object",
    "walk": "walking", "run": "running", "jump": "jumping", "sit": "sitting", "kneel": "kneeling",
}
_NUM = {"two": 2, "three": 3, "2": 2, "3": 3}
_SCENES = ["cherry blossom", "beach", "forest", "classroom", "city", "night", "sunset", "castle",
           "river", "rooftop", "cafe", "library", "living room", "bedroom", "snow", "rain", "space"]


def rule_plan(prompt: str) -> dict[str, Any]:
    p = prompt.lower()
    girls = _count(p, ("girl", "girls", "woman", "women", "lady", "ladies"))
    boys = _count(p, ("boy", "boys", "man", "men", "guy", "guys"))
    genders = (["female"] * girls + ["male"] * boys)[:MAX_CHARACTERS]
    if not genders:
        genders = [infer_identity(prompt)[0] or "other"]

    pose = next((v for k, v in _POSE_WORDS.items() if k in p), "standing")
    framing = "upper_body" if any(w in p for w in ("portrait", "close-up", "closeup", "upper body")) else "full_body"
    hair_color = next((c for c in _COLORS if re.search(rf"\b{c}(-| )?hair", p) or re.search(rf"\b{c}-haired", p)), "")
    eye_color = next((c for c in _COLORS if re.search(rf"\b{c} eyes", p)), "")
    characters = []
    for i, g in enumerate(genders):
        characters.append({
            "gender": g,
            "hair_color": hair_color if i == 0 else "",
            "hair_style": next((s for s in _HAIR_STYLES if s in p), "") if i == 0 else "",
            "eye_color": eye_color if i == 0 else "",
            "outfit": next((o for o in _OUTFITS if o in p), ""),
            "expression": next((e for e in _EXPRESSIONS if e in p), ""),
            "action": "",
            "pose": pose,
            "framing": framing,
            "position": ["left", "right", "center"][i] if len(genders) > 1 else "center",
        })
    props = [a for a in ("cat", "dog", "book", "umbrella", "sword", "guitar", "flower", "bookshelf") if a in p]
    spec = normalize_spec({
        "scene": ", ".join(s for s in _SCENES if s in p),
        "style": "",
        "props": props,
        "characters": characters,
    })
    apply_identity_hints(spec, prompt)
    return spec


def _count(p: str, words: tuple[str, ...]) -> int:
    """`words` alternates singular/plural forms, e.g. ("girl", "girls", "woman", "women")."""
    singulars, plurals = words[0::2], words[1::2]
    for plural in plurals:
        for num_word, n in _NUM.items():
            if re.search(rf"\b{num_word} {plural}\b", p):
                return n
    if any(re.search(rf"\b{w}\b", p) for w in plurals):
        return 2
    return 1 if any(re.search(rf"\b{w}\b", p) for w in singulars) else 0


assert set(_POSE_WORDS.values()) <= set(POSE_NAMES)
