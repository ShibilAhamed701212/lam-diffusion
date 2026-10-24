"""LAM Studio inference API. Runs on the GPU box (Colab) behind a tunnel.

Every request must carry the shared secret in the X-LAM-Token header, because the
tunnel URL is public.
"""

from __future__ import annotations

import base64
import io
import os
import secrets
import threading
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from planner import rule_plan
from poses import POSE_NAMES, FRAMING_CROP, render_openpose
from prompting import baseline_prompt, build_prompt, check_safe, spec_texts
from spec import ASPECTS, normalize_spec

TOKEN = os.environ.get("LAM_TOKEN", "")
USE_LLM = os.environ.get("LAM_PLANNER", "llm") == "llm"
ENGINE = os.environ.get("LAM_ENGINE", "controlnet")

app = FastAPI(title="LAM Studio")
_state: dict[str, Any] = {"status": "loading", "error": None, "planner": None, "generator": None}
_gpu = threading.Lock()


def _load_models() -> None:
    try:
        if ENGINE == "lam":  # the trained LAM-Diffusion model (paper §IV), no ControlNet
            from lam_engine import LAMEngine

            _state["generator"] = LAMEngine()
        else:  # the earlier ControlNet prototype
            from generator import Generator

            _state["generator"] = Generator()
        # The trained engine brings its own fine-tuned planner; the prompted base LLM is only the fallback.
        if USE_LLM and getattr(_state["generator"], "planner", None) is None:
            from planner import LLMPlanner

            _state["planner"] = LLMPlanner()
        _state["status"] = "ready"
    except Exception as e:  # surfaced through /health so the UI can show it
        _state["status"], _state["error"] = "error", f"{type(e).__name__}: {e}"
        raise


def start_loading() -> None:
    threading.Thread(target=_load_models, daemon=True).start()


def require_token(x_lam_token: str = Header(default="")) -> None:
    if not TOKEN or not secrets.compare_digest(x_lam_token, TOKEN):
        raise HTTPException(status_code=401, detail="bad token")


def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _render(spec: dict[str, Any]) -> dict[str, Any]:
    w, h = ASPECTS[spec["aspect"]]
    prompt, negative = build_prompt(spec)
    return {
        "spec": spec,
        "prompt": prompt,
        "negative": negative,
        "pose_image": _b64(render_openpose(spec["characters"], w, h)),
    }


def _ready() -> None:
    if _state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"models {_state['status']}: {_state['error'] or ''}")


class PlanReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    planner: Literal["llm", "rules"] = "llm"


class RenderReq(BaseModel):
    spec: dict[str, Any]


class GenerateReq(BaseModel):
    spec: dict[str, Any]
    user_prompt: str = Field(default="", max_length=1000)
    seed: int = Field(default=-1)
    steps: int = Field(default=28, ge=8, le=50)
    cfg: float = Field(default=5.0, ge=1.0, le=12.0)
    control_scale: float = Field(default=0.8, ge=0.0, le=1.5)
    with_baseline: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": _state["status"],
        "error": _state["error"],
        "llm_planner": _state["planner"] is not None or getattr(_state["generator"], "planner", None) is not None,
        "poses": POSE_NAMES,
        "framings": list(FRAMING_CROP),
        "aspects": list(ASPECTS),
    }


@app.post("/plan", dependencies=[Depends(require_token)])
def plan(req: PlanReq) -> dict[str, Any]:
    if reason := check_safe(req.prompt):
        raise HTTPException(status_code=400, detail=reason)
    t0 = time.time()
    spec, used = None, "rules"
    if req.planner == "llm" and getattr(_state["generator"], "planner", None) is not None:
        with _gpu:
            spec = _state["generator"].plan(req.prompt)
        used = "fine-tuned planner" if spec else "rules (planner output was not valid JSON)"
    elif req.planner == "llm" and _state["planner"] is not None:
        with _gpu:
            spec, used = _state["planner"].plan(req.prompt)
    if spec is None:
        spec = rule_plan(req.prompt)
    if reason := check_safe(*spec_texts(spec)):
        raise HTTPException(status_code=400, detail=reason)
    return {**_render(spec), "planner_used": used, "seconds": round(time.time() - t0, 2)}


@app.post("/render", dependencies=[Depends(require_token)])
def render(req: RenderReq) -> dict[str, Any]:
    spec = normalize_spec(req.spec)
    if reason := check_safe(*spec_texts(spec)):
        raise HTTPException(status_code=400, detail=reason)
    return _render(spec)


@app.post("/generate", dependencies=[Depends(require_token)])
def generate(req: GenerateReq) -> dict[str, Any]:
    _ready()
    spec = normalize_spec(req.spec)
    if reason := check_safe(req.user_prompt, *spec_texts(spec)):
        raise HTTPException(status_code=400, detail=reason)
    rendered = _render(spec)
    w, h = ASPECTS[spec["aspect"]]
    seed = req.seed if req.seed >= 0 else secrets.randbelow(2**31)
    pose = render_openpose(spec["characters"], w, h)
    opts = dict(width=w, height=h, seed=seed, steps=req.steps, cfg=req.cfg, control_scale=req.control_scale)
    gen = _state["generator"]

    t0 = time.time()
    with _gpu:
        if ENGINE == "lam":
            image, _ = gen.generate(spec, width=w, height=h, seed=seed, steps=req.steps, cfg=req.cfg)
        else:
            image = gen.generate(rendered["prompt"], rendered["negative"], pose, **opts)
        lam_seconds = time.time() - t0
        baseline = None
        if req.with_baseline and req.user_prompt.strip():
            bp, bn = baseline_prompt(req.user_prompt)
            if ENGINE == "lam":
                baseline = {"image": _b64(gen.baseline(req.user_prompt, width=w, height=h, seed=seed,
                                                       steps=req.steps, cfg=req.cfg)), "prompt": req.user_prompt}
            else:
                baseline = {"image": _b64(gen.generate(bp, bn, None, **opts)), "prompt": bp}
    return {
        **rendered,
        "image": _b64(image),
        "baseline": baseline,
        "seed": seed,
        "seconds": round(lam_seconds, 2),
    }
