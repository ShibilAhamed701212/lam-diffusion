"""Workflow / architecture diagrams for the report (matplotlib, no external deps)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

F = os.path.join(os.path.dirname(__file__), "figs"); os.makedirs(F, exist_ok=True)
COL = {"in": "#e8f0fe", "plan": "#fff4d6", "enc": "#e3f5ea", "unet": "#e6e1f7", "out": "#fde8e8", "data": "#eef1f4"}
EDGE = "#3b4252"

def box(ax, x, y, w, h, text, kind="enc", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=COL[kind], ec=EDGE, lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(ax, x0, y0, x1, y1, text="", fs=8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=EDGE,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text: ax.text((x0 + x1)/2, (y0 + y1)/2 + 0.12, text, ha="center", fontsize=fs, color="#555")

# ---------------------------------------------------------------- D1: inference workflow
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
box(ax, .2, 2.0, 1.9, 1.2, "User request\n\"a girl with long\npink hair and green\neyes, school uniform\"", "in")
box(ax, 2.7, 2.0, 1.9, 1.2, "LLM planner\nQwen2.5-3B-Instruct\n(few-shot)", "plan", bold=True)
arrow(ax, 2.1, 2.6, 2.7, 2.6)
box(ax, 5.2, 3.3, 2.1, 1.1, "Structured spec\ngender · age · hair · eyes\noutfit · pose · framing\nscene · boxes", "data")
arrow(ax, 4.6, 2.8, 5.2, 3.7, "descriptor")
box(ax, 5.2, 1.0, 2.1, 1.3, "Pose library →\nkeypoints, boxes\n(per character)", "data")
arrow(ax, 4.6, 2.4, 5.2, 1.7)
box(ax, 7.9, 1.9, 2.0, 1.5, "LAM Encoder\nglobal tokens (77×2048)\n+ per-character\naction tokens (40×2048)", "enc", bold=True)
arrow(ax, 7.3, 3.7, 7.9, 3.0); arrow(ax, 7.3, 1.6, 7.9, 2.2)
box(ax, 10.5, 1.9, 2.3, 1.5, "SDXL U-Net (full, trained)\ndual-stream conditioning:\nglobal text ⊕ box-masked\ncharacter tokens", "unet", bold=True)
arrow(ax, 9.9, 2.65, 10.5, 2.65)
box(ax, 10.5, 0.1, 2.3, 1.0, "Anime VAE decode\n→ image", "out", bold=True)
arrow(ax, 11.65, 1.9, 11.65, 1.1, "20–28 denoise steps\n(CFG 5)")
ax.text(6.5, 4.95, "LAM-Diffusion inference workflow", ha="center", fontsize=13, fontweight="bold")
ax.text(.2, 0.55, "Everything left of the encoder is the *plan* (text → structure). The right half is the trained model.\n"
        "Pose-library skeletons are looked up per requested action; no ControlNet is used.", fontsize=8.5, color="#444")
plt.savefig(f"{F}/d1_inference.png", bbox_inches="tight", dpi=150); plt.close()

# ---------------------------------------------------------------- D2: training pipeline
fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
stages = [("Data build", "Danbooru2024-sfw\nstratified 70k → 28,145\nRTMPose + WD tagger\n→ descriptors", "data", 0.1),
          ("Stage 1\nanime VAE", "6 000 steps\nMSE+LPIPS, EMA\nPSNR 31.5→32.3", "enc", 2.7),
          ("Stage 2a\naction-text", "HumanML3D captions\n2 500 steps\norder acc .49→.77", "enc", 4.9),
          ("Stage 2\nencoder align", "3 000 steps, batch 64\nCLIP-image InfoNCE\nR@1 .05→.83", "enc", 7.1),
          ("Stages 3–5\nfull U-Net", "11 000 steps, batch 4\ncomposite loss\ncurriculum + CFG drop", "unet", 9.5)]
for t, d, k, x in stages:
    box(ax, x, 1.7, 2.1, 2.1, f"{t}\n\n{d}", k, fs=8.5)
for x in (2.2, 4.8, 7.0, 9.2): arrow(ax, x, 2.75, x + 0.5, 2.75)
ax.text(6.5, 4.3, "Training pipeline (paper §IV-F stage numbers)", ha="center", fontsize=13, fontweight="bold")
ax.text(.1, 1.1, "Frozen throughout: VAE (after stage 1), base text encoders (LoRA only). Trained in Stages 3–5: every U-Net weight (lr 1e-5, 8-bit paged AdamW),\n"
        "LAM Encoder (lr 1e-4), dual-stream K/V low-rank adapters + gates. Planner: separate Qwen2.5-3B LoRA (optional in the app; LLM few-shot used for the demo).",
        fontsize=8.5, color="#444")
plt.savefig(f"{F}/d2_training.png", bbox_inches="tight", dpi=150); plt.close()

# ---------------------------------------------------------------- D3: dual-stream conditioning
fig, ax = plt.subplots(figsize=(11.5, 5.0)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.0); ax.axis("off")
box(ax, .2, 3.5, 2.5, .9, "Global tokens K,V\n(scene / style / props, 77)", "enc")
box(ax, .2, 1.9, 2.5, 1.2, "Character tokens (40 per char)\n4 identity + 6 fields × 6 word tokens\n+ pose/dynamics residual", "enc")
box(ax, .2, .3, 2.5, 1.0, "Temporal attention (stream 2)\nsub-action sequence → context", "plan")
arrow(ax, 1.45, 1.3, 1.45, 1.9)
box(ax, 3.6, 2.2, 2.7, 1.8, "ONE shared softmax per\ncross-attention layer\nQ · [K_global ; K_action]ᵀ\n+ [0 ; region_bias + gate]", "unet", bold=True)
arrow(ax, 2.7, 3.9, 3.6, 3.4); arrow(ax, 2.7, 2.5, 3.6, 2.7)
box(ax, 7.2, 2.7, 3.4, 1.4, "Region mask: a character's tokens are\nvisible only inside its (margin-padded)\nbox → binds attributes to the right person", "data")
arrow(ax, 7.2, 3.4, 6.3, 3.3)
box(ax, 3.6, .4, 2.7, 1.3, "K/V_action = frozen text K/V\n+ trainable low-rank update\n(zero-init), learned gate", "enc")
arrow(ax, 4.95, 1.7, 4.95, 2.2)
box(ax, 7.2, .5, 3.4, 1.3, "Applied in every cross-attention\nlayer: down / mid / up\n(350 low-rank + gate tensors)", "out")
arrow(ax, 6.3, 3.0, 7.2, 1.3, rad=-.2)
ax.text(5.7, 4.75, "Dual-stream conditioning pathway (paper §IV-C)", ha="center", fontsize=13, fontweight="bold")
plt.savefig(f"{F}/d3_dualstream.png", bbox_inches="tight", dpi=150); plt.close()
print("diagrams ok")
