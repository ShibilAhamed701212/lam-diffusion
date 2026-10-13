"""Paper figures and statistics (no GPU needed; all from logged data in reports/data/).

    python reports/scripts/make_paper_figs.py
Writes reports/figures/{d4..d10, p1..p6}.png, reports/analysis/{stats.csv, tables.tex}.
"""

import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
D, FIG, ANA = (os.path.join(HERE, "..", x) for x in ("data", "figures", "analysis"))
os.makedirs(FIG, exist_ok=True); os.makedirs(ANA, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25})
C = {"data": "#dbeafe", "model": "#dcfce7", "train": "#fef3c7", "io": "#f3f4f6", "loss": "#fee2e2", "frozen": "#e5e7eb",
     "edge": "#374151", "ours": "#2563eb", "base": "#9ca3af", "bad": "#dc2626", "good": "#16a34a", "amber": "#d97706"}
load = lambda f: json.load(open(os.path.join(D, f)))  # noqa: E731


# ------------------------------------------------------------------ flow-chart helpers
class Canvas:
    def __init__(self, w, h, title=None):
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off"); self.ax.grid(False)
        if title:
            self.ax.text(w / 2, h - 0.25, title, ha="center", va="center", fontsize=12, fontweight="bold")

    def box(self, x, y, w, h, text, kind="model", fs=8.5, bold=False):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=C[kind],
                                         ec=C["edge"], lw=1.1))
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                     fontweight="bold" if bold else "normal", linespacing=1.25)
        return (x, y, w, h)

    def arrow(self, a, b, label=None, style="-|>", rad=0.0, ls="-", color=None, fs=7.5, lx=0, ly=0.12, sa=None, sb=None):
        def side(r, sd):
            x, y, w, h = r
            return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[sd]

        def anchor(r, toward):
            x, y, w, h = r
            cx, cy = x + w / 2, y + h / 2
            tx, ty = toward
            dx, dy = tx - cx, ty - cy
            if abs(dx) * h > abs(dy) * w:
                return (x + w if dx > 0 else x, cy)
            return (cx, y + h if dy > 0 else y)
        ca = (a[0] + a[2] / 2, a[1] + a[3] / 2) if len(a) == 4 else a
        cb = (b[0] + b[2] / 2, b[1] + b[3] / 2) if len(b) == 4 else b
        pa = side(a, sa) if sa else (anchor(a, cb) if len(a) == 4 else a)
        pb = side(b, sb) if sb else (anchor(b, ca) if len(b) == 4 else b)
        self.ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle=style, mutation_scale=11, lw=1.2, ls=ls,
                                          color=color or C["edge"], connectionstyle=f"arc3,rad={rad}"))
        if label:
            self.ax.text((pa[0] + pb[0]) / 2 + lx, (pa[1] + pb[1]) / 2 + ly, label, ha="center", fontsize=fs,
                         color=color or C["edge"])

    def text(self, x, y, s, fs=8, **kw):
        self.ax.text(x, y, s, fontsize=fs, **kw)

    def save(self, name):
        self.fig.savefig(os.path.join(FIG, name)); plt.close(self.fig)


# d4: end-to-end system architecture
c = Canvas(13, 6.0, "LAM-Diffusion: system architecture")
n = c.box(0.2, 4.5, 1.7, 0.9, "User prompt N\n\"a boy and a girl\nin a classroom\"", "io")
pl = c.box(2.4, 4.5, 2.0, 0.9, "Planner  F_LAM\n(Qwen2.5-3B LLM\n+ rule fallback)", "model")
ds = c.box(4.9, 4.35, 2.7, 1.2, "Action descriptor A\ncharacters: gender, age,\nordered actions, keypoints,\nbbox, hair/eyes/outfit…\n+ global scene/style", "data", fs=8)
en = c.box(8.1, 4.5, 2.0, 0.9, "LAM Encoder\n(§IV-B)", "model", bold=True)
tk = c.box(10.6, 4.35, 2.2, 1.2, "Action tokens\nB × N × 2048\n(+ pooled, + CLIP\nembedding)", "data")
ds2 = c.box(9.6, 2.3, 3.2, 1.1, "Dual-stream conditioning (§IV-C)\n① shared box-masked cross-attention\n② temporal attention", "model", bold=True)
un = c.box(4.4, 2.3, 3.5, 1.1, "Denoising U-Net\nAnimagine XL 4.0 (fully trained)\nT steps, classifier-free guidance", "model", bold=True)
nz = c.box(1.9, 2.4, 1.8, 0.9, "noise z_T\n~ N(0, I)", "io")
z0 = c.box(5.2, 0.45, 2.1, 0.8, "clean latent z_0", "data")
vd = c.box(8.0, 0.45, 2.2, 0.8, "VAE decoder\n(anime fine-tuned)", "model")
im = c.box(11.3, 0.45, 1.5, 0.8, "Image I", "io", bold=True)
ls_ = c.box(0.2, 0.3, 4.6, 1.3, "Training only: Eq. 3\nL = λ1·L_diff + λ2·L_action-img\n+ λ3·L_perc + λ4·L_style", "loss", fs=8)
for a, b in [(n, pl), (pl, ds), (ds, en), (en, tk)]:
    c.arrow(a, b)
c.arrow(tk, ds2, sa="b", sb="t")
c.arrow(ds2, un, "conditions\nevery layer", sa="l", sb="r", ly=0.12)
c.arrow(nz, un, sa="r", sb="l")
c.arrow(un, z0, sa="b", sb="t", label="z_0", lx=0.35, ly=-0.05)
c.arrow(z0, vd, sa="r", sb="l"); c.arrow(vd, im, sa="r", sb="l")
c.arrow(un, ls_, "ε̂, x̂0", ls="--", sa="b", sb="r", rad=-0.2, lx=-0.7, ly=-0.35)
c.text(0.2, 3.55, "Eq. 1:  A = F_LAM(N)\nEq. 2:  I = G_diff(A)", fs=9, style="italic")
c.save("d4_system_architecture.png")

# d5: planner (Eq. 1)
c = Canvas(12, 4.4, "Planner  F_LAM  (Eq. 1): text → structured descriptor")
a = c.box(0.2, 1.7, 1.6, 0.9, "Prompt N", "io")
b = c.box(2.2, 1.7, 1.7, 0.9, "Safety filter\n(SFW only)", "frozen")
l = c.box(4.3, 1.7, 2.0, 0.9, "LLM planner\nQwen2.5-3B (+LoRA)\n→ JSON spec", "model")
r = c.box(4.3, 0.3, 2.0, 0.8, "Rule planner\n(keyword fallback)", "frozen")
nsp = c.box(6.9, 1.7, 1.9, 0.9, "normalize_spec\ndefaults, clamping,\nlayout boxes", "model")
br = c.box(9.1, 1.7, 1.5, 0.9, "spec_bridge\n+ pose library", "model")
d_ = c.box(10.8, 1.7, 1.1, 0.9, "Descriptor\nA", "data", bold=True)
ed = c.box(6.9, 3.1, 1.9, 0.6, "UI edit: boxes, poses", "io")
c.arrow(a, b); c.arrow(b, l); c.arrow(l, nsp)
c.arrow(b, r, sa="b", sb="l", rad=0.1)
c.arrow(r, nsp, "if the LLM output\ncan't be parsed", sa="r", sb="b", rad=-0.1, lx=0.2, ly=-0.55)
c.arrow(nsp, br); c.arrow(br, d_)
c.arrow(nsp, ed, "edit", sa="t", sb="b", ls="--", lx=0.35, ly=0)
c.text(8.9, 0.5, "keypoints = k-means centres of real\nskeletons per action, placed in each\ncharacter's box", fs=7.5)
c.save("d5_planner.png")

# d6: LAM Encoder internals
c = Canvas(13, 7.2, "LAM Encoder (§IV-B): descriptor → action embeddings")
inp = [("text fields ×6\nhair/eye/outfit/…", 5.9), ("categorical\ngender·age·framing·view", 4.6), ("keypoints (18×3)\n+ bbox", 3.3), ("ordered actions\n(≤ 4 sub-actions)", 2.0), ("global text\nscene/style/props", 0.7)]
ib = [c.box(0.2, y, 2.0, 0.9, t, "io", fs=8) for t, y in inp]
bb = c.box(3.0, 5.9, 2.6, 0.9, "Transformer backbone\nSDXL CLIP-L + OpenCLIP-bigG\n(LoRA r=16), per field", "model")
em = c.box(3.0, 4.6, 2.6, 0.9, "Learned embeddings\n(label-name initialised)", "model")
po = c.box(3.0, 3.3, 2.6, 0.9, "PoseEncoder\nFourier + joint-id,\n2-layer transformer, Perceiver", "model")
dy = c.box(3.0, 2.0, 2.6, 0.9, "DynamicsEncoder\nlabels + skeleton kinematics\n+ step embeddings", "model")
gb = c.box(3.0, 0.7, 2.6, 0.9, "Backbone (same encoder)\nSDXL text path", "model")
t1 = c.box(6.4, 5.9, 2.2, 0.9, "6 fields × 6 word\ntokens (2048-d)", "data")
t2 = c.box(6.4, 4.6, 2.2, 0.9, "4 identity tokens", "data")
t3 = c.box(6.4, 3.3, 2.2, 0.9, "4 pose tokens", "data")
t4 = c.box(6.4, 2.0, 2.2, 0.9, "T dynamics tokens", "data")
tm = c.box(9.3, 2.0, 2.0, 0.9, "Temporal attention\n(2-layer, tanh gate)\n→ context vector", "model", bold=True)
fu = c.box(9.3, 3.7, 2.0, 1.4, "Style attribute\nfusion transformer\n+ Perceiver resampler\n→ residual (zero-init)", "model", bold=True)
ct = c.box(11.6, 3.7, 1.3, 1.4, "Character\ntokens\n40 × 2048\n+ token mask", "data", bold=True)
gt = c.box(6.4, 0.7, 2.2, 0.9, "Global tokens\n77 × 2048", "data")
pe = c.box(9.3, 0.7, 2.0, 0.9, "pooled 1280\n+ action_embed 768", "data")
for x, y in zip(ib, [bb, em, po, dy, gb]):
    c.arrow(x, y)
for x, y in [(bb, t1), (em, t2), (po, t3), (dy, t4), (gb, gt)]:
    c.arrow(x, y)
c.arrow(t4, tm, sa="r", sb="l"); c.arrow(t3, fu, sa="r", sb="l"); c.arrow(t2, fu, sa="r", sb="l"); c.arrow(t1, fu, sa="r", sb="t", rad=-0.1)
c.arrow(tm, fu, "+ context", sa="t", sb="b", lx=0.55)
c.arrow(fu, ct, sa="r", sb="l")
c.arrow(t1, ct, "base tokens (text scale)", sa="r", sb="t", rad=-0.35, ls="--", ly=0.25, lx=1.4)
c.arrow(gt, pe, sa="r", sb="l"); c.arrow(ct, pe, "mean → CLIP space", sa="b", sb="r", ls="--", rad=-0.3, lx=0.3)
c.text(0.2, 0.05, "character token = [identity ; field word tokens] + fused residual + temporal context       (all 2048-d, SDXL cross-attention width)", fs=8.5, style="italic")
c.save("d6_lam_encoder.png")

# d7: one cross-attention layer
c = Canvas(12, 5.0, "Dual-stream conditioning: one U-Net cross-attention layer (§IV-C)")
tg = c.box(0.2, 3.6, 2.4, 0.8, "Global text tokens\nK_g, V_g (frozen)", "frozen")
ac = c.box(0.2, 0.4, 2.4, 1.1, "Character tokens\nK_a, V_a = W_text\n+ low-rank update (r=64)", "model")
cat = c.box(3.4, 1.9, 2.2, 0.9, "concatenate\nkeys / values", "frozen")
q = c.box(6.2, 3.6, 2.4, 0.8, "Latent features → Q", "io")
sm = c.box(6.2, 1.9, 2.4, 1.0, "ONE softmax\nQ[K_g;K_a]ᵀ/√d + bias", "model", bold=True)
bi = c.box(3.4, 0.2, 3.4, 1.0, "Bias [0 ; region mask + g]\nvisible only inside the character's\nbox (+4% margin)", "model")
ou = c.box(9.4, 1.9, 1.8, 1.0, "output\n→ to_out", "data")
c.arrow(tg, cat, sa="r", sb="t", rad=-0.1); c.arrow(ac, cat, sa="r", sb="b", rad=0.1)
c.arrow(cat, sm, sa="r", sb="l"); c.arrow(q, sm, sa="b", sb="t"); c.arrow(bi, sm, sa="r", sb="b", rad=0.1)
c.arrow(sm, ou, sa="r", sb="l")
c.text(0.2, 2.15, "outside every box the layer is\nexactly the pre-trained one", fs=8, style="italic")
c.text(9.0, 0.3, "v1 added a separate attention:\nAttn(text) + gate·Attn(action)\n→ doubled signal, muddy images\n(replaced by this shared softmax)", fs=7.5, color=C["bad"])
c.save("d7_attention_layer.png")

# d8: loss computation
c = Canvas(13, 5.7, "Loss computation (§IV-E, Eq. 3)")
eps = c.box(0.2, 4.2, 1.7, 0.7, "true noise ε", "io")
ld = c.box(4.6, 4.2, 3.2, 0.7, "L_diff = ‖ε̂ − ε‖²   (λ1 = 1)", "loss")
zt = c.box(0.2, 2.8, 1.7, 0.8, "noisy latent z_t", "io")
un = c.box(2.4, 2.8, 1.8, 0.8, "U-Net → ε̂", "model")
x0 = c.box(4.6, 2.8, 2.8, 0.8, "x̂0 = (z_t − √(1−ᾱ)·ε̂)/√ᾱ\n(only for t < 400)", "data")
dec = c.box(7.8, 2.8, 1.5, 0.8, "TAESDXL\ndecoder", "frozen")
img = c.box(9.7, 2.8, 1.4, 0.8, "x̂ image", "data")
gt = c.box(11.4, 2.8, 1.4, 0.8, "ground-truth\nimage", "io")
l1 = c.box(3.4, 0.4, 3.9, 0.8, "L_action-img   (λ2 = 0.005)\nCLIP InfoNCE vs the action embedding", "loss", fs=8)
l2 = c.box(7.7, 0.4, 2.4, 0.8, "L_perc   (λ3 = 0.02)\nLPIPS (VGG)", "loss", fs=8)
l3 = c.box(10.4, 0.4, 2.5, 0.8, "L_style   (λ4 = 200)\nGram, VGG relu1–4", "loss", fs=8)
c.arrow(zt, un); c.arrow(un, ld, "ε̂", sa="t", sb="l", rad=-0.2, lx=-0.2); c.arrow(eps, ld, sa="r", sb="l")
c.arrow(un, x0); c.arrow(x0, dec); c.arrow(dec, img)
c.arrow(img, l3, sa="b", sb="t"); c.arrow(img, l2, sa="b", sb="t", rad=0.1); c.arrow(img, l1, sa="b", sb="t", rad=0.2)
c.arrow(gt, l3, sa="b", sb="t", rad=-0.1); c.arrow(gt, l2, sa="b", sb="r", rad=0.15)
c.save("d8_loss.png")

# d9: Algorithm 1
c = Canvas(13, 3.4, "Algorithm 1 (Eq. 4–6): image → actions → image")
x = c.box(0.2, 1.4, 1.4, 0.9, "image x_i", "io")
f = c.box(2.0, 1.4, 1.9, 0.9, "VAE encoder F_φ\n(frozen)", "frozen")
h = c.box(4.3, 1.4, 1.5, 0.9, "latent h_i\n(Eq. 4)", "data")
g = c.box(6.2, 1.4, 2.5, 0.9, "G_θ head\nConvNet + transformer", "model", bold=True)
a = c.box(9.1, 1.4, 1.5, 0.9, "descriptor\na_i", "data")
lm = c.box(11.0, 1.4, 1.8, 0.9, "LAM Encoder +\ndual-stream U-Net", "model", fs=8)
dd = c.box(7.2, 0.1, 5.6, 0.8, "z_{t−1} = z_t − η∇L_diff   (Eq. 5)   →   z_0   →   VAE decoder (Eq. 6)", "frozen", fs=8)
for p_, q_ in [(x, f), (f, h), (h, g), (g, a), (a, lm)]:
    c.arrow(p_, q_)
c.arrow(lm, dd, sa="b", sb="t")
c.arrow(h, dd, ls="--", sa="b", sb="l", rad=0.25)
c.text(1.9, 0.55, "η pulls the clean estimate\ntoward h_i (dashed)", fs=7.5)
c.save("d9_algorithm1.png")

# d10: data pipeline
c = Canvas(13.5, 4.6, "Dataset construction")
b1 = c.box(0.2, 3.1, 2.1, 0.9, "Danbooru metadata\n(danbooru2024-sfw)", "io")
s1 = c.box(2.8, 3.1, 1.9, 0.9, "select\nbalanced shares", "data")
s2 = c.box(5.2, 3.1, 2.1, 0.9, "fetch\nHTTP range requests", "data")
s3 = c.box(7.8, 3.1, 2.8, 0.9, "annotate\nRTMPose-x + YOLOX poses,\nWD tagger per crop", "model", fs=8)
s4 = c.box(11.0, 3.1, 2.3, 0.9, "precompute\nlatents · CLIP · GT jpg", "data")
tt = c.box(11.0, 1.7, 2.3, 0.8, "train 27,146 / test 999\n(stratified)", "data", bold=True)
h1 = c.box(0.2, 0.5, 2.4, 0.8, "HumanML3D captions\n(text only)", "io")
h2 = c.box(3.2, 0.5, 2.7, 0.8, "split on connectives\n→ ordered sub-actions", "data")
h3 = c.box(6.5, 0.5, 2.5, 0.8, "map → 78-label\naction vocabulary", "data")
for p_, q_ in [(b1, s1), (s1, s2), (s2, s3), (s3, s4), (h1, h2), (h2, h3)]:
    c.arrow(p_, q_)
c.arrow(s4, tt, sa="b", sb="t")
c.text(0.2, 2.2, "28,145 triplets · 50,677 characters (after the size filter)", fs=9, style="italic")
c.text(9.4, 0.65, "11,254 ordered multi-step sequences\n→ stage 2a (temporal supervision)", fs=8)
c.save("d10_data_pipeline.png")

# ------------------------------------------------------------------ loss / gate figures
rows = list(csv.DictReader(open(os.path.join(D, "training_logs_recovered.csv"))))
for r in rows:
    for k in ("step", "total_steps", "diff", "action_img", "perc", "style", "clip_sim", "gate_mean", "gate_max"):
        r[k] = float(r[k]) if r[k] not in ("", "None") else np.nan
RUNS = [("v2_old_checkpoint", "v2, old-design checkpoint (fails)", C["bad"]),
        ("v2_fresh_additive", "v2, fresh encoder, additive attention", C["amber"]),
        ("v3_shared_softmax", "v3, shared softmax (final design)", C["good"])]


def series(rid, key):
    pts = sorted([(r["step"], r[key]) for r in rows if r["run_id"] == rid and not math.isnan(r[key])])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for k, (key, t) in enumerate([("diff", "L_diff (ε-MSE)"), ("action_img", "L_action-img"), ("perc", "L_perc (LPIPS)"), ("clip_sim", "CLIP similarity (↑)")]):
    for rid, lab, col in RUNS:
        x, y = series(rid, key)
        ax[k].plot(x, y, "o-", ms=3.5, color=col, label=lab)
    ax[k].set_title(t); ax[k].set_xlabel("optimizer step")
ax[0].legend(fontsize=6.5)
fig.suptitle("Training loss terms in the three 16-sample overfit runs (batch 1; logged every 50 steps)", y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p1_loss_terms_overfit.png")); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for rid, lab, col in RUNS:
    x, y = series(rid, "gate_mean"); ax[0].plot(x, y, "o-", ms=3.5, color=col, label=lab)
    x, y = series(rid, "gate_max"); ax[1].plot(x, y, "o-", ms=3.5, color=col, label=lab)
ax[0].set_title("Mean action-stream gate over layers"); ax[1].set_title("Max gate over layers")
for a_ in ax: a_.set_xlabel("optimizer step")
ax[0].set_yscale("symlog", linthresh=0.01); ax[1].set_yscale("symlog", linthresh=0.01); ax[0].legend(fontsize=6.5)
fig.suptitle("Gate values: v2 gate (multiplier, init 1) vs v3 logit bias (init 0)", y=1.03)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_gate_values.png")); plt.close()

# token-norm diagnosis (values measured by lam/tests/diag_step0.py; see docs/IMPLEMENTATION.md)
labels = ["old stage2.pt\n(step 0)", "v2 fresh\n(step 0)", "v2 trained\n(300 steps)", "v3 fresh\n(step 0)"]
char_n = [55.2, 10.3, 45.0, 34.1]; text_n = [39.1, 34.2, 34.3, 34.2]
fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(4)
ax.bar(x - .2, text_n, .4, label="global text tokens (reference)", color=C["base"])
ax.bar(x + .2, char_n, .4, label="character tokens", color=[C["bad"], C["amber"], C["amber"], C["good"]])
for i, v in enumerate(char_n): ax.text(i + .2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("mean L2 norm of a token")
ax.set_title("Token scale diagnosis: v3 matches the scale the U-Net expects"); ax.legend(fontsize=8)
plt.savefig(os.path.join(FIG, "p3_token_norms.png")); plt.close()

# ------------------------------------------------------------------ scorecard
s1 = load("stage1.json"); s2a = load("stage2a.json"); s2 = load("stage2_from2a.json"); inv = load("inverse.json")[-1]
lr = load("leakage_runs.json")["results"]; lf = load("leakage_final.json")["results"]["final"]; st = load("dataset_stats.json")
n_char = st["characters"]; maj_g = max(n for _, n in st["gender"]) / n_char; maj_f = max(n for _, n in st["framing"]) / n_char
cf = load("colour_factorial.json"); HAIR, EYES = cf["hair"], cf["eyes"]
fig, ax = plt.subplots(2, 3, figsize=(13, 6.2))
a = ax[0, 0]; a.bar(["stock", "anime"], [s1["stock"]["psnr"], s1["log"][-1]["psnr"]], color=[C["base"], C["ours"]]); a.set_ylim(30, 33)
a.set_title("Stage 1 · VAE PSNR (dB, ↑)")
for i, v in enumerate([s1["stock"]["psnr"], s1["log"][-1]["psnr"]]): a.text(i, v + .05, f"{v:.2f}", ha="center")
a = ax[0, 1]; a.bar(["before", "after 2a"], [s2a["before"]["order_acc"], s2a["log"][-1]["order_acc"]], color=[C["base"], C["ours"]])
a.axhline(.5, ls="--", color="k", lw=.8); a.set_ylim(0, 1); a.set_title("Stage 2a · order accuracy (↑; chance 0.5)")
for i, v in enumerate([s2a["before"]["order_acc"], s2a["log"][-1]["order_acc"]]): a.text(i, v + .02, f"{v:.2f}", ha="center")
a = ax[0, 2]; a.bar(["global only", "stage 2"], [s2["baseline"]["R@1"], s2["log"][-1]["R@1"]], color=[C["base"], C["ours"]]); a.set_ylim(0, 1)
a.set_title("Stage 2 · descriptor→image R@1 (↑)")
for i, v in enumerate([s2["baseline"]["R@1"], s2["log"][-1]["R@1"]]): a.text(i, v + .02, f"{v:.2f}", ha="center")
a = ax[1, 0]; a.bar(["majority", "G_θ"], [maj_g, inv["gender_acc"]], color=[C["base"], C["ours"]], width=.35)
a.bar([2, 3], [maj_f, inv["framing_acc"]], color=[C["base"], C["ours"]], width=.35); a.set_xticks([0, 1, 2, 3])
a.set_xticklabels(["gender\nmajority", "gender\nG_θ", "framing\nmajority", "framing\nG_θ"]); a.set_ylim(0, 1); a.set_title("Algorithm 1 head · accuracy (↑)")
a = ax[1, 1]; xx = np.arange(2); a.bar(xx - .2, [lr["baseline"]["hair_leak"], lf["hair_leak"]], .4, label="hair leak", color=C["amber"])
a.bar(xx + .2, [lr["baseline"]["eye_leak"], lf["eye_leak"]], .4, label="eye leak", color=C["ours"]); a.set_xticks(xx)
a.set_xticklabels(["stock init", "final"]); a.set_title("Attribute leakage (↓)"); a.legend(fontsize=8)
R = cf["results"]
def acc(sys_, field):
    labs, req = (HAIR, "hair_req") if field == "hair" else (EYES, "eye_req")
    return np.mean([labs[int(np.argmax(r[field]))] == r[req] for r in R if r["system"] == sys_])
a = ax[1, 2]; xx = np.arange(2); a.bar(xx - .2, [acc("baseline", "hair"), acc("baseline", "eye")], .4, label="stock Animagine", color=C["base"])
a.bar(xx + .2, [acc("lam", "hair"), acc("lam", "eye")], .4, label="LAM-Diffusion", color=C["ours"]); a.set_xticks(xx)
a.set_xticklabels(["hair correct", "eye correct"]); a.set_ylim(0, 1.05); a.set_title("Colour factorial (↑)"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p5_scorecard.png")); plt.close()

# ------------------------------------------------------------------ statistics: bootstrap CIs + exact McNemar
rng = np.random.default_rng(0)
key = lambda r: (r["hair_req"], r["eye_req"], r["seed"])  # noqa: E731
by = {s: {key(r): r for r in R if r["system"] == s} for s in ("lam", "baseline")}
keys = sorted(by["lam"])
def correct(sys_, k):
    r = by[sys_][k]
    return (HAIR[int(np.argmax(r["hair"]))] == r["hair_req"], EYES[int(np.argmax(r["eye"]))] == r["eye_req"])
M = {s: np.array([correct(s, k) for k in keys], dtype=float) for s in by}      # [n, 2]
M = {s: np.c_[v, v[:, 0] * v[:, 1]] for s, v in M.items()}                      # hair, eye, both
names = ["hair", "eye", "both"]; n = len(keys)
stats = []
def mcnemar_exact(a, b):  # a, b binary arrays, two-sided exact binomial on discordant pairs
    b01 = int(((a == 1) & (b == 0)).sum()); b10 = int(((a == 0) & (b == 1)).sum()); m = b01 + b10
    if m == 0: return b01, b10, 1.0
    k = min(b01, b10)
    p = 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return b01, b10, min(1.0, p)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for j, nm in enumerate(names):
    for o, s in enumerate(("baseline", "lam")):
        v = M[s][:, j]; boots = np.array([v[rng.integers(0, n, n)].mean() for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stats.append({"metric": nm, "system": s, "accuracy": round(v.mean(), 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3), "n": n})
        ax.errorbar(j + (o - .5) * .3, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", capsize=4,
                    color=C["base"] if s == "baseline" else C["ours"], label=("stock Animagine" if s == "baseline" else "LAM-Diffusion") if j == 0 else None)
    lam, base = M["lam"][:, j], M["baseline"][:, j]
    b01, b10, p = mcnemar_exact(lam, base)
    idx = [rng.integers(0, n, n) for _ in range(10000)]
    diffs = np.array([(lam[i] - base[i]).mean() for i in idx]); lo, hi = np.percentile(diffs, [2.5, 97.5])
    stats.append({"metric": nm, "system": "diff (LAM − stock)", "accuracy": round(float((lam - base).mean()), 3), "ci95_low": round(lo, 3),
                  "ci95_high": round(hi, 3), "n": n, "lam_only_correct": b01, "stock_only_correct": b10, "mcnemar_p": round(p, 4)})
ax.set_xticks(range(3)); ax.set_xticklabels(["hair colour", "eye colour", "both"]); ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (95% bootstrap CI)")
ax.set_title(f"Colour factorial with confidence intervals (n = {n} paired images)"); ax.legend(fontsize=8)
plt.savefig(os.path.join(FIG, "p6_confidence_intervals.png")); plt.close()
with open(os.path.join(ANA, "stats.csv"), "w", newline="") as f:
    cols = ["metric", "system", "accuracy", "ci95_low", "ci95_high", "n", "lam_only_correct", "stock_only_correct", "mcnemar_p"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(stats)

# ------------------------------------------------------------------ LaTeX tables
g = {(s["metric"], s["system"]): s for s in stats}
tex = [r"% generated by reports/scripts/make_paper_figs.py", r"\begin{table}[t]\centering", r"\caption{Colour factorial: accuracy of the requested colour (CLIP zero-shot reader), 128 paired images, 95\% bootstrap CI.}",
       r"\begin{tabular}{lccc}\toprule", r"System & Hair & Eye & Both \\ \midrule"]
for s, nm in (("baseline", "Stock Animagine XL"), ("lam", "LAM-Diffusion (ours)")):
    tex.append(nm + " & " + " & ".join(f"{g[(m, s)]['accuracy']*100:.1f} [{g[(m, s)]['ci95_low']*100:.1f}, {g[(m, s)]['ci95_high']*100:.1f}]" for m in names) + r" \\")
tex += [r"\bottomrule\end{tabular}\end{table}", "",
        r"\begin{table}[t]\centering", r"\caption{Per-stage results.}", r"\begin{tabular}{llcc}\toprule", r"Stage & Metric & Before & After \\ \midrule",
        f"1 anime VAE & PSNR (dB) / LPIPS & {s1['stock']['psnr']:.2f} / {s1['stock']['lpips']:.4f} & {s1['log'][-1]['psnr']:.2f} / {s1['log'][-1]['lpips']:.4f} \\\\",
        f"2a action-text & order accuracy & {s2a['before']['order_acc']:.3f} & {s2a['log'][-1]['order_acc']:.3f} \\\\",
        f"2a action-text & caption R@1 & {s2a['before']['caption_R@1']:.3f} & {s2a['log'][-1]['caption_R@1']:.3f} \\\\",
        f"2 triplets & descriptor$\\to$image R@1 / R@10 & {s2['baseline']['R@1']:.3f} / {s2['baseline']['R@10']:.3f} & {s2['log'][-1]['R@1']:.3f} / {s2['log'][-1]['R@10']:.3f} \\\\",
        f"Full U-Net & hair / eye leak & {lr['baseline']['hair_leak']} / {lr['baseline']['eye_leak']} & {lf['hair_leak']} / {lf['eye_leak']} \\\\",
        r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(ANA, "tables.tex"), "w").write("\n".join(tex) + "\n")
for s in stats: print(s)
print("figures:", sorted(f for f in os.listdir(FIG) if f[:2] in ("d4", "d5", "d6", "d7", "d8", "d9", "d1", "p1", "p2", "p3", "p5", "p6")))
