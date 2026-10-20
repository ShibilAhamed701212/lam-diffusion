"""Report figures that need only logged data (stage 1/2a/2, inverse head, dataset stats, leakage runs)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data"); F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)
C = {"ours": "#2a6fdb", "base": "#9aa3ad", "warn": "#d9534f", "good": "#2e9e6b", "amber": "#e0a030", "ink": "#222"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight"})
load = lambda n: json.load(open(os.path.join(D, n)))

# ---- F1 dataset composition
st = load("dataset_stats.json")
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
comp = dict(st["composition"]); names = {"solo_m": "1 boy", "solo_f": "1 girl", "duo_mf": "boy+girl", "trio": "3 chars", "duo_ff": "2 girls", "duo_mm": "2 boys"}
ax[0].bar([names[k] for k in comp], list(comp.values()), color=C["ours"]); ax[0].set_title("Images by composition (n=%d)" % st["images"])
ax[0].tick_params(axis="x", rotation=30)
fr = dict(st["framing"]); ax[1].bar(list(fr), list(fr.values()), color=C["amber"]); ax[1].set_title("Characters by framing (n=%d)" % st["characters"])
ax[1].tick_params(axis="x", rotation=20)
fc = st["field_coverage"]; ax[2].barh(list(fc)[::-1], [v * 100 for v in list(fc.values())[::-1]], color=C["good"]); ax[2].set_xlim(0, 100)
ax[2].set_title("Attribute-field coverage (% of characters)")
for i, v in enumerate(list(fc.values())[::-1]): ax[2].text(v * 100 + 1, i, f"{v*100:.0f}%", va="center")
plt.tight_layout(); plt.savefig(f"{F}/f1_dataset.png"); plt.close()

# ---- F2 action frequency
fa = [x for x in st["first_action"] if x[0] != "<none>"][:14]
fig, ax = plt.subplots(figsize=(7, 4)); ax.barh([a for a, _ in fa][::-1], [n for _, n in fa][::-1], color=C["ours"])
ax.set_title("Most frequent first sub-action (characters)"); plt.savefig(f"{F}/f2_actions.png"); plt.close()

# ---- F3 stage 1 VAE
s1 = load("stage1.json"); steps = [r["step"] for r in s1["log"]]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(steps, [r["psnr"] for r in s1["log"]], "o-", c=C["ours"]); ax[0].axhline(s1["stock"]["psnr"], c=C["base"], ls="--", label="stock SDXL VAE")
ax[0].set_title("Stage 1: PSNR (higher better)"); ax[0].set_xlabel("step"); ax[0].legend()
ax[1].plot(steps, [r["lpips"] for r in s1["log"]], "o-", c=C["ours"]); ax[1].axhline(s1["stock"]["lpips"], c=C["base"], ls="--", label="stock SDXL VAE")
ax[1].set_title("Stage 1: LPIPS (lower better)"); ax[1].set_xlabel("step"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{F}/f3_stage1.png"); plt.close()

# ---- F4 stage 2a
s2a = load("stage2a.json"); b = s2a["before"]; L = s2a["log"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); xs = [0] + [r["step"] for r in L]
ax[0].plot(xs, [b["caption_R@1"]] + [r["caption_R@1"] for r in L], "o-", c=C["ours"]); ax[0].set_title("Stage 2a: caption retrieval R@1"); ax[0].set_xlabel("step")
ax[1].plot(xs, [b["order_acc"]] + [r["order_acc"] for r in L], "o-", c=C["good"]); ax[1].axhline(.5, c=C["base"], ls="--", label="chance (0.5)")
ax[1].set_title("Stage 2a: temporal order accuracy"); ax[1].set_xlabel("step"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{F}/f4_stage2a.png"); plt.close()

# ---- F5 stage 2 retrieval
s2 = load("stage2_from2a.json"); L = s2["log"]; bl = s2["baseline"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); st_ = [r["step"] for r in L]
ax[0].plot(st_, [r["R@1"] for r in L], "o-", c=C["ours"], label="R@1"); ax[0].plot(st_, [r["R@10"] for r in L], "s-", c=C["good"], label="R@10")
ax[0].axhline(bl["R@1"], c=C["base"], ls="--", label="global-descriptor-only R@1"); ax[0].axhline(bl["R@10"], c=C["base"], ls=":", label="global-only R@10")
ax[0].set_title("Stage 2: descriptor→image retrieval (test, n=999)"); ax[0].set_xlabel("step"); ax[0].legend(fontsize=8)
ax[1].plot(st_, [r["median_rank"] for r in L], "o-", c=C["warn"]); ax[1].axhline(bl["median_rank"], c=C["base"], ls="--", label=f"baseline {bl['median_rank']:.0f}")
ax[1].set_yscale("log"); ax[1].set_title("Median rank of true image (lower better)"); ax[1].set_xlabel("step"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{F}/f5_stage2.png"); plt.close()

# ---- F6 inverse head
inv = load("inverse.json"); ep = [r["epoch"] for r in inv]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for k, lab in (("present_acc", "character present"), ("gender_acc", "gender"), ("age_acc", "age"), ("framing_acc", "framing"), ("view_acc", "view")):
    ax[0].plot(ep, [r[k] for r in inv], "o-", label=lab, ms=3)
ax[0].set_title("Inverse head G_θ (Alg. 1): accuracies"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
ax[1].plot(ep, [r["bbox_l1"] for r in inv], "o-", c=C["warn"], label="bbox L1 (↓)"); ax[1].plot(ep, [r["action_f1"] for r in inv], "s-", c=C["good"], label="action F1 (↑)")
ax[1].set_title("Inverse head: box error and action F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{F}/f6_inverse.png"); plt.close()

# ---- F7 leakage progression (40 test specs, CLIP zero-shot)
runs = ["Stock-init\n(encoder only)", "LoRA pilot\n(~250 steps)", "LoRA + swap loss\n(~150 steps)", "Final full U-Net\n(11 000 steps)"]
hair = [0.125, 0.025, 0.05, 0.05]; eye = [0.375, 0.175, 0.25, 0.10]
import numpy as np
x = np.arange(len(runs)); w = .36
fig, ax = plt.subplots(figsize=(8.2, 3.8))
b1 = ax.bar(x - w/2, hair, w, label="hair-colour leak (eye swapped → hair acc. drop)", color=C["ours"])
b2 = ax.bar(x + w/2, eye, w, label="eye-colour leak (hair swapped → eye acc. drop)", color=C["amber"])
for r in list(b1) + list(b2): ax.text(r.get_x() + r.get_width()/2, r.get_height() + .008, f"{r.get_height():.3f}", ha="center", fontsize=9)
ax.axhline(0.10, c=C["warn"], ls="--", lw=1); ax.text(3.45, 0.105, "threshold 0.10", color=C["warn"], ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(runs); ax.set_ylabel("accuracy drop"); ax.set_title("Attribute leakage across training stages (n=40 test specs)")
ax.legend(fontsize=8, loc="upper right"); plt.savefig(f"{F}/f7_leakage_progress.png"); plt.close()

# ---- F8 training throughput (measured: A100 by lam/tests/smoke_full.py at batch 1; H200 from the full-run log)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
cfgs = ["A100 40 GB\nmeasured 1.87 s per batch-1 step\n(× 4 for effective batch 4)", "H200, full run\nmeasured from the training log\nbatch 4 × accum 1"]
sec = [4 * 1.87, 4.60 * 3600 / 11000]
bars = ax.bar(cfgs, sec, color=[C["base"], C["ours"]])
for r, v, h in zip(bars, sec, ["extrapolated", "measured"]):
    ax.text(r.get_x() + r.get_width() / 2, v + .15, f"{v:.2f} s/step ({h})\n≈ {v*11000/3600:.1f} h / 11k steps", ha="center", fontsize=8)
ax.set_ylim(0, 10); ax.set_ylabel("seconds per optimizer step (effective batch 4)"); ax.set_title("Full-U-Net training throughput")
plt.savefig(f"{F}/f8_throughput.png"); plt.close()
print("figures:", sorted(os.listdir(F)))
