"""Training-dynamics analysis of the full 11,000-step run (from reports/data/full_run_phaseB.csv).

    python reports/scripts/make_training_analysis.py
Writes reports/analysis/{training_trends.csv, training_stages.csv, training_timing.csv} and the p7/p8 figures.
Everything is computed from the logged 50-step window means; nothing is estimated.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
D, ANA, FIG = (os.path.join(HERE, "..", x) for x in ("data", "analysis", "figures"))
MIL = os.path.join(HERE, "..", "milestones")
os.makedirs(ANA, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .25})
rows = list(csv.DictReader(open(os.path.join(D, "full_run_phaseB.csv"))))
col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
step, hours, pool = col("step"), col("hours"), col("curriculum_pool")
METRICS = [("diff", "L_diff"), ("action_img", "L_action-img"), ("perc", "L_perc"), ("style", "L_style"), ("clip_sim", "CLIP similarity")]
rng = np.random.default_rng(0)


def spearman_perm(x, y, n=10000):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rx, ry[rng.permutation(len(ry))])[0, 1] for _ in range(n)])
    return r, float((np.abs(null) >= abs(r)).mean())


# ---------------------------------------------------------------- trends: whole run and after the warm-up phase
trend = []
for k, name in METRICS:
    y = col(k)
    for label, m in (("all steps", step >= 0), ("after step 1000", step > 1000)):
        x, yy = step[m], y[m]
        slope = np.polyfit(x / 1000.0, yy, 1)[0]
        boots = []
        for _ in range(4000):
            i = rng.integers(0, len(x), len(x))
            boots.append(np.polyfit(x[i] / 1000.0, yy[i], 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rho, p = spearman_perm(x, yy)
        rel = slope * (x.max() - x.min()) / 1000.0 / abs(yy.mean())
        trend.append({"metric": name, "range": label, "mean": float(yy.mean()), "sd_across_records": float(yy.std()),
                      "slope_per_1000_steps": float(slope), "slope_ci95_low": float(lo), "slope_ci95_high": float(hi),
                      "spearman_rho": float(rho), "perm_p": p, "relative_change_over_range": float(rel)})
with open(os.path.join(ANA, "training_trends.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(trend[0])); w.writeheader(); w.writerows(trend)

# ---------------------------------------------------------------- per curriculum stage
stages = []
for lo, hi, nm in ((0, 3300, "stage A: easiest 40%"), (3301, 7700, "stage B: easiest 80%"), (7701, 11000, "stage C: all samples")):
    m = (step >= lo) & (step <= hi)
    s = {"stage": nm, "steps": f"{int(step[m].min())}-{int(step[m].max())}", "records": int(m.sum())}
    for k, name in METRICS:
        s[name + "_mean"] = float(col(k)[m].mean()); s[name + "_sd"] = float(col(k)[m].std())
    stages.append(s)
with open(os.path.join(ANA, "training_stages.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stages[0])); w.writeheader(); w.writerows(stages)

# ---------------------------------------------------------------- timing from the logged wall-clock
dt = np.diff(np.r_[0.0, hours]) * 3600.0 / 50.0          # seconds per step in each 50-step window
spikes = dt > np.median(dt) * 1.15
per_1000 = []
for a in range(0, 11000, 1000):
    m = (step > a) & (step <= a + 1000)
    per_1000.append({"steps": f"{a+1}-{a+1000}", "seconds_per_step": float(dt[m].mean()), "minutes_for_1000_steps": float(dt[m].mean() * 1000 / 60)})
timing = {"overall_s_per_step": float(hours[-1] * 3600 / step[-1]), "median_window_s_per_step": float(np.median(dt)),
          "p95_window_s_per_step": float(np.percentile(dt, 95)), "windows_over_1.15x_median": int(spikes.sum()),
          "total_hours": float(hours[-1]), "extra_seconds_in_slow_windows": float(((dt - np.median(dt))[spikes] * 50).sum())}
with open(os.path.join(ANA, "training_timing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["steps", "seconds_per_step", "minutes_for_1000_steps"]); w.writeheader(); w.writerows(per_1000)

for t in trend:
    print(f"{t['metric']:15s} {t['range']:16s} mean {t['mean']:.4g} slope/1k {t['slope_per_1000_steps']:+.3g} "
          f"[{t['slope_ci95_low']:+.3g},{t['slope_ci95_high']:+.3g}] rho {t['spearman_rho']:+.2f} p {t['perm_p']:.4f} rel {t['relative_change_over_range']:+.1%}")
for s in stages:
    print(s["stage"], s["steps"], {k: round(v, 4) for k, v in s.items() if k.endswith("_mean")})
print("timing", {k: round(v, 3) for k, v in timing.items()})

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(step, dt, color="#9ca3af", lw=.8); ax[0].axhline(np.median(dt), color="#2563eb", ls="--", label=f"median {np.median(dt):.2f} s/step")
ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("seconds per step (50-step window)"); ax[0].set_title("Throughput over the run"); ax[0].legend(fontsize=8)
ax[1].hist(dt, bins=30, color="#2563eb", alpha=.8); ax[1].set_xlabel("seconds per step"); ax[1].set_title("Window throughput distribution (slow tail = checkpoints, sample grids)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p8_throughput_over_run.png")); plt.close()

picks = [750, 3000, 6000, 9000, 11000]
ims = [Image.open(os.path.join(MIL, "full_run_samples", f"step{s:06d}.jpg")).convert("RGB") for s in picks]
w = 1500; ims = [im.resize((w, round(im.height * w / im.width))) for im in ims]
sheet = Image.new("RGB", (w + 120, sum(i.height for i in ims) + 6 * len(ims)), "white"); d = ImageDraw.Draw(sheet); y = 0
for s, im in zip(picks, ims):
    d.text((8, y + im.height // 2 - 6), f"step {s:,}", fill=(0, 0, 0)); sheet.paste(im, (120, y)); y += im.height + 6
sheet.save(os.path.join(FIG, "p7_full_run_progression.png"))
print("figures written")
