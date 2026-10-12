"""Results analysis: hypothesis tests, heat maps, confusion matrices, confidence thresholds, per-colour patterns,
comparison tables. Everything is computed from the logged data in reports/data/ (no GPU needed).

    python reports/scripts/make_analysis.py      ->  reports/analysis/{*.png, *.csv, tables.md}
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "analysis")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
NAME = {"lam": "LAM-Diffusion (trained)", "baseline": "Stock Animagine XL"}
load = lambda f: json.load(open(os.path.join(D, f)))  # noqa: E731

cf = load("colour_factorial.json")
HAIR, EYES, R = cf["hair"], cf["eyes"], cf["results"]
SYS = ["lam", "baseline"]


def rows(system):
    return [r for r in R if r["system"] == system]


def pred(probs, labels):
    return labels[int(np.argmax(probs))]


# ---------------------------------------------------------------- core arrays
joint, hair_ok, eye_ok, conf_hair, conf_eye = {}, {}, {}, {}, {}
for s in SYS:
    J = np.zeros((8, 8)); H = np.zeros((8, 8)); E = np.zeros((8, 8)); N = np.zeros((8, 8))
    CH = np.zeros((8, 8)); CE = np.zeros((8, 8))  # confusion: requested -> predicted
    for r in rows(s):
        i, j = HAIR.index(r["hair_req"]), EYES.index(r["eye_req"])
        hp, ep = pred(r["hair"], HAIR), pred(r["eye"], EYES)
        H[i, j] += hp == r["hair_req"]; E[i, j] += ep == r["eye_req"]
        J[i, j] += (hp == r["hair_req"]) and (ep == r["eye_req"]); N[i, j] += 1
        CH[i, HAIR.index(hp)] += 1; CE[j, EYES.index(ep)] += 1
    joint[s], hair_ok[s], eye_ok[s] = J / N, H / N, E / N
    conf_hair[s], conf_eye[s] = CH, CE


def heat(ax, M, xl, yl, title, cmap, vmin=0, vmax=1, fmt="{:.0%}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=45, ha="right")
    ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl); ax.set_title(title)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            v = M[a, b]
            ax.text(b, a, fmt.format(v), ha="center", va="center", fontsize=6.5,
                    color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 or (cmap == "RdBu" and abs(v) > 0.6) else "black")
    return im


# A1: joint-accuracy heat maps (requested hair x requested eye), both systems + difference
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for k, s in enumerate(SYS):
    im = heat(ax[k], joint[s], EYES, HAIR, f"{NAME[s]}\nboth colours correct (2 seeds)", "viridis")
    ax[k].set_xlabel("requested eye colour")
ax[0].set_ylabel("requested hair colour")
fig.colorbar(im, ax=ax[:2], fraction=0.025)
im = heat(ax[2], joint["lam"] - joint["baseline"], EYES, HAIR, "Difference (LAM − stock)", "RdBu", -1, 1, "{:+.0%}")
ax[2].set_xlabel("requested eye colour"); fig.colorbar(im, ax=ax[2], fraction=0.045)
plt.savefig(f"{OUT}/a1_heatmap_joint_accuracy.png"); plt.close()

# A2: hair-colour confusion matrices (requested -> what CLIP sees)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for k, s in enumerate(SYS):
    M = conf_hair[s] / conf_hair[s].sum(1, keepdims=True)
    heat(ax[k], M, HAIR, HAIR, f"{NAME[s]}: hair colour confusion", "magma")
    ax[k].set_xlabel("predicted hair colour"); ax[k].set_ylabel("requested hair colour")
plt.savefig(f"{OUT}/a2_confusion_hair.png"); plt.close()

# A3: eye-colour confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for k, s in enumerate(SYS):
    M = conf_eye[s] / conf_eye[s].sum(1, keepdims=True)
    heat(ax[k], M, EYES, EYES, f"{NAME[s]}: eye colour confusion", "magma")
    ax[k].set_xlabel("predicted eye colour"); ax[k].set_ylabel("requested eye colour")
plt.savefig(f"{OUT}/a3_confusion_eye.png"); plt.close()

# A4: per-colour accuracy (patterns)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); w = 0.38
for k, (labels, ok, axis, nm) in enumerate([(HAIR, hair_ok, 1, "hair"), (EYES, eye_ok, 0, "eye")]):
    x = np.arange(8)
    for o, s in enumerate(SYS):
        ax[k].bar(x + (o - .5) * w, ok[s].mean(axis=axis), w, label=NAME[s])
    ax[k].set_xticks(x); ax[k].set_xticklabels(labels, rotation=30); ax[k].set_ylim(0, 1.05)
    ax[k].set_title(f"{nm.capitalize()}-colour accuracy by requested colour"); ax[k].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/a4_per_colour_accuracy.png"); plt.close()

# A5: confidence threshold analysis — precision/coverage vs CLIP probability threshold on the requested colour
taus = np.linspace(0, 0.95, 39)
thr_rows = []
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for k, (field, labels, key) in enumerate([("hair", HAIR, "hair_req"), ("eye", EYES, "eye_req")]):
    for s in SYS:
        p = np.array([r[field][labels.index(r[key])] for r in rows(s)])          # prob of requested colour
        ok = np.array([pred(r[field], labels) == r[key] for r in rows(s)])
        acc_at = [((p >= t) & ok).mean() for t in taus]                           # correct AND confident
        ax[k].plot(taus, acc_at, label=NAME[s])
        for t in (0.3, 0.5, 0.7, 0.9):
            thr_rows.append({"field": field, "system": s, "threshold": t,
                             "share_correct_and_confident": round(float(((p >= t) & ok).mean()), 3),
                             "mean_prob_requested": round(float(p.mean()), 3)})
    ax[k].set_xlabel("CLIP probability threshold τ on the requested colour")
    ax[k].set_ylabel("share of images: correct and p ≥ τ"); ax[k].set_title(f"{field.capitalize()}: confidence threshold")
    ax[k].legend(fontsize=7); ax[k].set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/a5_threshold_curves.png"); plt.close()

# A6: distribution of the probability CLIP gives the requested colour
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
for k, (field, labels, key) in enumerate([("hair", HAIR, "hair_req"), ("eye", EYES, "eye_req")]):
    for s in SYS:
        p = [r[field][labels.index(r[key])] for r in rows(s)]
        ax[k].hist(p, bins=20, range=(0, 1), alpha=.55, label=NAME[s])
    ax[k].set_title(f"{field.capitalize()}: CLIP probability of the requested colour"); ax[k].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/a6_confidence_hist.png"); plt.close()

# A7: leakage across runs (stock-init -> pilots -> final)
lr = load("leakage_runs.json")["results"]; lf = load("leakage_final.json")["results"]["final"]
runs = [("Stock init", lr["baseline"]), ("LoRA pilot", lr["lora_noswap"]), ("LoRA + swap loss", lr["lora_swap"]), ("Final full U-Net", lf)]
fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(runs))
ax.bar(x - .2, [r[1]["hair_leak"] for r in runs], .4, label="hair leak"); ax.bar(x + .2, [r[1]["eye_leak"] for r in runs], .4, label="eye leak")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in runs]); ax.set_ylabel("accuracy drop when the other colour is swapped")
ax.set_title("Attribute leakage (40 held-out specs; lower is better)"); ax.legend()
plt.savefig(f"{OUT}/a7_leakage_runs.png"); plt.close()

# ---------------------------------------------------------------- tables
summ = []
for s in SYS:
    rr = rows(s)
    ph = np.mean([r["hair"][HAIR.index(r["hair_req"])] for r in rr]); pe = np.mean([r["eye"][EYES.index(r["eye_req"])] for r in rr])
    summ.append({"system": NAME[s], "n": len(rr), "hair_acc": round(float(hair_ok[s].mean()), 3),
                 "eye_acc": round(float(eye_ok[s].mean()), 3), "both_acc": round(float(joint[s].mean()), 3),
                 "mean_p_hair_requested": round(float(ph), 3), "mean_p_eye_requested": round(float(pe), 3),
                 "distinct_hair_colours_produced": int((conf_hair[s].sum(0) > 0).sum())})


def write_csv(name, data):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


write_csv("t1_colour_factorial_summary", summ)
write_csv("t2_per_colour", [{"colour_field": fld, "colour": c, **{f"{s}_acc": round(float(ok[s].mean(axis=ax_)[i]), 3) for s in SYS}}
                             for fld, labels, ok, ax_ in [("hair", HAIR, hair_ok, 1), ("eye", EYES, eye_ok, 0)] for i, c in enumerate(labels)])
write_csv("t3_threshold", thr_rows)
for s in SYS:
    write_csv(f"t4_confusion_hair_{s}", [{"requested": HAIR[i], **{HAIR[j]: int(conf_hair[s][i, j]) for j in range(8)}} for i in range(8)])

st = load("dataset_stats.json"); n_char = st["characters"]
maj_gender = max(n for _, n in st["gender"]) / n_char; maj_framing = max(n for _, n in st["framing"]) / n_char
inv = load("inverse.json")[-1]; s1 = load("stage1.json"); s2a = load("stage2a.json"); s2 = load("stage2_from2a.json")
stage_rows = [
    {"stage": "1 anime VAE", "metric": "PSNR dB (LPIPS)", "before": f"{s1['stock']['psnr']:.2f} ({s1['stock']['lpips']:.4f})", "after": f"{s1['log'][-1]['psnr']:.2f} ({s1['log'][-1]['lpips']:.4f})"},
    {"stage": "2a action-text", "metric": "sub-action order acc.", "before": f"{s2a['before']['order_acc']:.3f}", "after": f"{s2a['log'][-1]['order_acc']:.3f}"},
    {"stage": "2a action-text", "metric": "caption R@1 (1,000)", "before": f"{s2a['before']['caption_R@1']:.3f}", "after": f"{s2a['log'][-1]['caption_R@1']:.3f}"},
    {"stage": "2 triplets", "metric": "descriptor→image R@1 / R@10", "before": f"{s2['baseline']['R@1']:.3f} / {s2['baseline']['R@10']:.3f}", "after": f"{s2['log'][-1]['R@1']:.3f} / {s2['log'][-1]['R@10']:.3f}"},
    {"stage": "Algorithm 1 G_θ", "metric": "gender acc (majority baseline)", "before": f"{maj_gender:.3f}", "after": f"{inv['gender_acc']:.3f}"},
    {"stage": "Algorithm 1 G_θ", "metric": "framing acc (majority baseline)", "before": f"{maj_framing:.3f}", "after": f"{inv['framing_acc']:.3f}"},
    {"stage": "3-5 full U-Net", "metric": "hair / eye leak", "before": f"{lr['baseline']['hair_leak']} / {lr['baseline']['eye_leak']}", "after": f"{lf['hair_leak']} / {lf['eye_leak']}"},
]
write_csv("t5_stage_results", stage_rows)

# markdown version of the key tables
md = ["# Analysis tables (generated by reports/scripts/make_analysis.py)\n", "## Colour factorial summary\n",
      "| " + " | ".join(summ[0]) + " |", "|" + "---|" * len(summ[0])]
md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in summ]
md += ["\n## Stage results\n", "| stage | metric | before | after |", "|---|---|---|---|"]
md += [f"| {r['stage']} | {r['metric']} | {r['before']} | {r['after']} |" for r in stage_rows]
md += ["\n## Confidence thresholds (share of images with the requested colour predicted AND p ≥ τ)\n",
       "| field | system | τ=0.3 | τ=0.5 | τ=0.7 | τ=0.9 |", "|---|---|---|---|---|---|"]
for fld in ("hair", "eye"):
    for s in SYS:
        v = {r["threshold"]: r["share_correct_and_confident"] for r in thr_rows if r["field"] == fld and r["system"] == s}
        md.append(f"| {fld} | {NAME[s]} | {v[0.3]} | {v[0.5]} | {v[0.7]} | {v[0.9]} |")
open(f"{OUT}/tables.md", "w").write("\n".join(md) + "\n")
print("summary:", summ)
print("written:", sorted(os.listdir(OUT)))
