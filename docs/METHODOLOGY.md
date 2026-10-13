# Methodology report

How this project turned the LAM-Diffusion paper into a trained, evaluated system: the research questions, data,
model, training, evaluation and validation methods, and the limits of the evidence. Implementation detail lives in
the linked docs; this report explains *how* and *why*.

## 1. Problem and objective

Text-to-image diffusion models read a prompt as one undivided text. Attributes get mixed between characters
("silver hair, red eyes" → red hair), and pose, count and layout are only weakly controlled. The LAM-Diffusion
paper proposes putting a **Large Action Model** between the prompt and the image model:

```
A = F_LAM(N)       the prompt N becomes structured action descriptors A
I = G_diff(A)      the image is generated from A, not from the raw prompt
```

The paper's own experiments (§V) only used a rule-based planner with stock Stable Diffusion, inference only. Its
§IV architecture (learned encoder, dual-stream conditioning, trained U-Net, multi-term loss, five training stages)
was never built or trained.

**Objective:** implement the §IV architecture as specified, train it end to end, and measure whether it delivers
the controllability the paper claims.

## 2. Research questions

| RQ | Question | Hypotheses ([ANALYSIS.md](ANALYSIS.md#1-hypotheses-and-verdicts)) |
|---|---|---|
| RQ1 | Can each component of §IV be trained to do its job? | H3 (temporal order), H4 (descriptor–image alignment), H5 (anime VAE), H6 (Algorithm 1 head) |
| RQ2 | Does structured, per-character conditioning reduce attribute leakage? | H2 |
| RQ3 | Does the full system bind attributes better than a prompt-only diffusion model? | H1, H7, H8 |

## 3. Research approach

A **design-and-evaluate** approach:

1. **Specify:** map every element of the paper to a component (Eq. 1–6, §IV-B–F), and record interpretation choices and deviations explicitly ([IMPLEMENTATION.md](IMPLEMENTATION.md)).
2. **Build:** implement it on a strong pre-trained anime backbone (Animagine XL 4.0), so the experiment tests the paper's *conditioning* idea, not image quality from scratch.
3. **Gate each stage:** every stage must pass a measurable gate before the next one consumes compute (§7).
4. **Evaluate against a baseline:** the same pre-trained model driven by the prompt alone, with the same seeds and sampler.
5. **Report everything,** including failed designs and negative results.

## 4. Data methodology

Full detail: [DATASET.md](DATASET.md).

| Step | Method | Rationale |
|---|---|---|
| Source | Danbooru posts (`danbooru2024-sfw`), keeping only the tagger's "general" rating | Large, tagged, anime-specific. The paper calls for an anime action-pose-text dataset but publishes none. |
| Sampling | **Stratified** selection: fixed composition shares (solo / duo / trio, gender mix), boosted rare ages, inverse-√frequency posture weights | Raw Danbooru is dominated by "solo young girl standing", which caused gender and age errors in early prototypes |
| Pose labels | RTMPose-x + YOLOX (HumanArt) → 18 OpenPose keypoints per person; *n* largest skeletons, ordered left to right | Detectors trained on illustrations work on anime |
| Attribute labels | Solo images: human-written tags. Multi-character images: WD EVA02 tagger on each person's crop. | Per-crop tagging binds attributes to the right character |
| Filtering | Every character ≥ 20% of frame height, the largest ≥ 35% | Tiny figures carry little pose or attribute signal |
| Split | 27,146 train / 999 test, stratified by composition | Every scenario is evaluated |
| Action-text corpus | HumanML3D captions split into ordered sub-actions and mapped onto the same 78-label vocabulary | Gives real *ordered* action supervision that still images can't provide |
| Quality control | A stats sheet plus visual grids of random triplets before training | Catches mapping errors early |

## 5. Model methodology

Full detail: [ARCHITECTURE.md](ARCHITECTURE.md).

- **Planner** (Eq. 1): an instruction-tuned LLM (Qwen2.5-3B) with a rule-based fallback, emitting a strict JSON spec.
- **LAM Encoder** (§IV-B): SDXL's CLIP text transformers + pose, action-dynamics and style-fusion modules → per-character action tokens.
- **Dual-stream conditioning** (§IV-C): character tokens join the text tokens in one shared, box-masked attention in every cross-attention layer, plus a temporal attention module over ordered sub-actions.
- **Backbone** (§IV-D): Animagine XL 4.0. The U-Net is fully trained, the VAE encoder frozen, and the decoder anime-tuned.
- **Loss** (§IV-E): Eq. 3 = ε-MSE + action-image (CLIP InfoNCE) + LPIPS + Gram style.

**Design decisions were driven by evidence, not taste.** The first conditioning design (action attention *added*
to text attention, with averaged attribute vectors) failed its overfit gate. Step-0 diagnosis isolated the causes
(§7), and the **v3** design (shared softmax, full attribute token sequences) replaced it.

## 6. Training methodology

Full detail: [TRAINING.md](TRAINING.md).

| Stage | Objective | Key settings | Gate |
|---|---|---|---|
| 1 | Anime VAE decoder | MSE + 0.1·LPIPS, EMA, 6k steps, encoder frozen | Beat the stock decoder (PSNR and LPIPS) on held-out crops |
| 2a | Action-text pre-training | Caption contrastive + sequence↔caption + order classification + keep loss, 2.5k steps | Order accuracy well above chance |
| 2 | Descriptor ↔ image alignment | Contrastive vs CLIP image embeddings, 3k steps | Retrieval clearly above the global-descriptor-only baseline |
| 3–5 | Joint U-Net + encoder | Full U-Net, LR 1e-5, plain ε-MSE + aux losses (λ 0.005 / 0.02 / 200), curriculum 40→80→100% of samples, 10% CFG dropout + 10% per-field dropout, 11k steps, effective batch 4 | 16-sample overfit reproduces its training specs; sample grids stay clean |
| Algorithm 1 | Image latent → descriptor | ConvNet + transformer, 12 epochs | Above majority-class accuracy |

**Hyperparameter selection:** the auxiliary loss weights were chosen from short pilots. The original weights
produced swirly textures within 200 steps; 10× lower weights and diffusion-only both stayed clean.

**Compute:** Colab L4 and A100 for the data and early stages, a single H200 for the full run (1.51 s/step,
4.60 h, measured from the training log), and Kaggle T4s for inference and evaluation. Every long job is resumable from checkpoints.

## 7. Validation and debugging methodology

A fixed escalation ladder, cheapest first:

1. **Unit / smoke tests** (`lam/tests/smoke*.py`): tensor shapes; "action stream off = exactly the base model"; region masks match boxes at every resolution; losses are finite with gradients in the right places; a full-U-Net step fits in memory.
2. **Step-0 diagnosis** (`diag_step0.py`): render fixed specs with the *untrained* model, across variants (fresh vs old checkpoint, action on/off, masks on/off), and measure token norms against text-token norms.
3. **Overfit gate:** 16 training triplets, a few hundred steps. The model must reproduce them.
4. **Pilot runs:** short runs to set loss weights.
5. **Full run** with periodic sample grids.

This ladder found the failure of the first design. It traced the problem to three separate causes (a stale
checkpoint, additive attention and a CFG-amplified pooled shift) before any large compute was spent.

## 8. Evaluation methodology

| Protocol | What it measures | How |
|---|---|---|
| VAE reconstruction | Stage 1 quality | PSNR / LPIPS on held-out 512² anime crops, stock vs fine-tuned |
| Order classification | Temporal module (RQ1) | True vs shuffled sub-action order on 2,000 HumanML3D test sequences |
| Retrieval | Descriptor–image alignment (RQ1) | Descriptor → image R@1 / R@10 / median rank on 999 test triplets, vs a global-descriptor-only baseline |
| **Swap test (leakage)** | RQ2 | 40 single-character test specs; render as specified, eye-swapped and hair-swapped; a CLIP zero-shot colour reader on the character crop; leak = drop in the *other* attribute's accuracy |
| **Colour factorial** | RQ3 | 8 hair × 8 eye colours × 2 seeds, trained model vs stock Animagine (prompt only, same seed); CLIP reads both colours; accuracy, confusion, confidence thresholds |
| Qualitative sheets | All RQs | Fixed held-out specs: as specified / eye swapped / hair swapped / action off; side-by-side vs stock |
| §V-C suite (implemented) | Paper metrics | CLIP Score, FID/KID, LPIPS, SSIM, binding precision/recall, pose PCK, ablations (`lam/eval/`). Not yet run on the final weights. |

**Controls:** identical seeds, sampler (Euler-ancestral), steps, CFG scale and negative prompt for every system
being compared. The baseline shares the pre-trained backbone, so differences come from the conditioning and the
training.

## 9. Reproducibility

- **Code:** every stage, test and analysis script is in this repo. Figures and tables regenerate with `reports/scripts/make_figs_1.py` and `make_analysis.py`.
- **Data:** the descriptors and the precomputed training set are on Kaggle; the dataset can also be rebuilt from scratch with `lam/data/`.
- **Weights:** the final model, the anime VAE, the stage checkpoints and the Algorithm 1 head are on Kaggle ([ARTIFACTS.md](ARTIFACTS.md)).
- **Seeds:** fixed in the evaluation scripts and the sample notebooks.

## 10. Threats to validity

| Threat | Effect | Mitigation / status |
|---|---|---|
| Automatic labels (tagger, pose detector) | Noisy supervision, especially in multi-character images | Per-crop tagging, size filter; still a limitation |
| CLIP as the colour judge | Misreads stylised hair and small eyes | Same judge for both systems; a face-crop judge would be better |
| Small evaluation sets (40 specs; 128 images per system) | One image moves a cell by 50% | Reported as trends; larger runs needed |
| Single-character colour test | Doesn't measure multi-character binding, where the method should help most | Qualitative evidence only; quantitative multi-character test still open |
| One training run | No variance estimate across seeds or runs | Stated as a limitation |
| Backbone choice (Animagine XL vs the paper's Stable Diffusion) | Results are tied to a stronger anime base | Deliberate: tests the conditioning idea on a realistic base |

## 11. Ethics and licensing

- Only images rated "general" are used.
- The app has a safety filter, plus Animagine's `safe` tag and an `nsfw` negative prompt.
- Artworks belong to their artists. The source images are a public dataset on Hugging Face, and the published data credits it.
- Base models are used under their own licenses.

## 12. Summary of findings

Each component learns its objective (RQ1: yes), and per-character conditioning cuts attribute leakage (RQ2: yes).
But the end-to-end model **doesn't yet bind colours better than its prompt-only base** (RQ3: not supported). The
main cause is a training-induced collapse of hair colours toward white. The improvement plan is in
[ANALYSIS.md § 7](ANALYSIS.md#7-where-there-is-room-to-improve).
