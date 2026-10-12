# Implementation: paper → code

This repo implements the **learned** system of the paper's §IV. The paper's own experiments (§V, Table 3)
used a rule-assisted planner and stock Stable Diffusion on CPU, inference-only, and list "replace the
rule-based planner with a fully learnable transformer-based LAM" as future work (§VI-B). This is that version,
trained end to end.

## Paper → code map

| Paper | Code | Status |
|---|---|---|
| Eq. 1 `A = F_LAM(N)`, planner | `server/planner.py` (LLM + rules), `lam/planner_ft.py` (Qwen2.5-3B + LoRA) | ✅ |
| Structured descriptors `A = {a_1…a_n}` | `lam/descriptor.py` | ✅ |
| §IV-B pre-trained transformer backbone | `TextBackbone` (SDXL CLIP-L + bigG, LoRA) in `lam/encoder.py` | ✅ |
| §IV-B pose encoding | `PoseEncoder` | ✅ |
| §IV-B action dynamics encoding | `DynamicsEncoder` + `kinematics()` | ✅ |
| §IV-B style attribute fusion | `StyleAttributeFusion` | ✅ |
| Output `B × N × D_action` | `LAMOutput` (D = 2048) | ✅ |
| §IV-C cross-attention conditioning | `DualStreamAttnProcessor` in every U-Net cross-attention layer | ✅ |
| §IV-C temporal attention module | `TemporalAttention` | ✅ |
| §IV-D anime VAE encoder/decoder (frozen) | SDXL VAE; decoder fine-tuned in stage 1 (`lam/train_vae.py`) | ✅ |
| §IV-D denoising U-Net + CFG | Animagine XL 4.0 U-Net, **fully trained**; CFG with a null action embedding | ✅ |
| §IV-E Eq. 3, four-term loss | `lam/losses.py` | ✅ |
| §IV-F stage 1, anime VAE | `lam/train_vae.py` | ✅ |
| §IV-F stage 2, LAM pre-train on action-text corpus | `lam/train_stage2a.py` + `lam/data/humanml3d.py` | ✅ |
| §IV-F stage 2, LAM fine-tune on triplets | `lam/train_stage2.py` | ✅ |
| §IV-F stage 3, U-Net + joint LAM, frozen VAE | `lam/train_joint.py --unet-mode full --min-snr 0` | ✅ |
| §IV-F stage 4, progressive curriculum | `CurriculumSampler` (`lam/dataset.py`) | ✅ |
| §IV-F stage 5, CFG dropout | `collate(train=True)` | ✅ |
| §IV-F inference pipeline | `LAMDiffusionPipeline.__call__` | ✅ |
| Algorithm 1 / Eq. 4–6 | `lam/inverse.py`, `lam/train_inverse.py`, `LAMDiffusionPipeline.algorithm1` | ✅ |
| §V-C metrics | `lam/eval/metrics.py`, `lam/eval/generate.py`, `lam/tests/eval_leakage.py` | ✅ code; see [RESULTS.md](RESULTS.md) |

## Interpretation choices

The paper leaves some details open. These are the choices made, and why.

- **"Pre-trained transformer in the GPT/BERT paradigm" (§IV-B)** = SDXL's own CLIP text transformers. Their output space is exactly what the U-Net's cross-attention already reads.
- **"Large-scale action-text corpus" (§IV-B)** = HumanML3D captions (about three descriptions per motion; 69,896 training captions including the left/right-mirrored copies). Only the text is used. Ordered captions ("walks forward, then sits down") also supervise the temporal module.
- **Anime VAE (§IV-F stage 1).** Re-training the VAE *encoder* would move the latent space and break the pre-trained U-Net, so only the decoder is fine-tuned. The latents, and every precomputed sample, stay valid.
- **Algorithm 1 / Eq. 4–6.** `h = F_φ(x)` is the VAE encoder (its inverse is the decoder of Eq. 6). `G_θ` is a ConvNet + transformer head that predicts the descriptor from the latent. Eq. 5 `z ← z − η∇L_diff` is applied with a stop-gradient on the predicted noise, which works out to `ε̂ ← ε̂ + η (ẑ0 − h)/σ` before each scheduler step. η = 0 generates a new image from the predicted actions; η → 1 reconstructs the input.
- **"Style-discriminative network" for `L_style` (§IV-E)** is not named in the paper; VGG-16 Gram matrices are used.
- **λ1…λ4** are not given in the paper. They were set with pilots: 1 / 0.005 / 0.02 / 200.

## Additions and deviations (reported, not hidden)

| Kind | What | Why |
|---|---|---|
| Added | **Bounding-box region masks** on character tokens | Binds attributes to the right character (the paper's Fig. 3 layout prior). Switchable (`region_masking`) for ablation. |
| Added | **Per-layer logit bias ("gate")** on action tokens | Learned strength of the action stream per layer. Starts at 0 = read like text. |
| Added | **TAESDXL** decoder inside the auxiliary losses | A full VAE decode with gradients at 1024 px doesn't fit next to a fully trained U-Net. |
| Added | Balanced data selection, LLM planner fine-tune, data-mined pose library | The paper doesn't describe a dataset recipe; anime data is heavily skewed. |
| Changed | **One shared softmax** over text + action tokens (v3) | The first version *added* a separate action attention on top of the text attention, which made images muddy (see below). |
| Off | Min-SNR weighting of `L_diff` | The paper says "standard denoising loss". |
| Off | Attribute-swap consistency loss | Tried; it stayed at ≈1.6e-5 from step 0 and didn't reduce leakage. |
| Engineering | 8-bit / paged AdamW, gradient checkpointing, bf16 autocast, fp16 inference mode | Memory and throughput only; no effect on the method. |

## Design history: the v3 conditioning

The first full overfit check (16 training triplets) failed: no characters appeared, and the images turned into
textures. Step-0 diagnosis (`lam/tests/diag_step0.py`, sheets in `reports/milestones/m3_diag/`) separated three causes.

1. **A stale checkpoint.** An encoder checkpoint from an older design loaded large weights into layers the new design zero-initialises. Character tokens had norm 55–166, against 34 for text tokens. It produced the textures before any training.
2. **Additive attention.** `out = Attn(text) + gate · Attn(action)` doubled the signal inside character boxes. As training grew the tokens (norm 10 → 45), images went muddy and sepia.
3. **A pooled-embedding shift.** A learned character term added to SDXL's pooled embedding was amplified by CFG, because the unconditional branch has no characters.

**v3** fixed all three:
- one shared softmax with box masks;
- each attribute's real token sequence instead of an average, so tokens sit at text-token scale (34.1 vs 34.2);
- no pooled shift.

The untrained v3 model already draws characters with the specified hair and outfit, and a 600-step overfit
reproduced most specs. That is the design used for all later training.

## Repository layout

```
lam/                     the model (paper §IV)
  descriptor.py          action descriptor schema, CFG/field dropout
  encoder.py             LAM Encoder (§IV-B)
  conditioning.py        dual-stream pathway (§IV-C)
  model.py               backbone assembly, LoRA/full U-Net, checkpoints (§IV-D)
  pipeline.py            inference, Algorithm 1, prompt-only baseline
  losses.py              Eq. 3 (§IV-E)
  inverse.py             Algorithm 1 head G_θ
  dataset.py             samples + progressive curriculum (stage 4)
  train_vae.py           stage 1
  train_stage2a.py       stage 2a (action-text)
  train_stage2.py        stage 2 (triplets)
  train_joint.py         stages 3–5
  train_inverse.py       G_θ
  planner_data.py, planner_ft.py, spec_bridge.py, pose_library.py   planner F_LAM
  data/                  dataset build: select → fetch → annotate → precompute; humanml3d; tag_vocab
  eval/                  §V-C generation + metrics
  tests/                 GPU smoke tests, diagnosis sheets, leakage probe
  jobs/                  shell orchestration of the stages on a cloud GPU
server/                  FastAPI inference API (planner + engines)
web/                     Next.js app ("LAM Studio")
scripts/                 Colab helpers (VM setup, jobs, keep-alive, uploads, Drive mount, samples)
reports/                 figures/ + data/ (raw results) + scripts/ (regenerate figures) + milestones/ (diagnosis sheets, sample grids)
output/                  app demo video + sample generations
docs/                    this documentation
```
