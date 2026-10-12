# Workflow

The end-to-end workflow of the project: data, training, validation, evaluation, serving and publishing. Each step
lists its inputs, outputs and the gate that must pass before moving on.

## 1. The whole pipeline

```mermaid
flowchart LR
    subgraph DATA[1 · Data]
        A1[Danbooru metadata] --> A2[select<br/>balanced IDs]
        A2 --> A3[fetch<br/>range requests]
        A3 --> A4[annotate<br/>poses + per-character tags]
        A4 --> A5[precompute<br/>latents, CLIP, split]
        H1[HumanML3D captions] --> H2[ordered sub-actions]
    end
    subgraph TRAIN[2 · Training stages]
        S1[Stage 1<br/>anime VAE decoder]
        S2a[Stage 2a<br/>action-text pre-training]
        S2[Stage 2<br/>descriptor ↔ image]
        G[Overfit gate<br/>16 samples]
        S3[Stages 3–5<br/>full U-Net + encoder<br/>curriculum, CFG dropout]
        INV[Algorithm 1 head G_θ]
    end
    subgraph EVAL[3 · Evaluation]
        E1[stage metrics]
        E2[leakage swap test]
        E3[colour factorial<br/>vs stock Animagine]
        E4[analysis<br/>heat maps, thresholds]
    end
    subgraph SHIP[4 · Serve & publish]
        P1[LAM Studio app]
        P2[Kaggle weights + data + notebook]
        P3[GitHub code + docs]
    end
    A5 --> S1 & S2 & INV
    H2 --> S2a --> S2 --> G --> S3
    S1 --> S3
    S3 --> E2 & E3
    S1 & S2a & S2 & INV --> E1
    E1 & E2 & E3 --> E4
    S3 --> P1 & P2
    E4 --> P3
```

## 2. Step by step

| # | Step | Command | Input → output | Gate |
|---|---|---|---|---|
| 1 | Select | `python -m lam.data.select` | metadata → `selection.jsonl` (29,785 IDs) | Composition shares are as intended |
| 2 | Fetch | `python -m lam.data.fetch` | IDs → images | Most IDs fetched (resumable) |
| 3 | Annotate | `python -m lam.data.annotate` | images → `annotations.jsonl` | Visual QA grid looks right (`lam.data.report`) |
| 4 | Precompute | `python -m lam.data.precompute`, `python -m lam.pose_library` | → `samples/*.npz`, `index.jsonl` (27,146 / 999), `pose_library.json` | 28,145 kept after the size filter |
| 5 | Stage 1 | `python -m lam.train_vae` | images → `vae_anime/` | Beats the stock decoder on PSNR and LPIPS |
| 6 | Stage 2a | `python -m lam.train_stage2a` | HumanML3D → `stage2a.pt` | Order accuracy ≫ 50% |
| 7 | Stage 2 | `python -m lam.train_stage2 --init stage2a.pt` | triplets → `stage2_from2a.pt` | R@1 ≫ the global-only baseline |
| 8 | Overfit gate | `bash lam/jobs/a100_full.sh overfit` | 16 triplets → sample grids | Training specs reproduced, no artefacts |
| 9 | Full run | `LAM_GO=1 bash lam/jobs/phase_b.sh` | → `unet/`, `weights.pt` | Sample grids every 750 steps stay clean |
| 10 | G_θ | `python -m lam.train_inverse` | latents → `inverse.pt` | Above majority-class accuracy |
| 11 | Evaluate | `lam/tests/eval_leakage.py`, Kaggle `lam-eval`, `lam/eval/` | weights → JSON results | — |
| 12 | Analyse | `python reports/scripts/make_figs_1.py`, `make_analysis.py` | JSON → figures and tables | — |
| 13 | Serve | `scripts/colab_up.sh` or `uvicorn app:app` + `npm run dev` | weights → app | `/health` reports ready |
| 14 | Publish | `git push`; `kaggle datasets create/version`, `kaggle kernels push` | → GitHub, Kaggle | Links resolve; no secrets |

Shortcuts: steps 1–4 can be replaced by downloading
[`lam-diffusion-precomputed`](https://www.kaggle.com/datasets/shibilahamed0095/lam-diffusion-precomputed).
Steps 5–7 run in one go with `bash lam/jobs/paper_stages.sh`.

## 3. Development workflow (how a change gets accepted)

```mermaid
flowchart TD
    C[Code change] --> T1{Smoke test<br/>shapes, base-equivalence,<br/>masks, gradients}
    T1 -- fail --> C
    T1 -- pass --> T2{Step-0 render<br/>untrained model draws<br/>the spec?}
    T2 -- no --> D[Diagnose variants<br/>token norms, action on/off,<br/>masks on/off, checkpoints]
    D --> C
    T2 -- yes --> T3{Overfit 16 samples<br/>reproduced?}
    T3 -- no --> D
    T3 -- yes --> T4{Pilot ~200–600 steps<br/>clean sample grids?}
    T4 -- no, artefacts --> W[Tune loss weights / LR] --> T4
    T4 -- yes --> F[Full run on a large GPU]
    F --> EV[Evaluate vs baseline] --> R[Report, including negatives]
```

This ladder caught the failed first conditioning design at the overfit gate and diagnosed it at step 0, before
any long run. It also caught the swirl artefacts at the pilot stage.

## 4. Compute workflow

```mermaid
flowchart LR
    M[Mac: code, docs,<br/>orchestration] -- colab CLI --> L4[Colab L4<br/>data build, stages 1/2a/2,<br/>G_θ, app server]
    M -- colab CLI --> A100[Colab A100<br/>full-U-Net smoke test]
    M --> H200[H200<br/>11k-step full run]
    M -- kaggle CLI --> K[Kaggle 2×T4<br/>inference, evaluation]
    L4 & H200 -- checkpoints --> M
```

- **Background jobs:** `scripts/colab_job.sh start NAME cmd…` syncs the code and runs a detached job (`log`, `status`, `pull`).
- **Keep-alive:** Colab reclaims a VM about 20 min after the last cell execution, even mid-job, so `colab_keepalive.sh` pings it every 3 min. It runs on the Mac, so the Mac must stay awake.
- **Resumability:** every training script checkpoints periodically; `phase_a.sh` skips finished stages; `phase_b.sh` resumes from the newest checkpoint.
- **Large uploads:** `scripts/colab_upload_big.sh` splits files, uploads the pieces and verifies the checksum.

## 5. Serving workflow (one request in the app)

```mermaid
sequenceDiagram
    participant U as Browser (web/)
    participant P as Next.js /api/lam proxy
    participant S as FastAPI (server/) on GPU
    participant M as LAM-Diffusion model
    U->>P: POST /plan {prompt}
    P->>S: + X-LAM-Token
    S->>S: safety filter → LLM / rule planner → normalize_spec
    S-->>U: spec + pose skeleton image (editable)
    U->>P: POST /generate {edited spec, seed, steps, cfg}
    P->>S: + X-LAM-Token
    S->>M: spec → descriptor (pose library) → LAM Encoder → dual-stream U-Net → VAE
    M-->>S: image (+ optional stock baseline, same seed)
    S-->>U: images (base64)
```

The proxy keeps the GPU server's address and token out of the browser. `LAM_ENGINE=lam` serves the trained model;
`controlnet` serves the earlier prototype.

## 6. Publishing workflow

| Artifact | Destination | Rule |
|---|---|---|
| Code, docs, figures, analysis, milestone sheets, demo video | GitHub | Nothing secret (`.env`, tokens), no weights, no data archives (`.gitignore`) |
| Trained U-Net + weights, anime VAE, stage checkpoints, G_θ | Kaggle datasets (`final`, `extras`) | Each large file exists once |
| Descriptors, precomputed training set | Kaggle datasets (`descriptors`, `precomputed`) | Source credited |
| Inference demo | Kaggle notebook (`lam-diffusion-inference`) | Clones the GitHub repo, so it always runs the current code |

Before each push: compile all Python, `bash -n` every script, type-check the web app, check every doc link,
and scan for secrets. See [ARTIFACTS.md](ARTIFACTS.md) for the full map.
