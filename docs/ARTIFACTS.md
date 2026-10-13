# Where everything lives

Every artifact of the project exists in exactly one published place: GitHub for code, docs and small files,
Kaggle for weights and data. Nothing is duplicated across them.

## GitHub (this repo)

| Path | Contents |
|---|---|
| `lam/`, `server/`, `web/`, `scripts/` | All code: model, training stages, dataset build, evaluation, app, cloud-GPU helpers |
| `docs/` | Architecture, implementation, dataset, training, inference, results, this file |
| `reports/figures/`, `reports/data/`, `reports/scripts/` | Report figures, the raw logged numbers behind them, and the scripts that regenerate them (stage results, leakage runs, colour factorial, dataset stats) |
| `reports/data/logs/`, `reports/data/full_run_phaseB.csv` | The full-run training log and its parsed table, the launch scripts, and the final evaluation log (pulled from the training machine) |
| `reports/analysis/` | Analysis figures (heat maps, confusion matrices, thresholds) and CSV tables from `reports/scripts/make_analysis.py` |
| `reports/milestones/` | Diagnosis sheets and sample grids from every milestone (overfit failure, v3 fix, pilots, full-U-Net gate, final model, Kaggle precision checks) |
| `output/` | App demo video and sample generations |

## Kaggle (all public)

| Dataset / notebook | Contents |
|---|---|
| [`lam-diffusion-final`](https://www.kaggle.com/datasets/shibilahamed0095/lam-diffusion-final) | **Trained model:** full U-Net (`unet.safetensors` + config), all other trained weights (`weights.pt`: LAM Encoder, text-encoder LoRA, dual-stream processors), pose library, code snapshot |
| [`lam-diffusion-extras`](https://www.kaggle.com/datasets/shibilahamed0095/lam-diffusion-extras) | Stage 1 anime VAE decoder (`vae_anime/`), stage 2a and stage 2 encoder checkpoints, Algorithm 1 head `G_θ` (`inverse.pt`), stage logs |
| [`lam-diffusion-descriptors`](https://www.kaggle.com/datasets/shibilahamed0095/lam-diffusion-descriptors) | Dataset annotations for 28,145 images, train/test split, selection, pose library, stats. **No images.** |
| [`lam-diffusion-precomputed`](https://www.kaggle.com/datasets/shibilahamed0095/lam-diffusion-precomputed) | Ready-to-train set: SDXL latents, CLIP embeddings, half-size ground truth, index |
| [`lam-diffusion-training-logs`](https://www.kaggle.com/datasets/shibilahamed0095/lam-diffusion-training-logs) | **Full-run training record:** raw log and parsed per-50-step curves, 15 sample grids, evaluation log, launch scripts, trend statistics, NaN-diagnosis scripts |
| [`lam-diffusion-inference`](https://www.kaggle.com/code/shibilahamed0095/lam-diffusion-inference) (notebook) | Clones this repo, loads the trained model on a free T4 in fp16, renders samples next to stock Animagine |

## Not published (on purpose)

| Item | Why |
|---|---|
| Pilot checkpoints from the swirl-fix and swap-loss experiments | Obsolete side experiments. Their results are in `reports/milestones/fix*` and `reports/data/leakage_runs.json`. |
| Kaggle `lam-server` notebook | Contains a server token |
| The research paper PDF | Not ours to redistribute |
