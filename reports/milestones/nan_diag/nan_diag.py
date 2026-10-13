import random, torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from lam.dataset import SampleDataset, collate_samples, load_index
from lam.descriptor import collate
from lam.losses import AuxLosses, LossWeights, diffusion_loss, predict_x0
from lam.model import BASE_MODEL, LAMDiffusion

def fin(name, x):
    ok = bool(torch.isfinite(x.float()).all())
    print(f"{name:28s} finite={ok} absmax={x.float().abs().max().item():.4g}", flush=True)
    return ok

train = load_index("/content/data/index.jsonl", "train")[:8]
model = LAMDiffusion(gradient_checkpointing=True, unet_mode="full")
model.load_trainable("/content/ckpt/stage2_from2a.pt", strict=False)
bad = [n for n, p in model.named_parameters() if not torch.isfinite(p.float()).all()]
print("non-finite params after load:", len(bad), bad[:5], flush=True)
aux = AuxLosses()
sched = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
alphas = sched.alphas_cumprod.cuda()
ds = SampleDataset(train, "/content/data/samples")
dl = DataLoader(ds, batch_size=1, collate_fn=collate_samples)
rng = random.Random(0)
model.train()
for i, batch in enumerate(dl):
    lat = batch["latents"].cuda()
    fin("latents", lat); fin("gt image", batch["gt"].cuda())
    t = torch.randint(0, 1000, (1,), device="cuda"); ab = alphas[t]
    noise = torch.randn_like(lat)
    noisy = ab.sqrt().view(-1,1,1,1)*lat + (1-ab).sqrt().view(-1,1,1,1)*noise
    desc = collate(batch["descs"], train=True, rng=rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lamo = model.encode(desc)
        fin("global_hidden", lamo.global_hidden); fin("char_tokens", lamo.char_tokens); fin("pooled", lamo.pooled)
        eps = model.denoise(noisy, t, lamo, batch["time_ids"].cuda())
        fin("eps", eps)
        l = diffusion_loss(eps, noise, ab/(1-ab), gamma=0)
        print("t", t.item(), "L_diff", l.item(), flush=True)
        if t.item() < 400 and not desc.drop_all.any():
            x0 = predict_x0(noisy, eps.float(), ab); fin("x0", x0)
            r = aux(x0, batch["gt"].cuda(), lamo.action_embed)
            print({k: float(v) for k, v in r.items()}, flush=True)
    l.backward()
    g = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad.float()).all()]
    print("non-finite grads:", len(g), g[:5], flush=True)
    model.zero_grad(set_to_none=True)
    if i >= 7: break
print("DIAG DONE")
