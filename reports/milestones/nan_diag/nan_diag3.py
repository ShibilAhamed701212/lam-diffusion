import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from lam.dataset import SampleDataset, collate_samples, load_index
from lam.descriptor import collate
from lam.losses import diffusion_loss
from lam.model import BASE_MODEL, LAMDiffusion

ckpt = False
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_cudnn_sdp(False); print("math sdpa only")
train = load_index("/content/data/index.jsonl", "train")[:1]
model = LAMDiffusion(gradient_checkpointing=ckpt, unet_mode="full")
model.load_trainable("/content/ckpt/stage2_from2a.pt", strict=False)
alphas = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler").alphas_cumprod.cuda()
batch = next(iter(DataLoader(SampleDataset(train, "/content/data/samples"), batch_size=1, collate_fn=collate_samples)))
lat = batch["latents"].cuda()
model.train()
print("gradient_checkpointing =", ckpt, flush=True)
for tv in [50, 200, 350, 500, 600, 700, 800, 900, 990]:
    torch.manual_seed(0)
    t = torch.tensor([tv], device="cuda"); ab = alphas[t]
    noise = torch.randn_like(lat)
    noisy = ab.sqrt().view(-1,1,1,1)*lat + (1-ab).sqrt().view(-1,1,1,1)*noise
    desc = collate(batch["descs"], train=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lamo = model.encode(desc)
        eps = model.denoise(noisy, t, lamo, batch["time_ids"].cuda())
        l = diffusion_loss(eps, noise, ab/(1-ab), gamma=0)
    l.backward()
    n = sum(1 for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad.float()).all())
    first = next((k for k, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad.float()).all()), None)
    print(f"t={tv:4d} loss={l.item():.4f} nonfinite_grad_tensors={n} first={first}", flush=True)
    model.zero_grad(set_to_none=True)
