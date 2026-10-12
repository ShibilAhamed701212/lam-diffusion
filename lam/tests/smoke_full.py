"""GPU smoke test for the paper-completion additions (run on the A100 VM, no dataset needed).

1. Full U-Net training (§IV-F stage 3): Eq. 3 training steps at 1024px, peak memory, step time.
2. save_unet -> reload round trip.
3. Algorithm 1: G_θ forward, descriptor decoding, algorithm1() sampling with the Eq. 5 update.
4. Stage 2a pieces: sub-action parsing and a temporal-context forward/backward.
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from lam.descriptor import collate
from lam.losses import AUX_MAX_T, AuxLosses, LossWeights, diffusion_loss, predict_x0
from lam.model import LAMDiffusion
from lam.tests.smoke import fake_desc


def main() -> None:
    import bitsandbytes as bnb

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    gb = lambda: torch.cuda.max_memory_allocated() / 2**30  # noqa: E731
    print("GPU:", torch.cuda.get_device_name(), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

    # ---- 1. full U-Net training step
    t0 = time.time()
    model = LAMDiffusion(unet_mode="full")
    c = model.count()
    print(f"built in {time.time() - t0:.0f}s; total {c['total'] / 1e9:.2f}B trainable {c['trainable'] / 1e9:.2f}B; mem {gb():.1f} GB")
    groups = model.param_groups(lr_new=1e-4, lr_unet_lora=5e-5, lr_te_lora=2e-5, lr_gate=1e-3, lr_unet_full=1e-5)
    print("param groups:", {g["name"]: sum(p.numel() for p in g["params"]) for g in groups})
    assert any(g["name"] == "unet_full" for g in groups), "full mode must train the U-Net weights"
    opt = bnb.optim.PagedAdamW8bit(groups, weight_decay=0.01)
    aux = AuxLosses()
    w = LossWeights()
    desc = collate([fake_desc(2)], train=False)
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float32)
    gt = torch.rand(1, 3, 512, 512, device="cuda")
    unet_w = next(p for n, p in model.named_parameters() if model.is_base_unet_param(n))
    before = unet_w.detach().clone()
    times = []
    for i in range(4):
        torch.cuda.synchronize()
        s = time.time()
        lat = torch.randn(1, 4, 128, 128, device="cuda")
        t = torch.tensor([200], device="cuda")  # < AUX_MAX_T so all four Eq. 3 terms are active
        assert int(t) < AUX_MAX_T
        noise = torch.randn_like(lat)
        ab = torch.tensor([0.7], device="cuda")
        noisy = ab.sqrt() * lat + (1 - ab).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lam = model.encode(desc)
            eps = model.denoise(noisy, t, lam, time_ids)
            l_diff = diffusion_loss(eps, noise, ab / (1 - ab), gamma=0)
            r = aux(predict_x0(noisy, eps.float(), ab), gt, lam.action_embed)
            loss = w.diff * l_diff + w.action_img * r["action_img"] + w.perc * r["perc"] + w.style * r["style"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - s)
        print(f"step {i}: loss {loss.item():.4f} ({ {k: round(float(v), 4) for k, v in r.items()} }) "
              f"{times[-1]:.2f}s peak {gb():.1f} GB", flush=True)
    assert (unet_w.detach() - before).abs().max() > 0, "U-Net weights did not change"
    print(f"[full U-Net] OK: {sum(times[1:]) / 3:.2f}s/step at 1024px batch 1, peak {gb():.1f} GB")

    # ---- 2. save / reload
    from diffusers import UNet2DConditionModel

    model.save_unet("/content/_smoke_unet")
    re = UNet2DConditionModel.from_pretrained("/content/_smoke_unet", torch_dtype=torch.bfloat16)
    k = "conv_in.weight"
    diff = (re.state_dict()[k].float().cpu() - model.unet.state_dict()[k].float().cpu()).abs().max().item()
    assert diff < 1e-2, diff
    print(f"[save_unet] OK (reload max diff {diff:.1e})")
    del re, opt, aux
    torch.cuda.empty_cache()

    # ---- 3. Algorithm 1
    from lam.inverse import ActionFromLatent, build_vocab
    from lam.pipeline import LAMDiffusionPipeline

    vocab = build_vocab([fake_desc(2)])
    inv = ActionFromLatent(vocab).cuda().eval()
    pipe = LAMDiffusionPipeline(model)
    img = Image.new("RGB", (832, 1216), (200, 120, 160))
    out, descs = pipe.algorithm1([img], inv, width=832, height=1216, eta=0.5, steps=4)
    assert out[0].size == (832, 1216) and descs[0]["characters"], descs
    print("[algorithm1] OK; predicted descriptor (untrained G_θ):", {k: v for k, v in descs[0]["characters"][0].items() if k != "keypoints"})

    # ---- 4. stage 2a pieces
    from lam.data.humanml3d import sub_actions
    from lam.train_stage2a import act_ids

    seq = sub_actions("a person walks forward, then turns around and sits down")
    assert seq == ["walking", "looking_back", "sitting"], seq
    model.train()
    ids = act_ids([seq, seq[::-1]], "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok, valid = model.encoder.dynamics(ids, torch.zeros(2, 18, 3, device="cuda"))
        ctx = model.encoder.temporal.context(tok, valid)
    ctx.float().pow(2).mean().backward()
    g = model.encoder.temporal.enc.layers[0].self_attn.in_proj_weight.grad
    assert g is not None and g.abs().sum() > 0, "temporal module receives no gradient"
    assert (ctx[0] - ctx[1]).abs().max() > 0, "temporal context ignores order"
    print("[stage 2a] OK: parsing, order-sensitive temporal context, gradients flow")
    print("ALL OK")


if __name__ == "__main__":
    main()
