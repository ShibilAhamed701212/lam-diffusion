"""GPU smoke test for the LAM-Diffusion model (run on the Colab VM).

Checks: output shapes; gate=0 => identical to the base model; region masks match boxes at
every resolution; Eq. 3 losses finite with gradients reaching the right parameters; step
time and peak memory at 1024px for the training budget.
"""

from __future__ import annotations

import time

import torch

from lam.descriptor import N_KEYPOINTS, collate
from lam.encoder import TOKENS_PER_CHAR
from lam.losses import AuxLosses, LossWeights, diffusion_loss, predict_x0
from lam.model import LAMDiffusion


def fake_desc(n_chars: int = 2) -> dict:
    chars = []
    for i in range(n_chars):
        x0 = 0.05 + 0.5 * i
        kps = [[x0 + 0.2, 0.1 + 0.8 * j / N_KEYPOINTS, 0.9] for j in range(N_KEYPOINTS)]
        chars.append({"gender": ["male", "female"][i % 2], "age": "elderly" if i == 0 else "teen",
                      "hair_color": "silver", "hair_style": "long hair", "eye_color": "blue",
                      "outfit": "school uniform", "expression": "smile", "extra": "",
                      "actions": ["standing", "waving"], "framing": "full_body", "view": "front",
                      "bbox": [x0, 0.05, x0 + 0.4, 0.95], "keypoints": kps})
    return {"global": {"scene": "cherry blossoms, park", "style": "", "props": []}, "characters": chars}


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    t0 = time.time()
    model = LAMDiffusion(device=dev)
    print(f"built in {time.time() - t0:.0f}s; params {model.count()}")

    desc = collate([fake_desc(2)])
    lat = torch.randn(1, 4, 128, 128, device=dev)
    t = torch.tensor([500], device=dev)
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=dev, dtype=torch.float32)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lam = model.encode(desc)
        assert lam.global_hidden.shape == (1, 77, 2048), lam.global_hidden.shape
        assert lam.char_tokens.shape == (1, 3, TOKENS_PER_CHAR, 2048), lam.char_tokens.shape
        assert lam.pooled.shape == (1, 1280) and lam.action_embed.shape == (1, 768)
        eps_lam = model.denoise(lat, t, lam, time_ids)
        model.conditioning.state.scale = 0.0  # action stream off
        eps_off = model.denoise(lat, t, lam, time_ids)
        model.conditioning.state.scale = 1.0
        model.conditioning.state.tokens = None  # base model: no action stream at all
        eps_base = model.unet(lat, t, encoder_hidden_states=lam.global_hidden,
                              added_cond_kwargs={"text_embeds": lam.pooled, "time_ids": time_ids}).sample
    off = (eps_off.float() - eps_base.float()).abs().max().item()
    on = (eps_lam.float() - eps_base.float()).abs().mean().item()
    print(f"[action stream] off vs base max diff {off:.2e}; on vs base mean diff {on:.3e}")
    assert off < 1e-3, "with the action stream off the model must be exactly the base model"
    assert on > 1e-3, "characters must reach the U-Net from step 0"

    # Region masks: each character's tokens visible only inside its (margin-expanded) box.
    model.conditioning.bind(lam, latent_hw=(128, 128))
    st = model.conditioning.state
    for hw in (64 * 64, 32 * 32):
        bias = st.bias(hw, torch.float32)[0, 0]  # [hw, Na]
        allowed = (bias == 0).float()
        frac_char1 = allowed[:, 0].mean().item()  # first (gender) token of character 1
        frac_absent = allowed[:, 2 * TOKENS_PER_CHAR].mean().item()  # character slot 3 is empty
        print(f"[region mask {int(hw ** 0.5)}^2] char1 visible on {frac_char1:.2f} of positions (box area ~0.43), absent char {frac_absent:.2f}")
        assert 0.35 < frac_char1 < 0.55 and frac_absent == 0.0

    # Losses + gradients, and a timing benchmark of full training steps.
    aux = AuxLosses(dev)
    w = LossWeights()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(6):
        torch.cuda.synchronize()
        s0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lam = model.encode(collate([fake_desc(2)], train=True))
            noise = torch.randn_like(lat)
            tt = torch.tensor([200], device=dev)
            ab = torch.tensor([0.7], device=dev)
            noisy = ab.sqrt() * lat + (1 - ab).sqrt() * noise
            eps = model.denoise(noisy, tt, lam, time_ids)
            l_diff = diffusion_loss(eps, noise, ab / (1 - ab))
            a = aux(predict_x0(noisy, eps.float(), ab), torch.rand(1, 3, 512, 512, device=dev), lam.action_embed)
            loss = w.diff * l_diff + w.action_img * a["action_img"] + w.perc * a["perc"] + w.style * a["style"]
        loss.backward()
        if step == 0:
            gate_grad = sum(p.gate.grad.abs().sum().item() for p in model.conditioning.processors.values() if p.gate.grad is not None)
            lora_grad = sum(p.grad.abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None)
            print({k: round(float(v), 4) for k, v in {"l_diff": l_diff, **a}.items()})
            print(f"[grads] gates {gate_grad:.3e}, lora_B {lora_grad:.3e}")
            assert torch.isfinite(loss) and gate_grad > 0 and lora_grad > 0
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - s0)
    print(f"[benchmark 1024px, batch 1, all 4 losses] step times {[round(x, 2) for x in times]}; "
          f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("gates after 6 steps:", model.conditioning.gates()[:6].tolist())
    print("SMOKE OK")


if __name__ == "__main__":
    main()
