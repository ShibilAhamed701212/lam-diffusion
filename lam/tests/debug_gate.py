import collections

import torch

from lam.conditioning import DualStreamAttnProcessor
from lam.descriptor import collate
from lam.model import LAMDiffusion
from lam.tests.smoke import fake_desc

model = LAMDiffusion(gradient_checkpointing=False)
kinds = collections.Counter(type(p).__name__ for p in model.unet.attn_processors.values())
print("processor types after LoRA:", kinds)
proc = next(iter(model.conditioning.processors.values()))
installed = [p for p in model.unet.attn_processors.values() if p is proc]
print("conditioning processor is installed in unet:", bool(installed))

calls = collections.Counter()
orig = DualStreamAttnProcessor.__call__


def spy(self, attn, hidden_states, encoder_hidden_states=None, **kw):
    calls["called"] += 1
    if self.state.tokens is not None:
        calls["tokens"] += 1
    return orig(self, attn, hidden_states, encoder_hidden_states, **kw)


DualStreamAttnProcessor.__call__ = spy
d = collate([fake_desc(2)])
lat = torch.randn(1, 4, 128, 128, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    lam = model.encode(d)
    print("char_mask", lam.char_mask.tolist(), "char token norm", lam.char_tokens.float().norm().item())
    eps = model.denoise(lat, torch.tensor([200], device="cuda"), lam,
                        torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float32))
eps.float().pow(2).mean().backward()
print("processor calls:", dict(calls))
print("gate grads:", [p.gate.grad for p in list(model.conditioning.processors.values())[:3]])
