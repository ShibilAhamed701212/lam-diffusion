import os, glob
import numpy as np
CH = 1 << 20
def scan(p):
    size = os.path.getsize(p); n = z = first_zero = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(CH)
            if not b: break
            if not np.frombuffer(b, dtype=np.uint8).any():
                z += 1
                if first_zero is None or first_zero == 0: first_zero = n if first_zero == 0 else first_zero
            n += 1
    return size, n, z
paths = sorted(glob.glob(os.path.expanduser("~/lam_studio/content/ckpt_full/unet_step*/diffusion_pytorch_model.safetensors"))
               + glob.glob(os.path.expanduser("~/lam_studio/drive/ckpt/unet_latest/diffusion_pytorch_model.safetensors")))
for p in paths:
    s, n, z = scan(p)
    print(f"{z:5d}/{n} zero MiB-blocks ({z/max(n,1):.1%})  {s/1e9:.2f} GB  {p.replace(os.path.expanduser('~'), '~')}", flush=True)
