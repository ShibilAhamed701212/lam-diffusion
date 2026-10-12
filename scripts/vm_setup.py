# @title LAM-Diffusion VM setup
# Runs inside the Colab kernel (colab exec -f). Idempotent; re-run on every fresh VM.
import subprocess
import sys

STEPS = [
    # Colab's preinstalled torchao is too old for current peft, and nothing here uses it.
    "pip uninstall -y -q torchao onnxruntime",
    "pip install -q 'diffusers>=0.32' 'transformers>=4.45' 'accelerate>=0.34' 'peft>=0.13' bitsandbytes "
    "timm lpips clean-fid rtmlib onnxruntime-gpu fastapi uvicorn",
    # rtmlib depends on the CPU-only onnxruntime, which shadows the GPU build; swap it back.
    "pip uninstall -y -q onnxruntime",
    "pip install -q --force-reinstall --no-deps onnxruntime-gpu",
]
for cmd in STEPS:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(" ".join(cmd.split()[1:3]), "ok" if r.returncode == 0 else f"FAILED: {r.stderr[-400:]}")
r = subprocess.run([sys.executable, "-c", "import torch, peft, diffusers, onnxruntime as o; o.preload_dlls(); print('peft', peft.__version__, 'diffusers', diffusers.__version__, o.get_available_providers()[:2])"],
                   capture_output=True, text=True)
print(r.stdout or r.stderr[-600:])
