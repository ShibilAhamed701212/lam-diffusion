# @title Start LAM Studio server
# Runs *inside* the Colab kernel (sent by colab_up.sh). Unpacks the server, installs deps,
# starts uvicorn in a background thread, opens a Cloudflare quick tunnel and prints its URL.
# Model loading continues in the background; poll /health until status == "ready".
import os
import re
import subprocess
import sys
import tarfile
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = "/content/lam_server"
os.makedirs(ROOT, exist_ok=True)
with tarfile.open("/content/lam_server.tgz") as tf:
    tf.extractall(ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{ROOT}/requirements.txt"], check=True)

if not os.path.exists("/content/cloudflared"):
    subprocess.run(["wget", "-q", "-O", "/content/cloudflared",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                   check=True)
    os.chmod("/content/cloudflared", 0o755)

sys.path.insert(0, ROOT)
import uvicorn  # noqa: E402

import app as lam_app  # noqa: E402

lam_app.start_loading()
server = uvicorn.Server(uvicorn.Config(lam_app.app, host="127.0.0.1", port=8000, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()

subprocess.run(["pkill", "-f", "/content/cloudflared"])  # stale tunnel from a previous kernel
log = open("/content/cloudflared.log", "w")
tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"],
                          stdout=log, stderr=subprocess.STDOUT)
url = None
for _ in range(60):
    time.sleep(1)
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/cloudflared.log").read())
    if m:
        url = m.group(0)
        break
print(f"LAM_API_URL={url}" if url else "TUNNEL_FAILED")
