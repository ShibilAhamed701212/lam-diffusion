"""Mount Google Drive on a Colab CLI session without the Enter-key race.

`colab drivemount` continues on the first Enter it sees, so a pasted newline or a double press
authorizes the VM before you've granted access (Google then answers 400). This runs it in a
pseudo-terminal and only sends Enter after you type `done`.

    python3 scripts/drive_mount.py SESSION
"""

import fcntl
import os
import re
import select
import subprocess
import sys
import termios

session = sys.argv[1] if len(sys.argv) > 1 else "lamphasea"
master, slave = os.openpty()


def ctty():
    os.setsid()
    fcntl.ioctl(slave, termios.TIOCSCTTY, 0)


p = subprocess.Popen(["colab", "drivemount", "-s", session], stdin=slave, stdout=slave, stderr=slave,
                     preexec_fn=ctty, close_fds=True)
os.close(slave)
buf, prompted, sent = b"", False, False
while True:
    r, _, _ = select.select([master], [], [], 0.5)
    if r:
        try:
            data = os.read(master, 65536)
        except OSError:
            break
        if not data:
            break
        buf += data
        text = data.decode(errors="replace")
        sys.stdout.write(text.replace("Press Enter after you have granted access... ", ""))
        sys.stdout.flush()
    if not prompted and b"Press Enter after you have granted access" in buf:
        prompted = True
        m = re.search(rb"https://accounts\.google\.com\S+", buf)
        if m:
            subprocess.run(["open", m.group(0).decode()])
        print("\n>>> The approval page is opening in your browser.")
        print(">>> Choose your Google account, tick ALL boxes, click Continue/Allow until Google says done.")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)  # throw away any stray Enter already typed
        while True:
            ans = input(">>> Then type  done  and press Enter: ").strip().lower()
            if ans == "done":
                break
        os.write(master, b"\n")
        sent = True
    if p.poll() is not None and not r:
        break
print("\n[helper] finished:", "MOUNTED" if b"Mounted at" in buf else "not mounted (send me the output)")
