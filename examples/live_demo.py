#!/usr/bin/env python3
"""Start the sibling ComfyStream FastH3 demo and open its plain video page."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
import webbrowser
from pathlib import Path


def main() -> int:
    node_repo = Path(__file__).resolve().parents[1]
    coordinator = node_repo.parent / "comfystream"
    if not (coordinator / "src" / "comfystream" / "__main__.py").is_file():
        raise SystemExit("Clone ComfyStream beside this repo: ../comfystream")

    host = "127.0.0.1"
    port = int(os.environ.get("COMFYSTREAM_PORT", "8765"))
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((host, port)) == 0:
            raise SystemExit(f"Port {port} is already in use. Stop the other local demo first.")

    state_dir = Path(tempfile.mkdtemp(prefix="comfystream-fast-h3-"))
    env = os.environ.copy()
    env["COMFYSTREAM_H3_TEXT_ENCODER_PROFILE"] = "clipproj_4b_v31_mlp"
    gpu_url = env.get("COMFYUI_URL", "http://127.0.0.1:8188")
    command = [
        "uv", "run", "--no-sync", "python", "-m", "comfystream",
        "--renderer", "fasth3",
        "--gpu-endpoints", gpu_url,
        "--experimental-fasth3",
        "--generation-profile", "compact",
        "--topology", "separate_pods",
        "--director-provider", "disabled",
        "--host", host,
        "--port", str(port),
        "--state-path", str(state_dir / "state.json"),
        "--media-root", str(state_dir / "media"),
    ]
    process = subprocess.Popen(command, cwd=coordinator, env=env)
    player_url = f"http://{host}:{port}/live"
    try:
        for _ in range(60):
            exit_code = process.poll()
            if exit_code is not None:
                return exit_code
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            process.terminate()
            raise SystemExit("ComfyStream did not start within 60 seconds.")

        print(f"Live video: {player_url}", flush=True)
        webbrowser.open_new_tab(player_url)
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
