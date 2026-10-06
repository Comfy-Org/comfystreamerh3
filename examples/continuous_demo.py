#!/usr/bin/env python3
"""Launch the continuous HLS viewer against a ready Comfy Platform GPU."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
import webbrowser
from pathlib import Path

from basic_demo import WORKFLOW_PATH, discover_ready_gpu_profiles


ROOT = Path(__file__).resolve().parents[1]
COMFYSTREAM = ROOT.parent / "comfystream"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _flow_settings(prompt: str) -> str:
    continuation = f"Continue the same scene: {prompt}"
    return f'''[flow]
id = "comfystreamerh3-basic-live"
display_name = "ComfyStreamerH3 Live Demo"
description = "Continuous video from the ComfyStreamerH3 basic prompt."
phases = ["play", "consequence"]
play_phases = ["play", "consequence"]
commentary_phases = []
phase_duration_seconds = 15
sequences_per_run = 6
prebuffer_clips = 2
handoff_phase = "consequence"
cycle_key = "story_seed"
cycle_values = [{_toml_string(prompt)}]

[prompts]
play = {_toml_string(prompt)}
consequence = {_toml_string(continuation)}

[dialogue]
require_llm = false
forbidden_dialogue_tokens = []

[director]
default_prompt = {_toml_string(prompt)}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the continuous H3 live viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not (COMFYSTREAM / "src" / "comfystream" / "__main__.py").is_file():
        raise SystemExit("The ComfyStreamer continuous runtime checkout is missing beside this repository.")
    profiles = discover_ready_gpu_profiles()
    if not profiles:
        raise SystemExit("No ready RTX 5090 or RTX 6000 Pro deployment was found by Comfy CLI.")
    default_resolution = "512x320" if "512x320" in profiles else "448x256"
    default_profile = profiles[default_resolution]

    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((args.host, args.port)) == 0:
            raise SystemExit(f"Port {args.port} is already in use. Stop the current demo first.")

    graph = json.loads(WORKFLOW_PATH.read_text())["prompt"]
    prompt = graph["5"]["inputs"]["prompt"]
    state_dir = Path(tempfile.mkdtemp(prefix="comfystreamerh3-live-"))
    flow_path = state_dir / "flow.toml"
    flow_path.write_text(_flow_settings(prompt))
    env = os.environ.copy()
    env["COMFYSTREAM_H3_TEXT_ENCODER_PROFILE"] = "clipproj_4b_v31_mlp"
    env["COMFYSTREAM_RESOLUTION_PROFILES_JSON"] = json.dumps(profiles)
    env["COMFYSTREAM_DEFAULT_RESOLUTION"] = default_resolution
    command = [
        "uv", "run", "--no-sync", "--with", "comfy-cli==1.22.0",
        "python", "-m", "comfystream",
        "--flow-settings", str(flow_path),
        "--renderer", "platform",
        "--platform-deployment-url", default_profile["deploymentUrl"],
        "--topology", "separate_pods",
        "--director-provider", "disabled",
        "--host", args.host,
        "--port", str(args.port),
        "--state-path", str(state_dir / "state.json"),
        "--media-root", str(state_dir / "media"),
    ]
    process = subprocess.Popen(command, cwd=COMFYSTREAM, env=env)
    base_url = f"http://{args.host}:{args.port}"
    try:
        for _ in range(120):
            if process.poll() is not None:
                return process.returncode or 1
            try:
                with urllib.request.urlopen(base_url + "/healthz", timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit("The continuous ComfyStreamer app did not become healthy within 120 seconds.")

        connection_path = state_dir / "bootstrap" / "connection-v1.json"
        try:
            operator_token = json.loads(connection_path.read_text())["operator_token"]
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise SystemExit("ComfyStreamer did not provide its local operator token.") from error
        browser_url = f"{base_url}/live#operator={operator_token}"
        print("Ready GPU profiles: " + ", ".join(profile["gpuLabel"] for profile in profiles.values()), flush=True)
        print(f"Continuous video UI: {base_url}/live", flush=True)
        if not args.no_browser:
            webbrowser.open_new_tab(browser_url)
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        return 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
