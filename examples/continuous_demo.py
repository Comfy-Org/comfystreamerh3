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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "examples" / "basic_text_to_video_api.json"
COMFYSTREAM = ROOT.parent / "comfystream"


def _cli_error(stdout: str, stderr: str) -> str:
    for line in reversed(stdout.splitlines()):
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            continue
        error = envelope.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return (stderr or stdout).strip()[-1500:]


def _gpu_resolution(gpu_class: object) -> str | None:
    if not isinstance(gpu_class, str):
        return None
    normalized = gpu_class.lower().replace("_", "-")
    if "5090" in normalized:
        return "448x256"
    if "6000" in normalized:
        return "512x320"
    return None


def _gpu_name(resolution: str) -> str:
    return "RTX 5090" if resolution == "448x256" else "RTX 6000 Pro"


def _ready_gpu_profiles(deployments: list[object]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[tuple[str, str, str]]] = {"448x256": [], "512x320": []}
    for deployment in deployments:
        if not isinstance(deployment, dict) or deployment.get("status") != "ready":
            continue
        compute = deployment.get("computeConfig")
        gpu_class = compute.get("gpuClass") if isinstance(compute, dict) else None
        resolution = _gpu_resolution(gpu_class)
        deployment_id = deployment.get("id")
        deployment_url = deployment.get("endpointUrl")
        if (
            resolution
            and isinstance(deployment_id, str)
            and deployment_id
            and isinstance(deployment_url, str)
            and deployment_url
        ):
            candidates[resolution].append((
                str(deployment.get("updatedAt") or ""), deployment_id, deployment_url
            ))
    profiles: dict[str, dict[str, Any]] = {}
    for resolution, ready in candidates.items():
        if not ready:
            continue
        _, deployment_id, deployment_url = max(ready, key=lambda item: item[0])
        profiles[resolution] = {
            "deploymentId": deployment_id,
            "deploymentUrl": deployment_url,
            "width": 448 if resolution == "448x256" else 512,
            "height": 256 if resolution == "448x256" else 320,
            "gpuLabel": _gpu_name(resolution),
        }
    return profiles


def discover_ready_gpu_profiles() -> dict[str, dict[str, Any]]:
    """Find already-ready workspace GPUs through the logged-in Comfy CLI."""
    result = subprocess.run(
        ["comfy", "--json", "deploy", "ls", "--workspace", "--status", "ready"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(_cli_error(result.stdout, result.stderr) or "Comfy Platform discovery failed")
    envelope = json.loads(result.stdout)
    if not envelope.get("ok"):
        raise RuntimeError(_cli_error(result.stdout, result.stderr) or "Comfy Platform discovery failed")
    deployments = envelope["data"]["deployments"]
    if not isinstance(deployments, list):
        raise ValueError("Comfy CLI returned an invalid deployment list")
    return _ready_gpu_profiles(deployments)


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
    # The legacy COMFY_API_KEY is a ComfyUI/API-node key, not the account OAuth
    # credential required by `comfy deploy show/run`; let the CLI use its login.
    env.pop("COMFY_API_KEY", None)
    env.pop("COMFY_CLOUD_API_KEY", None)
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
