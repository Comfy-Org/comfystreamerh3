#!/usr/bin/env python3
"""Run the self-contained continuous Comfy Platform HLS demo."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "examples" / "basic_text_to_video_api.json"
HLS_PLAYER_PATH = ROOT / "examples" / "vendor" / "hls.light.min.js"
PROMPT_NODE = "5"
SAMPLER_NODE = "8"
OUTPUT_NODE = "12"
MAX_PROMPT_LENGTH = 12_000
MAX_PLAYLIST_SEGMENTS = 180
# AAC priming and trailing packets can extend beyond the video EXTINF duration.
# Keep the next clip's DTS clear of that encoder padding for both media tracks.
CLIP_BOUNDARY_SAFETY_SECONDS = 0.1
RETIRED_SEGMENT_GRACE_SECONDS = 60
RESOLUTION_PROFILES = {
    "448x256": {"width": 448, "height": 256, "gpuLabel": "RTX 5090"},
    "512x320": {"width": 512, "height": 320, "gpuLabel": "RTX 6000 Pro"},
}


PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#090b0f">
  <title>ComfyStreamerH3 Live</title>
  <script src="/static/hls.light.min.js"></script>
  <style>
    :root { color-scheme: dark; font: 14px system-ui, sans-serif; background: #090b0f; color: #f3f4f2; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main { width: min(calc(100% - 32px), 1100px); margin: 0 auto; padding: 24px 0; }
    header { margin-bottom: 12px; color: #aab0ba; font-size: 11px; letter-spacing: .12em; }
    h1 { margin: 4px 0 0; color: #f3f4f2; font-size: 24px; letter-spacing: -.03em; }
    video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #000; border: 1px solid #2d3138; object-fit: contain; }
    form { margin-top: 12px; padding: 12px; border: 1px solid #2d3138; background: #0d1014; }
    label { display: block; margin-bottom: 7px; color: #b0b6c0; font-size: 11px; }
    .prompt-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(170px, auto) auto; align-items: end; gap: 10px; }
    textarea, select, button { min-height: 56px; padding: 10px; color: #f3f4f2; background: #090b0f; border: 1px solid #343941; border-radius: 3px; font: inherit; }
    textarea { width: 100%; resize: vertical; font-size: 12px; line-height: 1.45; }
    select { font-size: 12px; }
    .resolution-control { display: grid; gap: 6px; margin: 0; }
    .resolution-options { display: flex; align-items: center; gap: 8px; }
    #gpu-label, #status { color: #b0b6c0; font-size: 12px; white-space: nowrap; }
    button { cursor: pointer; }
    button:disabled { cursor: wait; opacity: .65; }
    #status { display: block; min-height: 16px; margin-top: 8px; color: #9298a1; }
    @media (max-width: 680px) {
      main { width: calc(100% - 16px); padding: 10px 0; }
      .prompt-row { grid-template-columns: 1fr; }
      textarea, select, button { min-height: 42px; }
      .prompt-row button { justify-self: start; }
    }
  </style>
</head>
<body>
  <main>
    <header>COMFYSTREAMERH3<h1>Continuous live video</h1></header>
    <video id="video" muted playsinline controls></video>
    <form id="prompt-form">
      <label for="prompt">Prompt</label>
      <div class="prompt-row">
        <textarea id="prompt" rows="2" maxlength="12000" required></textarea>
        <label class="resolution-control" for="resolution">Resolution
          <span class="resolution-options">
            <select id="resolution">
              <option value="448x256">448 × 256</option>
              <option value="512x320">512 × 320</option>
            </select>
            <span id="gpu-label"></span>
          </span>
        </label>
        <button id="submit" type="submit">Run prompt</button>
      </div>
      <small id="status" role="status">Finding an existing ready GPU…</small>
    </form>
  </main>
  <script>
    const video = document.querySelector('#video');
    const form = document.querySelector('#prompt-form');
    const promptInput = document.querySelector('#prompt');
    const resolution = document.querySelector('#resolution');
    const gpuLabel = document.querySelector('#gpu-label');
    const submit = document.querySelector('#submit');
    const status = document.querySelector('#status');
    let started = false;
    let resolutionInitialized = false;
    let player = null;
    let retryTimer = null;
    let retryCount = 0;
    let playbackError = '';
    let profileMap = {};

    function updateGpuLabel() {
      gpuLabel.textContent = profileMap[resolution.value]?.gpuLabel || 'not running';
    }
    function startPlayback() {
      if (started) return;
      started = true;
      const source = `/live.m3u8?client=${Date.now()}`;
      if (window.Hls && Hls.isSupported()) {
        player = new Hls({
          liveSyncDuration: 30, liveMaxLatencyDuration: Infinity,
          maxBufferLength: 90, maxMaxBufferLength: 120, backBufferLength: 30,
        });
        player.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(() => {}));
        player.on(Hls.Events.ERROR, (_event, data) => {
          if (data.fatal) recoverPlayback('Reconnecting video…');
        });
        player.attachMedia(video);
        player.loadSource(source);
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = source;
        video.play().catch(() => {});
      } else {
        playbackError = 'This browser does not support HLS playback.';
      }
    }
    function recoverPlayback(message) {
      playbackError = message;
      status.textContent = message;
      if (retryTimer) return;
      retryTimer = window.setTimeout(() => {
        retryTimer = null;
        if (player) player.destroy();
        player = null;
        started = false;
        startPlayback();
      }, Math.min(1000 * 2 ** retryCount++, 30000));
    }
    video.addEventListener('error', () => {
      recoverPlayback('Reconnecting video…');
    });
    video.addEventListener('playing', () => {
      playbackError = '';
      retryCount = 0;
      if (retryTimer) window.clearTimeout(retryTimer);
      retryTimer = null;
    });
    async function refresh() {
      try {
        const state = await (await fetch('/api/state', {cache: 'no-store'})).json();
        profileMap = state.profiles || {};
        for (const option of resolution.options) {
          option.disabled = !profileMap[option.value];
          option.textContent = option.value.replace('x', ' × ')
            + (option.disabled ? ' (not running)' : '');
        }
        if (!resolutionInitialized && state.defaultResolution && profileMap[state.defaultResolution]) {
          resolution.value = state.defaultResolution;
          resolutionInitialized = true;
        }
        if (!profileMap[resolution.value]) {
          resolution.value = Object.keys(profileMap)[0] || '';
        }
        if (state.prompt && promptInput.value === '') promptInput.value = state.prompt;
        updateGpuLabel();
        if (state.playable) startPlayback();
        const message = state.error || state.status;
        status.textContent = playbackError || message;
      } catch (error) {
        status.textContent = `Local demo error: ${error.message}`;
      }
    }
    resolution.addEventListener('change', updateGpuLabel);
    form.addEventListener('submit', async event => {
      event.preventDefault();
      submit.disabled = true;
      status.textContent = 'Queueing prompt…';
      try {
        const response = await fetch('/api/run', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({prompt: promptInput.value, resolution: resolution.value}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
        status.textContent = result.message;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        submit.disabled = false;
        await refresh();
      }
    });
    void refresh();
    window.setInterval(refresh, 1500);
  </script>
</body>
</html>'''


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


def _ready_gpu_profiles(deployments: list[object]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[tuple[str, str, str]]] = {key: [] for key in RESOLUTION_PROFILES}
    for deployment in deployments:
        if not isinstance(deployment, dict) or deployment.get("status") != "ready":
            continue
        compute = deployment.get("computeConfig")
        gpu_class = compute.get("gpuClass") if isinstance(compute, dict) else None
        resolution = _gpu_resolution(gpu_class)
        deployment_id = deployment.get("id")
        endpoint_url = deployment.get("endpointUrl")
        if (
            resolution in candidates
            and isinstance(deployment_id, str) and deployment_id
            and isinstance(endpoint_url, str) and endpoint_url
        ):
            candidates[resolution].append((str(deployment.get("updatedAt") or ""), deployment_id, endpoint_url))
    profiles: dict[str, dict[str, Any]] = {}
    for resolution, entries in candidates.items():
        if not entries:
            continue
        _, deployment_id, endpoint_url = max(entries, key=lambda entry: entry[0])
        profiles[resolution] = {
            **RESOLUTION_PROFILES[resolution],
            "deploymentId": deployment_id,
            "deploymentUrl": endpoint_url,
        }
    return profiles


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("COMFY_API_KEY", None)
    env.pop("COMFY_CLOUD_API_KEY", None)
    return env


def discover_ready_gpu_profiles() -> dict[str, dict[str, Any]]:
    """Find existing ready workspace GPUs with the user's Comfy CLI login."""
    result = subprocess.run(
        ["comfy", "--json", "deploy", "ls", "--workspace", "--status", "ready"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError(_cli_error(result.stdout, result.stderr) or "Comfy Platform discovery failed")
    envelope = json.loads(result.stdout)
    if not envelope.get("ok"):
        raise RuntimeError(_cli_error(result.stdout, result.stderr) or "Comfy Platform discovery failed")
    deployments = envelope.get("data", {}).get("deployments")
    if not isinstance(deployments, list):
        raise ValueError("Comfy CLI returned an invalid deployment list")
    return _ready_gpu_profiles(deployments)


class ContinuousDemo:
    def __init__(self, profiles: dict[str, dict[str, Any]], default_prompt: str, root: Path):
        self.profiles = profiles
        self.default_resolution = "512x320" if "512x320" in profiles else "448x256"
        self.root = root
        self.media_root = root / "media"
        self.media_root.mkdir(parents=True)
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.requested = {"prompt": default_prompt, "resolution": self.default_resolution}
        self.entries: list[tuple[int, float, str, bool]] = []
        self.retired_segments: list[tuple[float, Path]] = []
        self.next_sequence = 0
        self.discontinuity_sequence = 0
        self.total_duration_seconds = 0.0
        self.previous_resolution: str | None = None
        self.attempt_count = 0
        self.clip_count = 0
        self.status = "Generating the default prompt…"
        self.error: str | None = None
        self.active_process: subprocess.Popen[str] | None = None
        self.worker = threading.Thread(target=self._run_forever, name="h3-continuous-generator", daemon=True)

    def start(self) -> None:
        self.worker.start()

    def request(self, prompt: str, resolution: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"prompt must contain 1 to {MAX_PROMPT_LENGTH} characters")
        if resolution not in self.profiles:
            raise ValueError("select a resolution with a ready GPU")
        with self.lock:
            self.requested = {"prompt": prompt.strip(), "resolution": resolution}
            self.status = "Prompt queued; current clip will finish first."
            self.error = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "profiles": {
                    key: {"width": value["width"], "height": value["height"], "gpuLabel": value["gpuLabel"]}
                    for key, value in self.profiles.items()
                },
                "defaultResolution": self.requested["resolution"],
                "prompt": self.requested["prompt"],
                "status": self.status,
                "error": self.error,
                "playable": self.clip_count >= 2,
                "clips": self.clip_count,
                "segments": len(self.entries),
            }

    def playlist(self) -> str:
        with self.lock:
            entries = list(self.entries)
            playable = self.clip_count >= 2
            discontinuity_sequence = self.discontinuity_sequence
        if not playable or not entries:
            return "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n"
        target = max(2, math.ceil(max(entry[1] for entry in entries)))
        lines = [
            "#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{target}",
            f"#EXT-X-MEDIA-SEQUENCE:{entries[0][0]}",
            f"#EXT-X-DISCONTINUITY-SEQUENCE:{discontinuity_sequence}",
            "#EXT-X-INDEPENDENT-SEGMENTS",
        ]
        for _sequence, duration, uri, discontinuity in entries:
            if discontinuity:
                lines.append("#EXT-X-DISCONTINUITY")
            lines.extend((f"#EXTINF:{duration:.3f},", uri))
        return "\n".join(lines) + "\n"

    def stop(self) -> None:
        self.stopping.set()
        with self.lock:
            process = self.active_process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.worker.is_alive():
            self.worker.join()

    def _run_process(
        self, command: list[str], *, timeout: float, env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with self.lock:
            if self.stopping.is_set():
                raise InterruptedError("demo is stopping")
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.active_process = process
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError(f"{command[0]} timed out after {timeout:g} seconds")
        finally:
            with self.lock:
                self.active_process = None
        if self.stopping.is_set():
            raise InterruptedError("demo is stopping")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def _run_forever(self) -> None:
        while not self.stopping.is_set():
            with self.lock:
                request = dict(self.requested)
                self.attempt_count += 1
                clip_number = self.attempt_count
                self.status = f"Generating {request['resolution']} on {self.profiles[request['resolution']]['gpuLabel']}…"
                self.error = None
            work = self.root / f"clip-{clip_number:06d}"
            try:
                work.mkdir()
                video = self._generate(request, work)
                with self.lock:
                    timestamp_offset = (
                        self.total_duration_seconds
                        + self.clip_count * CLIP_BOUNDARY_SAFETY_SECONDS
                    )
                published = self._segment(video, clip_number, timestamp_offset)
                with self.lock:
                    self.clip_count += 1
                    discontinuity = (
                        self.previous_resolution is not None
                        and request["resolution"] != self.previous_resolution
                    )
                    for index, duration, uri in published:
                        self.entries.append((self.next_sequence, duration, uri, discontinuity and index == 0))
                        self.next_sequence += 1
                    self.total_duration_seconds += sum(duration for _, duration, _ in published)
                    self.previous_resolution = request["resolution"]
                    if len(self.entries) > MAX_PLAYLIST_SEGMENTS:
                        removed = self.entries[:-MAX_PLAYLIST_SEGMENTS]
                        self.entries = self.entries[-MAX_PLAYLIST_SEGMENTS:]
                        self.discontinuity_sequence += sum(1 for entry in removed if entry[3])
                        for _, _, uri, _ in removed:
                            self.retired_segments.append((
                                time.monotonic() + RETIRED_SEGMENT_GRACE_SECONDS,
                                self.root / uri,
                            ))
                    now = time.monotonic()
                    remaining = []
                    for deadline, path in self.retired_segments:
                        if deadline <= now:
                            path.unlink(missing_ok=True)
                        else:
                            remaining.append((deadline, path))
                    self.retired_segments = remaining
                    self.status = "Live — generating the next clip."
                    self.error = None
            except Exception as error:
                if self.stopping.is_set():
                    break
                with self.lock:
                    self.status = "Generation failed; retrying."
                    self.error = str(error)[:1200]
                if self.stopping.wait(5):
                    break
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def _generate(self, request: dict[str, str], work: Path) -> Path:
        workflow = json.loads(WORKFLOW_PATH.read_text())["prompt"]
        profile = self.profiles[request["resolution"]]
        inputs = workflow[PROMPT_NODE]["inputs"]
        inputs.update(prompt=request["prompt"], width=profile["width"], height=profile["height"])
        workflow[SAMPLER_NODE]["inputs"]["run_nonce"] = f"local-continuous-{uuid.uuid4().hex}"
        workflow[SAMPLER_NODE]["inputs"]["seed"] = secrets.randbelow(2**32)
        workflow[OUTPUT_NODE]["inputs"]["filename_prefix"] = f"h3-live/{uuid.uuid4().hex}"
        workflow_path = work / "workflow.json"
        output_dir = work / "outputs"
        workflow_path.write_text(json.dumps(workflow))
        output_dir.mkdir()
        command = [
            "comfy", "--json", "deploy", "run", "--workflow", str(workflow_path),
            "--deployment", profile["deploymentId"], "--output-dir", str(output_dir), "--timeout", "900",
        ]
        result = self._run_process(command, env=_cli_env(), timeout=960)
        if result.returncode:
            raise RuntimeError(_cli_error(result.stdout, result.stderr) or "Comfy Platform generation failed")
        videos = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4")
        if not videos:
            raise RuntimeError("Comfy Platform completed without returning an MP4")
        return videos[0]

    def _segment(self, video: Path, clip_number: int, timestamp_offset: float) -> list[tuple[int, float, str]]:
        clip_dir = self.media_root / f"clip-{clip_number:06d}"
        clip_dir.mkdir()
        local_playlist = clip_dir / "clip.m3u8"
        segment_pattern = clip_dir / "segment-%05d.ts"
        result = self._run_process([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", "24", "-keyint_min", "24",
            "-sc_threshold", "0", "-c:a", "aac", "-b:a", "128k", "-f", "hls",
            "-hls_time", "1", "-hls_list_size", "0", "-hls_flags", "independent_segments",
            "-output_ts_offset", f"{timestamp_offset:.6f}",
            "-hls_segment_filename", str(segment_pattern), str(local_playlist),
        ], timeout=300)
        if result.returncode:
            raise RuntimeError((result.stderr or "FFmpeg could not segment the generated video")[-1200:])
        published: list[tuple[int, float, str]] = []
        lines = local_playlist.read_text().splitlines()
        for index, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            segment = (clip_dir / line).resolve()
            if not segment.is_relative_to(self.media_root.resolve()) or not segment.is_file():
                raise RuntimeError("FFmpeg returned an invalid segment path")
            duration_line = None
            if index > 0 and lines[index - 1].startswith("#EXTINF:"):
                duration_line = lines[index - 1]
            duration = float(duration_line.split(":", 1)[1].split(",", 1)[0]) if duration_line else 1.0
            uri = segment.relative_to(self.root.resolve()).as_posix()
            published.append((len(published), duration, uri))
        if not published:
            raise RuntimeError("FFmpeg produced no HLS segments")
        return published


def make_handler(demo: ContinuousDemo) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if path in {"/", "/live"}:
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._send(200, b'{"ok":true}', "application/json")
            elif path == "/static/hls.light.min.js":
                self._send(200, HLS_PLAYER_PATH.read_bytes(), "application/javascript")
            elif path == "/api/state":
                self._json(200, demo.snapshot())
            elif path == "/live.m3u8":
                self._send(200, demo.playlist().encode(), "application/vnd.apple.mpegurl")
            elif path.startswith("/media/"):
                relative = urllib.parse.unquote(path.removeprefix("/media/"))
                target = (demo.media_root / relative).resolve()
                if not target.is_relative_to(demo.media_root.resolve()):
                    self.send_error(404)
                    return
                try:
                    body = target.read_bytes()
                except (FileNotFoundError, IsADirectoryError):
                    self.send_error(404)
                    return
                self._send(200, body, "video/mp2t")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if urllib.parse.urlsplit(self.path).path != "/api/run":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_PROMPT_LENGTH + 256:
                    raise ValueError("request body is missing or too large")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("request body must be an object")
                demo.request(request.get("prompt"), request.get("resolution"))
                self._json(202, {"message": "Prompt queued; current clip will finish first."})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})

        def _json(self, status: int, value: Any) -> None:
            self._send(status, json.dumps(value).encode(), "application/json; charset=utf-8")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the self-contained continuous H3 live viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if shutil.which("comfy") is None:
        raise SystemExit("Install and sign in to comfy-cli before starting this demo.")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Install ffmpeg before starting this demo.")
    if not HLS_PLAYER_PATH.is_file():
        raise SystemExit("The repo-local HLS player file is missing.")
    profiles = discover_ready_gpu_profiles()
    if not profiles:
        raise SystemExit("No ready RTX 5090 or RTX 6000 Pro deployment was found by Comfy CLI.")
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((args.host, args.port)) == 0:
            raise SystemExit(f"Port {args.port} is already in use. Stop the current demo first.")

    workflow = json.loads(WORKFLOW_PATH.read_text())["prompt"]
    default_prompt = workflow[PROMPT_NODE]["inputs"]["prompt"]
    state_dir = Path(tempfile.mkdtemp(prefix="comfystreamerh3-live-"))
    demo = ContinuousDemo(profiles, default_prompt, state_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    base_url = f"http://{args.host}:{args.port}"
    demo.start()
    print("Ready GPU profiles: " + ", ".join(profile["gpuLabel"] for profile in profiles.values()), flush=True)
    print(f"Continuous video UI: {base_url}/live", flush=True)
    if not args.no_browser:
        webbrowser.open_new_tab(base_url + "/live")
    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        demo.stop()
        shutil.rmtree(state_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
