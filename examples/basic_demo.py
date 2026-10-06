#!/usr/bin/env python3
"""Serve a tiny prompt-and-play page for the basic ComfyStreamerH3 workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "examples" / "basic_text_to_video_api.json"
PAGE_PATH = ROOT / "examples" / "basic_demo.html"
PROMPT_NODE = "5"


class DemoHandler(BaseHTTPRequestHandler):
    comfy_url = "http://127.0.0.1:8188"
    platform_mode = False
    deployments_by_resolution: dict[str, str] = {}
    default_resolution = "448x256"
    page = b""
    workflow: dict[str, Any] = {}
    workflow_lock = threading.Lock()
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    output_files: dict[str, Path] = {}

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path)
        if path.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.page)))
            self.end_headers()
            self.wfile.write(self.page)
            return
        if path.path.startswith("/api/history/"):
            prompt_id = path.path.removeprefix("/api/history/")
            if not prompt_id or "/" in prompt_id:
                self.send_error(400, "invalid prompt id")
                return
            if self.platform_mode:
                self._platform_history(prompt_id)
                return
            self._proxy_json(f"/history/{urllib.parse.quote(prompt_id, safe='')}")
            return
        if path.path == "/api/view":
            self._proxy_view(path.query)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urllib.parse.urlsplit(self.path).path != "/api/prompt":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 33_000:
                raise ValueError("request body is too large or empty")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("request body must be an object")
            prompt = request.get("prompt")
            resolution = request.get("resolution")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32_000:
                raise ValueError("prompt must contain 1 to 32000 characters")
            dimensions = {"448x256": (448, 256), "512x320": (512, 320)}
            if resolution not in dimensions:
                raise ValueError("unsupported resolution")
            width, height = dimensions[resolution]
            with self.workflow_lock:
                workflow = json.loads(json.dumps(self.workflow))
            inputs = workflow["prompt"][PROMPT_NODE]["inputs"]
            inputs.update(prompt=prompt.strip(), width=width, height=height)
            workflow["prompt"]["8"]["inputs"]["run_nonce"] = f"basic-demo-{uuid.uuid4().hex}"
            deployment_id = self.deployments_by_resolution.get(resolution)
            if self.platform_mode and deployment_id is None:
                raise ValueError(f"No ready Comfy Platform GPU is available for {resolution}")
            if deployment_id is not None:
                prompt_id = uuid.uuid4().hex
                output_dir = Path(tempfile.mkdtemp(prefix=f"comfystreamerh3-{prompt_id}-"))
                workflow_path = output_dir / "workflow.json"
                # `comfy deploy run --workflow` takes the raw API node graph;
                # only the direct HTTP `/prompt` endpoint needs the outer wrapper.
                workflow_path.write_text(json.dumps(workflow["prompt"]))
                with self.jobs_lock:
                    self.jobs[prompt_id] = {"state": "running", "filename": None, "error": None}
                threading.Thread(
                    target=self._run_platform_job,
                    args=(prompt_id, workflow_path, output_dir, deployment_id),
                    daemon=True,
                ).start()
                self._send_json(200, {"prompt_id": prompt_id})
                return
            body = json.dumps(workflow).encode("utf-8")
            upstream = urllib.request.Request(
                f"{self.comfy_url}/prompt",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(upstream, timeout=30) as response:
                result = json.loads(response.read())
            self._send_json(200, result)
        except urllib.error.HTTPError as error:
            self._send_json(error.code, self._read_error(error))
        except (urllib.error.URLError, TimeoutError) as error:
            self._send_json(502, {"error": f"Could not reach ComfyUI: {error}"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})

    def _proxy_json(self, path: str) -> None:
        try:
            with urllib.request.urlopen(f"{self.comfy_url}{path}", timeout=30) as response:
                self._send_json(response.status, json.loads(response.read()))
        except urllib.error.HTTPError as error:
            self._send_json(error.code, self._read_error(error))
        except (urllib.error.URLError, TimeoutError) as error:
            self._send_json(502, {"error": f"Could not reach ComfyUI: {error}"})
        except json.JSONDecodeError:
            self._send_json(502, {"error": "ComfyUI returned invalid JSON"})

    def _proxy_view(self, query: str) -> None:
        values = urllib.parse.parse_qs(query)
        filename = values.get("filename", [""])[0]
        subfolder = values.get("subfolder", [""])[0]
        kind = values.get("type", ["output"])[0]
        if not filename or kind != "output" or "/" in filename or "\\" in filename:
            self.send_error(400, "invalid output reference")
            return
        if self.platform_mode:
            self._serve_platform_video(filename)
            return
        upstream_query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": kind}
        )
        try:
            headers = {}
            if byte_range := self.headers.get("Range"):
                headers["Range"] = byte_range
            request = urllib.request.Request(
                f"{self.comfy_url}/view?{upstream_query}", headers=headers
            )
            with urllib.request.urlopen(
                request, timeout=60
            ) as response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "video/mp4"))
                if content_length := response.headers.get("Content-Length"):
                    self.send_header("Content-Length", content_length)
                if content_range := response.headers.get("Content-Range"):
                    self.send_header("Content-Range", content_range)
                self.send_header("Accept-Ranges", response.headers.get("Accept-Ranges", "bytes"))
                self.end_headers()
                while chunk := response.read(1024 * 1024):
                    self.wfile.write(chunk)
        except urllib.error.HTTPError as error:
            self._send_json(error.code, self._read_error(error))
        except (urllib.error.URLError, TimeoutError) as error:
            self._send_json(502, {"error": f"Could not load video: {error}"})

    def _platform_history(self, prompt_id: str) -> None:
        with self.jobs_lock:
            job = self.jobs.get(prompt_id)
        if job is None:
            self._send_json(404, {"error": "unknown demo run"})
            return
        state = job["state"]
        outputs: dict[str, Any] = {}
        if state == "completed":
            outputs["12"] = {
                "images": [{"filename": job["filename"], "subfolder": "", "type": "output"}]
            }
        status: dict[str, Any] = {
            "completed": state in {"completed", "failed"},
            "status_str": "success" if state == "completed" else "error" if state == "failed" else "running",
            "messages": [["execution_error", job["error"]]] if state == "failed" else [],
        }
        self._send_json(200, {prompt_id: {"outputs": outputs, "status": status}})

    def _run_platform_job(
        self, prompt_id: str, workflow_path: Path, output_dir: Path, deployment_id: str
    ) -> None:
        downloads_dir = output_dir / "outputs"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "comfy", "--json", "deploy", "run",
            "--workflow", str(workflow_path),
            "--deployment", deployment_id,
            "--output-dir", str(downloads_dir),
            "--timeout", "900",
        ]
        try:
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=960, check=False
            )
            if result.returncode != 0:
                detail = _cli_error(result.stdout, result.stderr)
                raise RuntimeError(detail or "Comfy Platform job failed")
            videos = sorted(
                path for path in (output_dir / "outputs").rglob("*")
                if path.is_file() and path.suffix.lower() == ".mp4"
            )
            if not videos:
                raise RuntimeError("Comfy Platform completed without returning an MP4")
            filename = f"{prompt_id}.mp4"
            with self.jobs_lock:
                self.output_files[filename] = videos[0]
                self.jobs[prompt_id] = {"state": "completed", "filename": filename, "error": None}
        except Exception as error:  # Keep platform failures visible in the page's status line.
            with self.jobs_lock:
                self.jobs[prompt_id] = {"state": "failed", "filename": None, "error": str(error)[:1500]}

    def _serve_platform_video(self, filename: str) -> None:
        with self.jobs_lock:
            path = self.output_files.get(filename)
        if path is None or not path.is_file():
            self.send_error(404, "video is not available")
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        byte_range = self.headers.get("Range")
        if byte_range and byte_range.startswith("bytes="):
            requested_start, _, requested_end = byte_range[6:].partition("-")
            start = int(requested_start or 0)
            end = min(int(requested_end), size - 1) if requested_end else size - 1
            if start > end or start >= size:
                self.send_error(416, "invalid byte range")
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _read_error(error: urllib.error.HTTPError) -> dict[str, str]:
        detail = error.read().decode("utf-8", "replace")
        return {"error": detail or str(error)}

    def _send_json(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ComfyStreamerH3 basic prompt demo.")
    parser.add_argument("--comfy-url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    parser.add_argument("--platform", action="store_true", help="submit jobs to the existing Comfy Platform deployment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    workflow = json.loads(WORKFLOW_PATH.read_text())
    DemoHandler.comfy_url = args.comfy_url.rstrip("/")
    DemoHandler.platform_mode = args.platform
    if args.platform:
        listing = subprocess.run(
            ["comfy", "--json", "deploy", "ls", "--workspace", "--status", "ready"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if listing.returncode != 0:
            raise SystemExit(_cli_error(listing.stdout, listing.stderr) or "Could not discover Comfy Platform deployments")
        try:
            envelope = json.loads(listing.stdout)
            if not envelope.get("ok"):
                raise ValueError(_cli_error(listing.stdout, listing.stderr) or "deployment discovery failed")
            deployments = envelope["data"]["deployments"]
            if not isinstance(deployments, list):
                raise ValueError("deployment discovery returned an invalid list")
            DemoHandler.deployments_by_resolution = _ready_gpu_deployments(deployments)
            if not DemoHandler.deployments_by_resolution:
                raise ValueError("no ready RTX 5090 or RTX 6000 Pro deployments were found")
            DemoHandler.default_resolution = (
                "448x256" if "448x256" in DemoHandler.deployments_by_resolution else "512x320"
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Could not discover a ready GPU deployment: {error}") from error
        print(
            "Ready GPU profiles: " + ", ".join(
                _gpu_name(resolution) for resolution in DemoHandler.deployments_by_resolution
            ),
            flush=True,
        )
    DemoHandler.workflow = workflow
    DemoHandler.page = _render_page(
        PAGE_PATH,
        platform_mode=DemoHandler.platform_mode,
        available_resolutions=tuple(DemoHandler.deployments_by_resolution),
        default_resolution=DemoHandler.default_resolution,
    )
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"ComfyStreamerH3 demo: {url}", flush=True)
    if not DemoHandler.platform_mode:
        print(f"ComfyUI endpoint: {DemoHandler.comfy_url}", flush=True)
    if not args.no_browser:
        webbrowser.open_new_tab(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


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


def _ready_gpu_deployments(deployments: list[object]) -> dict[str, str]:
    """Pick the most recently updated ready deployment for each supported GPU."""
    candidates: dict[str, list[tuple[str, str]]] = {"448x256": [], "512x320": []}
    for deployment in deployments:
        if not isinstance(deployment, dict) or deployment.get("status") != "ready":
            continue
        compute = deployment.get("computeConfig")
        gpu_class = compute.get("gpuClass") if isinstance(compute, dict) else None
        resolution = _gpu_resolution(gpu_class)
        deployment_id = deployment.get("id")
        if resolution and isinstance(deployment_id, str) and deployment_id:
            candidates[resolution].append((str(deployment.get("updatedAt") or ""), deployment_id))
    return {
        resolution: max(ready, key=lambda item: item[0])[1]
        for resolution, ready in candidates.items()
        if ready
    }


def _gpu_name(resolution: str) -> str:
    return "RTX 5090" if resolution == "448x256" else "RTX 6000 Pro"


def _render_page(
    path: Path, *, platform_mode: bool, available_resolutions: tuple[str, ...],
    default_resolution: str,
) -> bytes:
    config = {
        "platformMode": platform_mode,
        "availableResolutions": list(available_resolutions or ("448x256", "512x320")),
        "defaultResolution": default_resolution,
    }
    content = path.read_text().replace(
        "__DEMO_CONFIG__", json.dumps(config).replace("<", "\\u003c")
    )
    return content.encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
