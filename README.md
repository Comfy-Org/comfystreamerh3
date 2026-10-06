# ComfyStreamerH3

The most optimized live-video-capable custom node for ComfyUI, built to run
MiniMax-H3 with ComfyStream's FastH3 runtime on budget-friendly RTX 5090 and
RTX 6000 Pro instances for $4/hour or less.

Generate live video at 448 × 256 on a single RTX 5090, then upscale it
for higher-resolution output.

## Install

Clone the repository into ComfyUI's `custom_nodes` directory and restart
ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Comfy-Org/comfystreamerh3.git
```

## Runtime and models

The managed runtime targets Linux with an NVIDIA GPU, Python 3.12, CUDA 13
PyTorch, matching TorchVision and TorchAudio builds, and `comfy-kitchen` 0.2.34.
ComfyStream's [build spec](https://github.com/Comfy-Org/comfystreamer/blob/main/deploy/comfy-build-shared-obs.yaml)
pins the runtime and model assets. The node repository does not include model
weights.

| Model | ComfyUI folder | Managed build filename |
|---|---|---|
| FastH3 checkpoint | `models/diffusion_models/` | `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` |
| Text encoder | `models/text_encoders/` | `qwen3vl_4b_fp8_scaled.safetensors` |
| ClipProj MLP | `models/clip_projections/` | `mmh3-4b-ClipProj-v3.1-mlp.safetensors` |
| Audio VAE | `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` |
| Video VAE | `models/vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` |

## Standard workflow

1. **ComfyStreamerH3 Optimized Loader** loads a FastH3 preset and provides the
   model, sampler, sigmas, and profile.
2. **ComfyStreamerH3 Image to Video** sets the prompt, output size, and frame
   count. First/last frames and reference images are optional.
3. **ComfyStreamerH3 Sampler** generates video and audio latents.
4. **ComfyStreamerH3 Video Decode** and **ComfyStreamerH3 Audio Decode** decode
   the latents.
5. **ComfyStreamerH3 Output** creates a file-backed `VIDEO` for preview,
   saving, or downstream nodes.

For a minimal text-to-video graph with model setup and exact socket connections,
see the [Quick start](QUICKSTART.md).

## Live throughput and sample clips

On one RTX 5090, a warm 448×256 run produced two 15.08-second clips in a
median **23.51 seconds** across 11 jobs: **1.28× real time**. The run used
four-step VSA. See the
[ComfyStreamer benchmark and method](https://github.com/Comfy-Org/comfystreamer/blob/main/benchmark-results/live-threegpu-lium-20261004/RESULTS.md).

These four 15-second clips use the prompt “Will Smith eating spaghetti”; each
uses a different visual style. They are examples, not the throughput runs
above. Output: 448×256, 24 fps.

| Style | Clip |
|---|---|
| Photorealistic | <a href="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/photorealistic.mp4"><img src="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/photorealistic.preview.png" alt="Photorealistic Will Smith eating spaghetti; open clip" width="240"></a> |
| Anime | <a href="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/anime.mp4"><img src="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/anime.preview.png" alt="Anime-style Will Smith eating spaghetti; open clip" width="240"></a> |
| Stylized 3D | <a href="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/stylized-3d.mp4"><img src="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/stylized-3d.preview.png" alt="Stylized 3D Will Smith eating spaghetti; open clip" width="240"></a> |
| Watercolor | <a href="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/watercolor.mp4"><img src="benchmark-results/fast-h3-15s-rtx5090-2026-10-06/watercolor.preview.png" alt="Watercolor Will Smith eating spaghetti; open clip" width="240"></a> |

## Deploy to the Comfy Developer Platform

The ComfyStream build packages this node with its pinned runtime and model
files. A local ComfyUI install or GPU is not required to build or deploy it.

1. Copy this repository's node pack into
   `comfystream/deploy/custom_nodes/fasth3_deploy/`. For example:

   ```sh
   rsync -a --delete --exclude='.git/' --exclude='benchmark-results/' \
     /path/to/comfystreamerh3/ \
     /path/to/comfystream/deploy/custom_nodes/fasth3_deploy/
   ```

   This replaces the pack directory with the files from this repository.
2. Make the ComfyStream build spec include the model assets listed above. The
   build spec controls which weights are included. Update its model check to
   match too: the current ComfyStream helper still expects the 32B text encoder,
   while this node uses Qwen 4B and ClipProj.
3. Sign in with the Comfy CLI using an account with Developer Platform access.
   Set these cost limits in the same shell before starting the deployment:

   | Variable | Purpose |
   |---|---|
   | `COMFYSTREAM_BUILD_ESTIMATE_USD` | Estimated build cost. |
   | `COMFYSTREAM_BUILD_BUDGET_USD` | Maximum allowed build estimate. |
   | `COMFYSTREAM_GPU_HOURLY_USD` | Declared hourly rate per GPU. |
   | `COMFYSTREAM_OVERLAP_BUDGET_USD` | Maximum cost for overlapping workers. |
   | `COMFYSTREAM_OVERLAP_MAX_SECONDS` | Maximum overlap duration. |

4. From the ComfyStream repository root, create or resume the deployment and
   check the worker status and URL:

```sh
cd /path/to/comfystream
./scripts/deploy_comfystreamer.sh start
./scripts/deploy_comfystreamer.sh status
```

The helper uploads the node pack and pinned runtime, creates a Linux/NVIDIA
release when needed, and waits for the worker to become ready. Run
`./scripts/deploy_comfystreamer.sh stop` to stop the GPU worker while retaining
the deployment and storage. Run `./scripts/deploy_comfystreamer.sh remove` to
delete the cloud deployment and build while keeping the local build spec.
Running workers incur GPU charges; stopped deployments retain storage.
