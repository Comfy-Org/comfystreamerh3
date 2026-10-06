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
| Text encoder | `models/text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
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

## Deploy to the Comfy Developer Platform

ComfyStream maintains the deployment build and packages its custom-node source
with the pinned runtime and model assets. Integrate changes from this repository
into the ComfyStream build before deploying. From the ComfyStream repository
root, follow the [deployment guide](https://github.com/Comfy-Org/comfystreamer/blob/main/deploy/README.md#deploy-the-current-pack)
to set cost controls, then create or resume the deployment and check its status:

```sh
./scripts/deploy_comfystreamer.sh start
./scripts/deploy_comfystreamer.sh status
```

Use `./scripts/deploy_comfystreamer.sh stop` to stop workers and keep the
deployment for later. Running GPU workers incur charges.
