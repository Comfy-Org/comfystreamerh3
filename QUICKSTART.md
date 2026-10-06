# Quick start: ComfyStream's fast H3 path

This small text-to-video graph follows the optimized FastH3 path used by ComfyStream's direct provider. It uses the B1 four-step VSA-20 preset at 448×256 for 362 frames (about 15 seconds at 24 fps), with QKV compilation, fused NVFP4 MLP when the GPU supports it, tiled GPU decode, and NVENC output. It leaves experimental attention and producer flags off.

## Requirements

Install this node pack under `ComfyUI/custom_nodes/comfystreamerh3`, install the pinned managed runtime, and restart ComfyUI. The production B1 preset requires Linux, an NVIDIA GPU, Python 3.12, CUDA 13 PyTorch, and `comfy-kitchen` 0.2.34 with the matching CUDA extension.

Place these model files in the indicated ComfyUI folders, then restart or refresh model lists:

| File | Folder |
|---|---|
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

The model files are not included in this repository. Use the managed build's matching model assets and runtime; see [README.md](README.md#runtime-and-models).

## Queue the example through ComfyUI's API

The ready-to-run API prompt is [`examples/basic_text_to_video_api.json`](examples/basic_text_to_video_api.json). With ComfyUI running at the default local address, submit it from the repository root:

```bash
curl -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/basic_text_to_video_api.json
```

The response includes a `prompt_id` for queue/history lookup. The MP4 is saved under `ComfyUI/output/comfystream/quickstart/`. To build the same graph on the canvas instead, add and connect the nodes below.

## Add these nodes

Create the following nodes from ComfyUI's node search:

- Built in: `CLIPLoader`, two `VAELoader` nodes, and `BasicGuider`.
- This pack: `ComfyStreamerH3 Optimized Loader`, `ComfyStreamerH3 Image to Video`, `ComfyStreamerH3 Sampler`, `ComfyStreamerH3 Video Decode`, `ComfyStreamerH3 Audio Decode`, and `ComfyStreamerH3 Output`.

Set the built-in loaders:

- `CLIPLoader`: choose `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`; set type to `minimax`. This is ComfyStream's default `native_32b` text-encoder profile when `COMFYSTREAM_H3_TEXT_ENCODER_PROFILE` is unset. The optional 4B FP8 + ClipProj profile is a separate configuration and is not part of this example.
- First `VAELoader` (video): choose `minimax_h3_video_vae_int8_convrot.safetensors`.
- Second `VAELoader` (audio): choose `minimax_h3_audio_vae_fp32.safetensors`.

On `ComfyStreamerH3 Optimized Loader`, leave the default B1 preset and other defaults selected. The provider's fast settings use width `448`, height `256`, length `362`, and 24 fps; the ready-to-run API prompt sets these values explicitly. B1 uses four-step V2 sampling and VSA-20 attention. QKV `torch.compile` is enabled; fused NVFP4 MLP conversion is attempted automatically on supported GPUs and falls back to the standard MLP otherwise. The profile/report records the selected precision and any fallback. The default text encoder is `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` with type `minimax`; ComfyStream's 4B FP8 + ClipProj profile is an optional alternative.

The decoder settings follow the fast provider path: finalized-tile streaming, output on GPU, overlapping audio decode, final-frame retention, fused FF/QK/RoPE decode, and NVENC output. OMEGA flags, native masked retile, scratch-pool experiments, and candidate attention backends remain off. H3 frame lengths follow `17 × n + 5`; 362 frames is a valid 15-second-class clip. The same frame rule is used in [Comfy-Org's native H3 workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json).

## Connect the nodes

| From | To |
|---|---|
| Optimized Loader: `model` | Basic Guider: `model` |
| Optimized Loader: `sampler` | ComfyStreamerH3 Sampler: `sampler` |
| Optimized Loader: `sigmas` | ComfyStreamerH3 Sampler: `sigmas` |
| Optimized Loader: `profile` | Image to Video: `profile`; Sampler: `profile` |
| CLIPLoader: `CLIP` | Image to Video: `clip` |
| Video VAELoader: `VAE` | Image to Video: `vae`; Video Decode: `vae` |
| Audio VAELoader: `VAE` | Video Decode: `audio_vae`; Audio Decode: `vae` |
| Image to Video: `positive` | Basic Guider: `conditioning` |
| Image to Video: `latent` | Sampler: `latent_image` |
| Basic Guider: `GUIDER` | Sampler: `conditioning` |
| Sampler: `output` | Video Decode: `samples`; Audio Decode: `samples` |
| Sampler: `report` | Video Decode: `report` |
| Video Decode: `images` | Output: `images` |
| Video Decode: `report` | Audio Decode: `report` |
| Audio Decode: `report` | Output: `report` |
| Audio Decode: `audio` | Output: `audio` |

Set the Sampler's `run_nonce` to a non-empty label such as `quickstart-001`; change it for a new run. The API example uses `comfystream/quickstart` as its output prefix.

The output node writes an MP4 under ComfyUI's output folder and exposes the video preview. The video decoder's `audio_vae` connection starts audio decoding alongside video decoding; Audio Decode passes the updated report and audio track to Output. ComfyStream's direct provider wraps this same optimized graph with progressive clip delivery; submitting the example directly to `/prompt` returns a completed MP4.

## If it does not start

- A missing-model error usually means a filename or ComfyUI model folder does not match the table above.
- A CUDA/Kitchen error means the worker does not have the pinned production runtime and compiled Kitchen CUDA extension. This pack's B1 preset does not run on CPU.
- For a valid first H3 request, keep the frame length at 5, 22, 39, 56, 73, 90, 107, 124, or another value matching `17 × n + 5`.

This is a starting graph, not a speed or quality benchmark. Use the same prompt, seed, dimensions, and frame count when comparing changes.
