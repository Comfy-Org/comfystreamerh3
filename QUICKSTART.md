# FastH3 live demo

This example starts the optimized FastH3 flow and opens a plain video player at `http://127.0.0.1:8765/live`. It does not open the control room.

## Run it

You need ComfyUI running with this node pack, the model files below, and the sibling `../comfystream` checkout. For a remote GPU, forward its ComfyUI port first:

```bash
ssh -N -L 8188:127.0.0.1:8000 -p <ssh-port> root@<worker-host>
```

Then, from this repository:

```bash
COMFYUI_URL=http://127.0.0.1:8188 python examples/live_demo.py
```

The script opens the local video page and starts a fresh demo run. Press `Ctrl+C` in the terminal to stop it. For a local ComfyUI, the default `http://127.0.0.1:8188` is used, so run `python examples/live_demo.py`.

## Models

Install the pinned `ComfyUI-ClipProj` node:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nicolab28/ComfyUI-ClipProj.git
git -C ComfyUI-ClipProj checkout c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c
```

Put these model files in the matching ComfyUI folders:

| File | Folder |
|---|---|
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| [qwen3vl_4b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/02f0d3eefb59528799653d16735818801177ef1f/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | `models/text_encoders/` |
| [mmh3-4b-ClipProj-v3.1-mlp.safetensors](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/resolve/2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808/mmh3-4b-ClipProj-v3.1-mlp.safetensors) | `models/clip_projections/` |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

The example uses the Qwen3-VL 4B encoder and ClipProj MLP, the B1 four-step VSA-20 preset, 448×256 output, 362 frames (about 15 seconds), and NVENC. Models are not included in this repository.

## One MP4 instead of a live stream

For one completed clip, submit the ready-to-run API graph:

```bash
curl -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/basic_text_to_video_api.json
```

The response includes a `prompt_id`; the MP4 is saved by ComfyUI under `output/comfystream/quickstart/`.
