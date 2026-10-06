# Benchmarking DMAD with ComfyStream

This is an opt-in experiment. The loader requires a gate-free, statically merged DMAD transformer and does not modify the default FastH3 profile.

## Prepare the checkpoint

Accept the MiniMax H3 Community License for the base model and adapter. Run the converters on a compatible Linux worker with the existing ComfyUI Torch and Safetensors environment. They do not install packages or download the large weights automatically.

~~~bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --local-dir /models/MiniMax-H3 \
  --exclude 'FL2VA/*' --exclude 'Ref2VA/*' --exclude 'transformer_ref/*'

hf download ZhengmingYu/DMAD \
  --revision 723954a331b372747f390afdc097e8f5699acad9 \
  --include 'minimax_h3/*' \
  --local-dir /models/DMAD
~~~

Merge each rank-128 adapter into the Diffusers transformer, then convert that merged model to ComfyUI's H3 checkpoint format:

~~~bash
python scripts/prepare_dmad_h3.py \
  --transformer-dir /models/MiniMax-H3/transformer \
  --adapter /models/DMAD/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors \
  --variant lora_critic \
  --output-dir /models/dmad-h3/merged-lora-critic

python scripts/convert_dmad_comfy_h3.py \
  --transformer-dir /models/dmad-h3/merged-lora-critic \
  --output-file /ComfyUI/models/diffusion_models/dmad_h3_lora_critic.safetensors

python scripts/prepare_dmad_h3.py \
  --transformer-dir /models/MiniMax-H3/transformer \
  --adapter /models/DMAD/minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors \
  --variant full_critic \
  --output-dir /models/dmad-h3/merged-full-critic

python scripts/convert_dmad_comfy_h3.py \
  --transformer-dir /models/dmad-h3/merged-full-critic \
  --output-file /ComfyUI/models/diffusion_models/dmad_h3_full_critic.safetensors
~~~

The converter reorders all 52 SwiGLU FC1 weights from Diffusers `[up,gate]` to Comfy `[gate,up]` and records `activation_layout: comfy_gate_up`; regenerate older exports. Both steps create SHA-256 manifests and leave the source weights intact. Keep each Comfy checkpoint beside its .safetensors.dmad.json receipt, and restart ComfyUI after deploying the updated fasth3_deploy files. The Comfy loader rejects a wrong base revision or variant, changed checkpoint hashes, incomplete modules, dynamic LoRAs, FastH3 attention gates, and an incompatible ComfyUI Core commit.

The converter writes the complete Comfy checkpoint as a single Safetensors file. Check available disk space and CPU memory before converting or loading it. No benchmark is valid without the exact dependency and component hashes in its report. The published student was trained at 124 frames; longer H3-compatible lengths such as 362 frames are extrapolation tests and must not stand in for the trained-length quality check.


Export the actual FP8 storage arm separately:

~~~bash
python scripts/convert_dmad_comfy_h3.py \
  --transformer-dir /models/dmad-h3/merged-lora-critic \
  --output-file /ComfyUI/models/diffusion_models/dmad_h3_lora_critic_fp8.safetensors \
  --linear-weight-dtype fp8_e4m3fn
~~~

An experimental `--linear-weight-dtype int8_convrot` mode writes the 208 QKV/output/MLP matrices shared with B1 as INT8 ConvRot (group size 256), plus Comfy `comfy_quant` metadata and per-row F32 scales. It leaves DMAD AdaLN matrices in BF16. B1's other 50 INT8 tensors are its gate-compress weights, which DMAD does not have. The exporter requires the pinned Comfy Kitchen CUDA extension. Pair this checkpoint with `precision_mode=nvfp4_mlp` so the 50 main-block MLPs are fused to NVFP4 at load time. The combined INT8 + NVFP4 path has now loaded on an RTX 5090, passed a 124-frame media-integrity screen, and completed a matched 15-second timing matrix; it remains experimental pending human quality review and the architectural gate-quality difference from B1. See [`benchmark-results/dmad-h3/int8-convrot-lium-2026-10-04/RESULTS.md`](../../../benchmark-results/dmad-h3/int8-convrot-lium-2026-10-04/RESULTS.md).

~~~bash
python scripts/convert_dmad_comfy_h3.py \
  --transformer-dir /models/dmad-h3/merged-lora-critic \
  --output-file /ComfyUI/models/diffusion_models/dmad_h3_lora_critic_int8_convrot.safetensors \
  --linear-weight-dtype int8_convrot
~~~

## Run a functional comparison

On the Linux ComfyUI worker with the converted checkpoints, FastH3 B1 preset, and H3 text encoder/VAEs installed:

~~~bash
python3 scripts/benchmark_dmad_h3.py \
  --base-url http://127.0.0.1:8188 \
  --output-dir /workspace/comfystream/benchmark-results/dmad-h3 \
  --checkpoint-lora-critic dmad_h3_lora_critic.safetensors \
  --limit 2
~~~

The first two jobs run current FastH3 B1 and dense BF16 lora_critic on the same prompt, seed, resolution, and frame count. This is an integration smoke test; two clips cannot support a quality or real-time claim.

The default matrix tests BF16, FP8 E4M3FN, and fused NVFP4 MLP for each supplied DMAD adapter, plus the gate-less VSA fine-attention route. Add `--include-combined` for FP8+VSA and FP4+VSA. FP4 covers the 100 trunk MLP projections; it is not whole-model FP4. FP8 requires a separate checkpoint exported with `--linear-weight-dtype fp8_e4m3fn`: 260 matrix weights use genuine FP8 storage while norms and precision islands retain their source types. Computation remains BF16; this arm does not use native FP8 GEMM. Select one precision per command and provide its matching checkpoint; BF16/FP4 use the source export, while FP8 uses the FP8 export. Add --checkpoint-full-critic to include the second adapter. Use --family, --precision-mode, --variant, and --attention-mode to isolate arms on memory-constrained workers. Sweep existing request-scoped inference controls one at a time with `--memory-only` and repeatable --memory-flag options, for example --memory-flag cache_inference_rope --memory-flag share_timestep_silu. Unwired, killed, conflicting, or unengaged features receive an explicit result and do not count as speed measurements. Profiling controls are diagnostic and are excluded from timing comparison.

The loader defaults to reserving 8 GiB for H3 activations and workspaces, then lets Comfy partially offload weights with `force_full_load=False`; tune this with `--vram-reserve-gib` when comparing 24/32 GiB GPUs. On Linux, it also requests reclaim of clean checkpoint page-cache pages after model construction so host-RAM headroom remains for FP4 conversion. This cache hint is best effort and does not change model weights.

Decoder tests can select --decoder-mode reference, --decoder-mode fused_ff, --decoder-mode fused_ff_qk_rope, --decoder-mode scale_cache, or --decoder-mode native_v036. Existing options also select CUDA graph replay, Q/K in-place execution, pinned/pageable transfers, output packing, staging, tile batch, VAE accumulation policy, and Kitchen VAE fusion. Compare each change separately, then rerun combined candidates. Attention and modulation changes require the quality screen from the accompanying test specification.

Use --text-encoder, --video-vae, and --audio-vae to test separate conditioning and VAE precision variants. The runner records the selected encoder, video VAE, audio VAE and DMAD checkpoint hashes. It never silently treats a model name as a precision-equivalent reference.

--show-matrix prints the proposed runs without contacting ComfyUI. --warmup and --repeat control unscored warmups and repeats. Every completed job produces an MP4, a hash-bound JSON receipt and an entry in the per-run REPORT.md and REPORT.json; failed jobs remain in the result matrix. Loss of the worker stops further submissions and marks remaining rows as skipped. Open the paired MP4s and listen to the audio before drawing quality conclusions.

The **default performance and memory test is a 15-second-class clip**: 512×320, 362 raw frames at 24 fps (15.08 seconds before any output trimming). The 362-frame length satisfies H3's `17*n+5` temporal constraint. Use the 12-case fixture `docs/experiments/dmad-h3-benchmark-prompts-2026-10-03.json` for the standard prompt/seed matrix; the runner and `build_matrix()` also default to 362 frames when no workload fixture is supplied. Keep resolution, seeds, audio, warmups, and repeats matched when comparing implementations.

The separate held-out fixture `docs/experiments/dmad-h3-evaluation-prompts-2026-10-03.json` remains at 124 frames (5.17 seconds), the DMAD student's trained length. Use it as a **trained-length quality check**, not as the default performance or memory benchmark. A 362-frame result is an extrapolation for quality and must not replace the 124-frame quality check. Historical 124-frame speed results are short-clip diagnostics and cannot be projected linearly to 15 seconds.

Use `--save-pixel-samples` for first/middle/last decoded PNGs; those runs are diagnostic. Run warmup 1/repeat 3 separately without profiling or PNG capture for speed comparisons. Build the local clip/frame gallery with `python3 scripts/build_dmad_comparison.py`.

For a maximal compatible DMAD speed profile against B1, combine INT8 ConvRot shared projections + NVFP4 MLPs, VSA-20, the five warmed inference-cache/release flags, B1's VSA bootstrap-measurement skip, and (when built) the native masked-retile/scratch-pool controls. The B1 and DMAD paths can receive the implemented OMEGA controls through repeatable `--omega-flag` options. Native masked retile requires the compiled artifact described in [`native_attention/README.md`](native_attention/README.md); runtime compilation is intentionally disabled. B1's matching trained gate-compress branch remains a model-quality difference, not an inference flag.

`--compile-transformer` is an opt-in experimental DMAD loader setting; it lazily compiles `diffusion_model.forward`, while suppressing Comfy's AIMDO allocator graph only during that call because its `torch.Stream` API is incompatible with Dynamo. Warmup must pay compilation cost before measured repeats. Check `transformer_compile` in each receipt to confirm it engaged and inspect graph counts. Do not combine `decoder_mode=native_v036` with `--kitchen-vae-fusions`: the native decoder already owns those fusions. CUDA Graph decode stays disabled because its earlier capture path failed qualification. The registry also contains planned flags without runtime call sites; the benchmark only exposes flags shared by the runtime registry and inference-feature implementation.

### Composed precision and inference-cache runs

Memory singles and bundles can use a selected precision and attention route. The default memory precision remains BF16. If `--precision-mode` selects one precision, memory arms inherit it; otherwise they remain BF16. Use `--memory-precision-mode` to override that selection. `--memory-attention-mode` works the same way for attention. `--memory-combined-only` selects the bundle before `--limit`, so a small limit cannot silently replace it with an earlier control row. The runner refuses an empty filtered memory selection.

The cache bundle worth carrying forward from the 5090 ablation is:

```text
cache_fixed_adaln
cache_inference_rope
cache_text_tag_runs
release_dit_attention_input
fuse_modulation_segments
```

Preview an FP8 plus bundle job without submitting it:

```bash
python3 scripts/benchmark_dmad_h3.py \
  --show-matrix --output-dir /workspace/comfystream/benchmark-results/dmad-h3 \
  --checkpoint-lora-critic dmad_h3_lora_critic_fp8.safetensors \
  --family dmad --variant lora_critic --precision-mode fp8_e4m3fn \
  --attention-mode dense --memory-only --memory-combined-only \
  --memory-flag cache_fixed_adaln \
  --memory-flag cache_inference_rope \
  --memory-flag cache_text_tag_runs \
  --memory-flag release_dit_attention_input \
  --memory-flag fuse_modulation_segments
```

Use the BF16 merged checkpoint for `--precision-mode nvfp4_mlp`. Compare each composed profile with fresh plain FP8 dense or plain NVFP4 MLP dense on the same device, critic and media contract. The report validator checks selected precision, actual FP8 storage coverage or fused MLP block count, requested memory flags, the four-step DMAD sampler, audio and frame contract. A successful timing requires every requested cache to have a nonzero call count. Keep decoder/streaming settings and component hashes in the run manifest.

On 32 GiB GPUs, start the worker with `--disable-cuda-malloc --reserve-vram 8` and match OMP/MKL threads to the allocation. The requested non-overlap path now keeps audio serial; the earlier concurrent audio path raced Comfy residency management. Both FP8 critic variants completed their paired visual and warm timing groups with asynchronous weight offload after that fix. Synchronous offload remains an explicit fallback; never pool the two policies in one timing group. Run each checkpoint/precision group in a fresh Comfy process: node caches can retain the previous 60 GB model while constructing the next, exhausting a 107.5 GiB host allocation. Attention and memory flag comparisons within one model group can reuse the shared base. This process isolation is a benchmark lifecycle requirement, not proof that arbitrary in-process model changes are safe. See `docs/experiments/dmad-h3-lium-5090-2026-10-03.md` for verified results.
