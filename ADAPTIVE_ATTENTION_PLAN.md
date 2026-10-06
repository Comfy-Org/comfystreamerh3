# Adaptive region attention plan for FastH3 generation

> **Intent clarification (2026-10-02):** The user wants to evaluate existing
> pretrained Bev-style decision models, rather than begin by training a custom
> attention controller. This document's 75/899-parameter experiments remain a
> reviewed alternative, not the selected starting approach. Identify the intended
> Bev release, inspect its supported inputs and measure local decision overhead
> before using this implementation handoff. A candidate is
> [bev-decider-0.4B](https://huggingface.co/avbiswas/bev-decider-0.4B), which accepts
> text/JSON and returns typed decision probabilities. Its name does not imply
> bird's-eye-view perception or support for raw video latents.

> **Latest objective clarification:** First test whether reallocating the same
> attention work across query regions improves quality. Treat a Bev decision
> model as an optional policy chooser whose end-to-end input/latency costs count.
> The reported 75/899-parameter local heads and newly trained weights are not the
> starting assumption.

Date: 2026-10-02. Status: experiment plan reviewed by Planner, Architect and Critic; GPU results pending.
Scope: the MiniMax H3 generator in `deploy/custom_nodes/fasth3_deploy`.
This artifact defines future implementation and evaluation. Controller weights
and FastH3 speed/quality results still require the experiments below.

## Matched-budget precedent and purpose

This exact reallocation question has been tested in published work. LoSA
compares fixed Top-K with a variable number of key/value blocks per head and
query block, selected until cumulative attention mass reaches its threshold. On
Wan2.1-1.3B, the paper reports a fixed and adaptive strategy at equal 66.1%
average coverage: both run at 1.37× dense speed, while retained attention mass
goes from 96.5% to 99.0% (and worst-layer recall from 86.6% to 97.3%). **This is
an attention-fidelity result, not a matched-speed video-quality A/B.** The
separate end-to-end table reports 1.36× versus dense and a −0.06 VBench Overall
change for LoSA. [LoSA paper](https://arxiv.org/html/2608.12032)

SVG2 also selects a variable count per query cluster with top-p over its
coarse query/key estimates, then executes the selected variable-size blocks.
The authors report end-to-end gains on Wan2.1 and HunyuanVideo. [SVG2 paper](https://arxiv.org/html/2505.18875v5)
and [public implementation](https://github.com/svg-project/Sparse-VideoGen)
provide a direct algorithm/kernel reference.

These establish that fixed average work can be assigned unevenly by query and
head. The remaining question for this node is whether that reallocation improves
**generated video** quality at equal measured latency when replacing B1's
uniform 20% keep count. That requires holding the model, prompt, seed, sampler,
step count, audio, hardware and output contract fixed; attention recall alone
does not answer it. The fixed-budget-policy experiment is primary. Bev is
tested only as an optional controller if its structured input can describe
regions meaningfully and it improves the same quality/latency frontier after
tokenization, model inference and device transfers are included.

## Feasibility answer

The mechanism is plausible: retain every output query cube, spend less fine
attention on inexpensive regions, and let difficult regions retain a larger KV
budget. It will work as an optimization only if **both** a useful region-level
cost/quality tradeoff exists in this checkpoint and a compatible GPU kernel
converts smaller per-query counts into shorter execution. A tiny controller
alone establishes neither condition.

The comparison is the node's **existing sparse B1 path**, whose configured
retention is 20%, rather than dense attention. Its policy currently says
`awaiting_gpu_evidence`. Establish the actually executed manifest and performance
before treating B1 as a qualified baseline. An extension to already sparse
attention has less available savings than the published dense comparisons.

The proposed first learned controller is a new shared **24 → 3 linear head**:
75 parameters including biases, or 150 bytes of BF16 parameters excluding
runtime storage. A **24 → 32 → 3 ReLU MLP**, with 899 parameters / 1,798 bytes
BF16, is its measured nonlinear alternative, trained on identical splits.
Neither has pretrained weights; both need
task-specific supervision. These parameter counts indicate storage, not
microsecond latency. Feature extraction, launches, selection and scheduling are
part of the controller's cost.

## RALPLAN-DR decision framing

Principles:

1. Protect generation semantics: every query produces output; prefix conditioning,
   the H3 coarse branch, padding and coordinate restoration retain their meaning.
2. Count complete costs: features, predictor, routing, indices, launches and
   attention must together save time against executed B1.
3. Screen potential benefit before training: a bounded teacher-assisted local
   budget study and synthetic device counts are separate early feasibility gates.
4. Prefer existing latent features and the smallest sufficient learned head.
5. Keep experiments reversible and identified separately from the B1 preset.

Top three decision drivers:

1. Complete warm node latency on the actual worker GPU, including p95 behavior.
2. Noninferior generation and temporal/audio quality on paired held-out requests.
3. Kernel/feature compatibility with the installed Kitchen ABI, without host
   synchronization or a new dependency.

Fair options:

| Option | Why it is credible | Main limitation | Decision |
| --- | --- | --- | --- |
| A. Keep uniform B1 budgets | Established source semantics; zero additional learned scorer or feature path | Cannot exploit heterogeneous marginal attention benefit | Mandatory baseline and fallback |
| A2. No learned head: calibrated cheap statistic rule | Can use genuinely precomputed summaries and avoid learned-head execution | Coverage/top-p/sorting and new reductions are not free; calibrating thresholds still needs validation | Compare only if feature audit finds genuinely cheap available statistics |
| B. New 24→3 linear head on latent/routing features | Very small; can learn layer/noise-conditioned thresholds | Limited nonlinear decisions; still requires labels and efficient variable counts | First learned candidate after feasibility gates |
| C. New 24→32→3 shared MLP | Small nonlinear scorer; fixed shape; needs no image decoding | Extra feature and launch cost can exceed its savings; new weights required | Measured alternative on identical labels/splits; no presumed preference |
| D. DSV-style learned connection predictor | Direct video-diffusion precedent for small predictors and measured dispatch | Replaces more of routing, needs low-rank Q/K training and kernel integration; query-specific budget is not established | Reserve for failure caused by existing ranking quality |
| E. Image/BEV backbone side model | BEV token-halting supports learned spatial compute scoring | Raw image may not exist during denoising; backbone execution and semantic transfer are unqualified | No first-stage implementation; revisit only with specific evidence |

The recommended choice is a latent-feature controller experiment comparing A2,
B and C. A learned head must earn its place against both the rule and linear
control. Select from measured speed/quality, not the MLP's parameter count.
Using A remains a valid outcome if experiments fail.

First-phase classes are **5%, 10%, 20%**, with **20% uncertainty fallback**.
The exact 20% B1 teacher output cannot teach that 40% is better by minimizing
teacher output error: 20% already has zero error. A later 40% option requires an
independently qualified better teacher or task-utility objective and a separate
phase; it is excluded from first-phase labels and policy.

## Evidence and implementation anchors

Primary sources already reviewed in
[node research notes](ADAPTIVE_ATTENTION_RESEARCH.md):

- [VSA v5](https://arxiv.org/html/2505.13389v5), §§2.2–2.4, 3.1, 3.3–3.5:
  pooled 4×4×4 cube routing with fixed Top-K; coarse global branch; selection
  overhead can dominate coarse execution. Adaptive Top-K is an extension.
- [DSV v3](https://arxiv.org/html/2502.07590v3), §§3, 5–6, 10 and appendix B:
  trained low-rank query/key predictors, staged sparse adaptation and overhead
  aware dispatch; per-query sparsity remains future work.
- [BEV dynamic token halting](https://arxiv.org/abs/2303.05078v2), §§4.2–4.6:
  MobileNetV2 U-Net followed by a small MLP predicts spatial continuation in
  detection; neither weights nor detection objective solve generation budgets.
- [DyDiT v2](https://arxiv.org/html/2410.03456v2), §§3.2–3.4:
  trained compute allocation with generation/FLOPs losses; its spatial router
  controls MLP bypass rather than per-query KV count.
- [RegenHance v6](https://arxiv.org/html/2407.16990v6), §§3.2–3.4:
  learn benefit of extra work for a task; pixel macroblock analytics labels and
  decoded-video reuse do not transfer directly to denoising latents.
- [DynamicRad v1](https://arxiv.org/html/2604.20470v1), §3.4: prompt-embedding
  MLP selects semantic/motion sparsity regimes once per video; reported router
  overhead is under 2 ms in that implementation. This is a tiny-router
  precedent, not per-region routing or a latency bound for this node.
- [HEART, current title of arXiv 2605.14513](https://arxiv.org/html/2605.14513):
  calibrates per-head budgets through denoising-velocity error and explains why
  local attention-output error alone is misleading. This motivates combined
  policy replay; it does not validate this proposed classifier.
- [Official torchvision MobileNetV3-Small specification](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html):
  2,542,856 parameters and RGB image classification weights. It is a real small
  image model, but those weights are not a latent attention-budget controller.

Node anchors (paths relative to this node directory):

| File/symbol | Planning consequence |
| --- | --- |
| `sol_attn_minimax_v5.py::_vsa_plan`, `_make_producer_forward` | Preserve cube coordinates and place eventual controller at the existing GPU producer/attention seam |
| `vsa_sm120/producer.py::project_chunks`, producer lifecycle | Measurement/emission traversals can duplicate costs; count actual evaluations |
| `vsa_sm120/reference.py::route_blocks`, `sparse_attn` | Fixed Top-K ranking, exact prefix rules and distinct coarse output supply reference semantics |
| `vsa_sm120/layout.py::n_video_keep`, `token_mask` | Current keep count is scalar; `block_len` means valid tokens, never budget |
| `vsa_sm120/dispatch.py::run_vsa_chunked` | Installed production Kitchen interface currently passes scalar `topk_ratio`; audit native feature visibility and extension support |
| `native_attention/csrc/sol_attn_route.cu` | Local candidate routing accepts per-batch/head/query thresholds and emits block indices/counts; the count-consuming fine path offers an extension seam, not evidence of B1 backend equivalence |
| `native_attention/route_compaction.py` | CPU oracle only, `COMPACTION_PRODUCTION_ENABLED=False`; never use as a production shortcut |
| `loader.py::GatedH3Model` | Existing checkpoint `to_gate_compress` is a coarse-output gate, not the new budget predictor |
| `runtime.py::preset_manifest`, `execution_profile_manifest`; `kitchen_baseline.py` | Pin resolved B1/checkpoint/kernel/precision identities; metadata is not GPU evidence |
| `benchmark_video.py`, `benchmark_report.py`, `migration_capture.py` | Extend existing capture/report surfaces if suitable; confirm their interfaces during execution planning |

## Proposed controller and kernel contract

These are proposed design constraints for architecture review, not validated
ABI capabilities or an implementation decision made by the papers.

**Input:** one fixed 24-value feature vector per head/video query cube, using one
shared set of controller weights. Initial explicit schema:

- 16 query features: eight channel-group means and eight channel-group RMS
  summaries of the existing 128-dimensional pooled query centroid.
- Four context features: two existing pooled-K variance summaries, centroid
  quantization scale and valid query-token fraction.
- Four conditioning scalars: normalized layer, actual noise/timestep, head
  index and log video-cube count.

These proposed inputs are conditional on native availability and usefulness.
Audit the head width, quantization-scale identity and K-summary lifetime; do not
assume every backend already provides them. Normalization is fitted offline
and frozen. Fuse reductions into existing pooling where viable. Batch all
heads/tiles into one invocation per attention call; a fused execution path is
preferred if measurements justify it. A cheaper head-aggregated tile-only policy
is an ablation if the teacher-assisted study supports its quality.
The exact feature schema is a Gate 1 deliverable. Freeze it, normalization and
ordering before collecting training data. Do not silently materialize another
full-width tensor or add a reduction pass to meet an arbitrary feature width.
If the audit shows 24 useful cheap features are unavailable, revise both heads'
input widths and parameter counts explicitly.

Feature availability must be resolved in the native production path: the Python
reference's score tensor is not evidence that Kitchen exposes it. Pre-selection
score statistics are an optional later schema only if already provided cheaply;
no full QK recomputation or second selection pass is assumed. Chosen-K
statistics cannot determine K in the same invocation without an explicit first
pass. Any shortlist, top-two gap, entropy, coverage or sorting costs must be
counted. A 20% shortlist cannot supply a future 40% choice. A Q/conditioning-only
head is a fallback comparator when context reductions are expensive; revise
input width and parameter counts instead of forcing 24 features.

**Output:** three budget logits per head/query cube. Initial experimental classes
are 5%, 10% and 20% video KV retention. They are hypotheses, not product defaults.
Each head/query cube receives a budget and retains its own connection ranking.
For `N_video > 0`, the initial requested Top-K budget `r` maps to
`min(N_video, max(1, ceil(r * N_video)))`; preserve established handling of zero
video/prefix-only geometry. Low query budget never removes that region from
other queries' KV candidates.

Threshold routing can yield executed counts different from requested K because
of ties, forced neighbors and protected connections. Preserve the installed
backend's tie and protection semantics and report actual executed counts; do
not claim strict K retention from a threshold vector alone.

**Kernel representation:** explicit video count plus valid selected KV cube
indices for every head/query row, using a reviewed versioned ABI. Either row
offsets or bounded capacity plus valid counts may be supported; choose after
kernel review. Prefix inclusion remains independent. Lower counts must stop
loads and dot-product work, not merely zero a mask after evaluating a row padded
to the maximum class. Grouped budget launches are an alternative only if measured gathering,
launches and occupancy preserve the benefit.

The checked-in native routing/fine path already demonstrates threshold vectors
and variable block counts, so an extension at threshold generation may suffice
without rewriting fine attention. This backend remains separate from the locked
Kitchen B1 path until uniform-policy equivalence is demonstrated. Evaluate
backend equivalence first, then vary policy within that qualified path; changing
backend and policy together would confound any speed/quality conclusion. CPU
oracle compaction is never an inference integration route.

**Selection:** calculating 20% Top-K for every row then retaining 5% may erase
routing savings and still must beat B1 including overhead. Audit whether per-row selection, reusable ranking
or budget bucketing lowers total routing cost. Charge all allocation, reductions,
sorting and load imbalance to the candidate; fewer theoretical interactions
alone do not pass a gate.

**Execution:** controller, feature reductions, selection, count/index buffers and
attention remain in the generator's CUDA process/device/stream. Use bounded
reusable workspace and stable capture capacities where needed. No `.item()`,
CPU branch per tile, frame decoding, readback or network call in the hot path.
Fallback to scalar B1 at a request/shape boundary when the experimental ABI or
trained weights do not support a request. Do not add per-tile host inspection as
a fallback mechanism.

## Delivery sequence and ownership

| Work package | Owner role and write scope when execution begins | Dependencies | Deliverable and stop/go evidence |
| --- | --- | --- | --- |
| WP0. Pin baseline and protocol | Performance/verifier; capture/report artifacts only initially | None | Resolved manifest, real backend execution, warm full-node/generator/stage timings, frozen paired quality protocol |
| WP1. Audit ABI and features | Architect plus kernel specialist; analysis artifact, no premature runtime edit | WP0 identity | Available tensors/reductions, exact controller feature schema, invocation lifecycle, candidate count/index ABI, fusion/launch estimate |
| WP2A. Bounded teacher-assisted budget screen | Capture executor; proposed `scripts/qualify_h3_adaptive_attention.py` and experiment reports | WP0, WP1 semantics | Local error/budget curves, combined-policy velocity replay and eight-clip pilot against B1; captures obey hard storage/replay allowances |
| WP2B. Synthetic counts and uniform emulation | Kernel executor exclusively owns `native_attention/{_abi.py,csrc/*}`, `vsa_sm120/{reference,dispatch,layout}` | WP1 ABI | Valid-count contract, uniform-20% compatibility and measured synthetic lower-count savings on target GPU |
| WP3. Calibrate rule and train both heads | ML executor owns proposed `adaptive_budget.py`, `scripts/train_h3_budget_controller.py`, controller weight/manifests; never dispatch/producer | WP2A and WP2B pass; feature schema fixed | Frozen B1 teacher labels, calibrated no-head rule, trained linear and MLP heads, normalization/calibration manifest and held-out report |
| WP4. Integrate selected head | Integration executor owns `sol_attn_minimax_v5.py`, `vsa_sm120/{producer,config}`, `runtime.py`, `attention_execution.py` and report hooks; requests dispatch changes from kernel owner | WP2B and WP3 pass | GPU-resident policy with new identity, bounded workspace and B1 fallback; all stage costs measured |
| WP5. Independent verification | Verifier/quality reviewer; reports, minimal fixes through owner | WP4 | Paired latency/quality report and accept/reject decision against preregistered thresholds |
| WP6. Document or retire | Writer/integrator; node README/research/experiment documentation | WP5 | Accepted policy plus limitations, or negative finding and retained B1 path |

WP2A and WP2B can run in parallel after WP1 freezes semantics. They have separate
ownership: the teacher-assisted screen explores local/replayed quality; the kernel experiment proves
real execution savings. Training begins only after both justify its cost. The
linear/MLP fitting may share captured data, but comparisons use identical splits
and feature accounting. Do not concurrently edit shared dispatch/producer files.

`adaptive_budget.py`, `scripts/train_h3_budget_controller.py`,
`scripts/qualify_h3_adaptive_attention.py` and
`tests/test_adaptive_attention_budget.py` are **proposed new files**, not existing
or implemented artifacts. Node module paths are relative to this node directory; `scripts/` and
`tests/` paths are relative to the repository root. The test-engineer owns the proposed semantic-contract test file;
capture owner alone updates qualification tooling. Kernel owner alone edits
`_abi.py`, native preprocessing/routing and dispatch seam throughout the work.
ML owner consumes the frozen interface and never edits dispatch/producer.
Integration owner consumes the completed controller and asks its owner for
module changes; any explicit ownership transfer occurs serially at a gate.

"Oracle" is shorthand for a teacher-assisted local screening experiment, never
a mathematically optimal whole-model policy. Per-query errors can interact, and
changed outputs change later latent states. Only combined-policy replay and
closed-loop clip evaluation test those effects.

Required comparisons, reported separately:

1. Pinned B1 backend/uniform-20% versus candidate backend/uniform-20%:
   establish numerical compatibility and isolate backend cost.
2. Qualified candidate backend/uniform-20% versus the same backend/adaptive
   5/10/20% policy: isolate the controller/policy effect.
3. Adaptive candidate versus original pinned B1: establish the user's actual
   complete-node quality/latency benefit.

## Stop/go gates and exact proposed acceptance criteria

The numbers below are recommended initial engineering thresholds, not user
provided quality tolerances, benchmark evidence or authorized compute spend.
Freeze them before examining candidate evaluation results. A failed gate
produces a documented reason to retain B1 rather than automatically buying more
compute or expanding architectural scope.

**Gate 0 — baseline is executable and reproducible.**

- Record hardware, driver, CUDA/PyTorch/Kitchen wheel provenance, checkpoint
  hash, resolved preset, precision, sampler, dimensions, frames, audio, decoder
  and export policy. Expected configured B1 is V2/four steps/20%/64-token cubes,
  Kitchen 0.2.34, but reconcile it with actual execution logs.
- Confirm the intended sparse backend executed and record fallbacks. Record
  warm complete-node median/p95, generator median/p95, changed-stage times and
  peak VRAM. Cold load/compile is separate.
- Calculate the attainable complete-node benefit from the measured B1 affected
  stage fraction before kernel/training investment. For affected fraction `f`,
  stage speed factor `s` and additional overhead fraction `o`, the estimated
  node latency ratio is `(1-f) + f/s + o`. Even deleting the affected stage
  cannot save more than `f`. Stop if the measured attainable envelope cannot
  reach the proposed 5% node improvement. Use measured stage behavior to set
  `s`; arithmetic interaction reduction does not imply proportional speed.
- Stop if no compatible target GPU or executable pinned B1 is available;
  preserve the plan and report the exact prerequisite, without inventing speed.

**Gate 1 — a cheap feature and real count contract exists.**

- Freeze ordered feature schema, normalization, actual timestep/noise semantics,
  head sharing, producer traversal count, per-query count/index bounds and
  zero-video/prefix behavior.
- Identify whether installed Kitchen supports extension or which existing native
  source owns the necessary change; a scalar retention call cannot pass.
- No new dependency; any required backend change is identified separately from
  the pinned B1 control. Stop and re-scope if native changes cannot be built or
  features require prohibitive extra materialization.

**Gate 2 — attainable quality and execution savings justify training.**

- Teacher-assisted local labels choose the smallest of 5/10/20% satisfying
  a preregistered B1 attention-output error tolerance. Evaluate all three
  choices together at a captured call. Replay a **combined mixed policy**
  through transformer output/denoising velocity once per captured latent;
  do not replay once per query. Sweep tolerance on development captures and
  report both local and replayed curves. At 20%, exact teacher compatibility
  remains available; uncertain examples use 20%. A higher-budget teacher and
  40% labels belong to a separately qualified later phase.
- At least one held-out screen operating point must reduce **sampled-row**
  valid video token-pair work by **at least 15% against those same B1 sampled
  rows** at the chosen tolerance. This is a local sensitivity screening
  threshold, not a 15% whole-request requirement for the 8,192-record capture.
  Report three distinct quantities: sampled-row work reduction; sampled rows'
  fraction of baseline executed work; and observed whole-request work reduction.
  With unsampled rows retained at 20%, whole-request savings attributable to
  the sample are bounded by sampled baseline work share × sampled-row reduction
  before additional routing/kernel overhead. Use valid-work weights rather
  than equating sampled row count with work share. No gain is extrapolated
  from the sample to all queries without a full-coverage policy experiment.
- Synthetic count patterns spanning uniform 20%, low/high mixes and worst-case
  clustering must actually skip loads/work and run the **complete changed stage
  faster than B1**, including selection and packing. Mean-20% redistribution
  alone provides no theoretical reduction. Synthetic counts establish kernel
  execution potential, not an attainable learned-policy work or latency gain.
  Full-coverage rule/trained policies still require actual executed-work and
  complete-node latency evidence at the later gates.
- Uniform-20% emulation must retain B1 selection/prefix/padding semantics and
  satisfy its numerical compatibility gate before evaluating adaptive outputs:
  require bitwise identity if the same operations/order are retained; otherwise
  freeze separately justified thresholds from precision/reference analysis and
  qualified uniform-policy behavior **before adaptive policy tests**, with no
  new nonfinite values. B1 A/A deterministic repeatability can be zero and
  cannot by itself define numerical allowance. Retain A/A as a repeatability
  check, not the sole justification of epsilon.
- Include forced-neighbor rules, routing ties, requested versus executed count,
  edge padding and conditioning behavior in the equivalence evidence. Keep
  fixed policy while comparing backends; enable adaptive policy only afterward.
- Count executed tile interactions and
  `sum(valid_query_tokens * valid_selected_key_tokens)` per row. Report
  mandatory prefix, forced-neighbor and tie contributions separately, and
  measure padded kernel work. Requested ratios alone cannot satisfy the gate.
- Enforce the bounded feasibility protocol below. Automatically stop captures
  or replays at the preregistered request/time/storage allowance; no expansion
  or rental spend follows automatically from an inconclusive screen.

**Gate 3 — trained head pays for its complete feature path.**

- The no-head rule calibrates on development data; both learned heads train on
  the same teacher-assisted labels. Compute penalty uses measured
  cost buckets; include deterministic held-out prompt/seed splits, class
  confusion, misclassification error and calibration. Parameter storage is not
  the acceptance metric.
- No learned adaptive production profile without real trained weights and a
  manifest pinning schema, normalization, checkpoint, shapes and training
  version. A rule profile instead pins its calibrated thresholds, feature
  schema, checkpoint and supported shapes.
- Total added **feature extraction/aggregation + predictor + budget conversion** time,
  summed over all actual invocations, must be **≤1% of B1 generator time** on
  the evaluated requests. Any routing/packing changes remain separately charged
  in the full changed-stage comparison. No hidden duplicate bootstrap pass.
- Compare every head to the rule with full feature costs included. Prefer the
  simpler rule or linear head when the MLP does not improve the held-out complete
  node tradeoff. If no learned head fits the cost allowance, stop
  or explicitly review fusion/reduced features before enlarging the model.

**Gate 4 — user-visible win with noninferior quality.**

- Complete warm-node median latency improves **≥5%**, and the upper bound of a
  paired, request-clustered 95% bootstrap confidence interval for candidate/B1
  median latency ratio is **<1.0**. Generator and changed-stage medians also
  decrease.
- Complete warm-node **p95 ratio ≤1.00** on the same balanced request set;
  report its uncertainty and any per-geometry tail regression. Stop promotion
  if evidence is insufficient to resolve meaningful tail degradation.
- No peak VRAM increase above **2%**, no OOM, new NaN/Inf, indexing error,
  missing query output, audio/prefix omission or new fallback on qualified
  requests. These are proposed promotion bounds; record absolute bytes too.
- Recommended primary quality protocol: blinded paired scoring on a frozen
  1–5 rubric, separately for overall visual fidelity and temporal/motion quality.
  Candidate minus B1 must have a request-clustered one-sided 95% confidence
  lower bound **≥−0.10 points for each category**. Report prompt adherence,
  small details and audio alignment separately; any reproducible severe failure
  introduced by the candidate fails promotion regardless of the mean.
- Auxiliary automated prompt/video and temporal metrics use identical evaluators
  and paired outputs. Freeze evaluator identities and thresholds before test
  use; do not use one embedding metric as a substitute for temporal/audio review.

**Gate 5 — bounded launch and explicit rollback.**

- Candidate profile is opt-in and versioned separately from B1. Unsupported
  shapes, missing/invalid controller weights or ABI incompatibility retain the
  established request-level fallback with an observable reason.
- Document supported geometry/checkpoint/hardware, confidence and quality
  margins, observed overhead, latency percentiles and unresolved cases. Promote
  beyond the experimental profile only with the Gate 4 report.

## Training and measurement protocol

Preflight the smallest genuinely supported production geometry, using a real
request counted within the initial allowance. Measure activation and compact
record sizes and elapsed time before continuing. Recommended initial screen:

- Four prompts × two seeds = **eight B1 requests**, spanning simple/static and
  difficult/motion content. Preserve their B1 outputs for paired review.
- Sample four layers, all four actual denoising evaluations, four heads and
  16 query cubes per sampled call. This yields at most **8,192 compact
  head/query records** if all slots are valid. Adjust the sample to actual
  supported geometry/layers/heads at preflight and record the changed caps.
- Evaluate 5/10/20% jointly within each sampled call. Stream compact features,
  errors, valid-work counts and labels; no all-layer activation archive or
  token-pair score tensor dump. Retain **≤2 GiB feasibility data** and **at most
  one attention invocation's captured activation set on GPU** beyond the live
  request's ordinary memory. Release it before capturing the next invocation.
- For mixed-policy replay, aggregate sampled decisions into one policy at a
  captured latent and replay the transformer **once per latent**, never once
  per tile/head. At most eight requests × four evaluations = **32 such
  transformer evaluations** in this initial allowance. Reuse request-boundary
  latent states rather than retaining all layer activations.
- Allow at most **eight single-injection suffix replays** and **eight complete
  candidate pilot clips**, paired with the eight B1 outputs, before expansion.
  These pre-training clips are **teacher-assisted fixed-map experiments**.
  Every budget entry is indexed by request ID, exact geometry/layout signature,
  layer, actual denoising evaluation, head and query position. Apply each map
  only to the same prompt, seed, geometry and actual schedule from which its
  B1 capture was generated; validate the map signature before replay. Unsampled
  rows retain 20%. Pin maps alongside request/manifest IDs within the retained
  feasibility-data cap, with sampled work share and both sampled/whole-request
  counts in their reports.
  Suffix replays measure downstream sensitivity; complete clips test changes
  feeding later layers/evaluations. These fixed maps do not adapt to changed
  candidate latents, are not trained/general controllers, and cannot be applied
  to unrelated requests. Neither replay type proves final noninferiority.
- Before starting, convert the request/replay caps to a frozen wall-time or
  GPU-time allowance using preflight measurements. Stop automatically at the
  first request, replay, time or storage limit. An inconclusive result triggers
  review of the next bounded experiment; it does not authorize more runs or
  rented capacity. Existing compute availability is a prerequisite.

These are recommended first-screen limits, not user-provided budgets or a claim
that eight clips train a general controller. At each captured call preserve
the coarse and conditioning outputs. Label local sensitivity to video KV count;
check combined policies through transformer velocity and full pilot clips.
Weight excessive under-budget error more than unnecessary high budget. Add
measured compute penalty and calibrate uncertainty to **20% B1 retention**.
Combined local labels are only teacher-assisted proposals; interacting errors
and changing latent states require separate closed-loop evidence.

A development-calibrated **causal rule** may be a separate optional full-coverage
experiment if its inputs pass the cheap-feature audit. Identify its policy and
freeze its own cost/request/replay allowance before execution; account for
features and decisions on every row and measure whole-request work/latency.
It is distinct from the eight teacher-assisted fixed-map clips and is not
silently included in, or added beyond, their frozen caps.

Split by prompts/scenes before generating tile samples so neighboring cubes or
same-request layers cannot leak into evaluation. Include camera movement,
static scenes, tiny subjects, texture, scene cuts, occlusion, hands, text and
speech. Use the production noise/layer distribution; no generic BEV foreground
labels or decoded-image classifier is the training target. Predictor-only
classification quality must be followed by closed-loop video evaluation, where
changed outputs feed later layers/steps. If fitting cannot preserve generation,
joint adaptation/distillation is a separately reviewed extension, not an
automatic step.

Only after the bounded pilot and synthetic kernel gates pass, preregister a
separate training and final evaluation allowance. A recommended final held-out
protocol could use 80 prompts × three seeds, balanced across supported
geometry/audio cases, with enough raters/timing repetitions to resolve the
proposed margins. Determine its sample size from development variance and
available resources before accessing held-out candidate scores. It is not an
automatic expansion of the initial eight-clip pilot. Training capture volume,
label tolerances, raters, compute access and any rental spend remain execution
prerequisites; none was supplied by the user or approved by this plan.

Warm both variants, alternate or randomize paired runs, keep clocks/power and
background load comparable, and synchronize CUDA events outside the hot path.
Time pooling/reductions, head, budget conversion, selection, indices,
gather/retiling, fine attention, coarse/merge, generator and complete node.
Report controller invocation counts and distributions, budgets by
layer/step/geometry, requested and executed counts, valid token-pair work,
mandatory/protected interactions, padded work, launch count, peak memory,
latency spread and all fallbacks.
Account for unchanged QKV, MLP, VAE, audio and export in complete-node time.

## ADR: start with latent budget heads behind two feasibility gates

**Status:** experiment design approved by Architect and Critic; implementation and results pending.

**Context:** VSA already supplies content-dependent cube connections with a
uniform retention fraction. The user wants very fast variable attention work
by region in the generator's process. BEV token halting supports small learned
spatial policies but its detection objective and early U-Net are not qualified
for this denoising path. Published VSA/DSV gains do not establish savings over
this node's sparse B1 configuration.

**Decision proposed:** establish actual B1 execution; run bounded teacher-assisted
local screening and synthetic kernel experiments; if both pass, calibrate a
genuinely cheap rule if available and train a shared-weight **24→3 linear head
first**, comparing a 24→32→3 MLP on identical splits. Integrate only the
winner on the same CUDA device/stream with explicit per-query counts and
all prefix/coarse/output semantics protected. Initial classes are 5/10/20%,
with uncertainty falling back to 20%. Separate backend equivalence, within-backend
policy benefit and actual B1 user benefit. A 40% later phase needs independently
qualified utility/teacher labels rather than assuming more work is better.

**Rejected for the first experiment:** a raw-image/BEV backbone, because current
decoded frames need not exist and feature/backbone overhead is unqualified;
repurposing checkpoint compression gates, because their semantics differ;
maximum-K masking, because skipping logical connections without skipping
kernel work need not save time; direct dense teacher substitution, because it
changes the trained VSA comparison semantics.

**Consequences:** new task-specific labels and weights are required; a native
ABI extension may be the main blocker; fusion and selection may dominate
head compute. A negative screen/kernel result ends this experiment with useful
evidence and no adaptive policy promotion. DSV-style ranking or joint adaptation
requires a new scoped decision if existing ranking or learned quality fails.

**Revisit when:** cheaper exposed feature reductions appear; kernel support
changes; the local/replay screen finds no useful budgets; the linear head meets all gates; or
closed-loop quality requires checkpoint adaptation. Record actual evidence,
not only a hypothetical parameter or FLOP estimate.

## Review revision record

Architecture review revisions incorporated on 2026-10-02:

1. Replaced 10/20/40% with identifiable 5/10/20% labels and 20% fallback; separated
   any future 40% utility/teacher phase.
2. Made the 75-parameter linear head the first learned candidate and the
   899-parameter MLP a measured same-split alternative; removed presumed MLP
   superiority and free coverage/score statistics.
3. Bounded initial requests, sampled records, storage, GPU activation lifetime,
   replay counts and pilot clips; added automatic allowance stops.
4. Distinguished local teacher-assisted labels from whole-model optimality and
   combined velocity/full-clip evaluation.
5. Added actual valid token-pair and padded-work accounting and three separate
   backend/policy/user-benefit comparisons.
6. Separated reference/precision numerical tolerance justification from B1 A/A
   repeatability; froze it before adaptive tests.
7. Specified a causal initial 24-feature schema conditional on available pooled
   tensors, and added DynamicRad, HEART and official image-model evidence.

Critic blocking revisions incorporated on 2026-10-02:

1. Restricted the 15% screen to sampled rows; separated sampled reduction,
   baseline sampled work share and whole-request reduction, with the explicit
   unsampled-20% savings bound. Synthetic patterns establish kernel potential;
   full-coverage real policy gates remain later.
2. Defined pre-training candidate clips as signature-checked teacher-assisted
   fixed maps for their original prompt/seed/geometry/schedule, with unsampled
   20% rows. Kept existing caps; separated any optional full-coverage causal
   rule and required its own frozen allowance and accounting.

## Execution handoff roster and mode hints

This is a planning handoff; no team is launched here. Use known role surfaces
with bounded ownership when execution begins:

| Lane | Role surface | Responsibility |
| --- | --- | --- |
| Coordination | `planner` | Maintain gates, allowances, dependencies and experiment identity |
| Baseline/measurement | `verifier` | WP0 manifest, timing protocol and Amdahl screen; independent final evidence |
| ABI/feature design | `architect` | WP1 feature/ABI review and backend-equivalence acceptance |
| Native kernel | `executor` | WP2B only: counts/threshold seam, routing/fine semantics and actual work counters |
| Teacher screen/training | `executor` | WP2A/WP3: bounded captures/replays, rules, labels and both heads |
| Integration | `executor` | WP4 producer/dispatch/profile wiring after both prerequisite gates |
| Quality/checks | `test-engineer`, then `verifier` | Design meaningful semantic/numerical checks; independently evaluate locked paired quality/latency |
| Review | `critic` | Challenge gate evidence, resource bounds, causal feature choices and unsupported claims |

Choose `$team` when WP2A/WP2B have independent approved write scopes and real
parallel work; keep one owner for shared dispatch/producer integration. A
single-owner `$ralph` loop is suitable after the reviewed plan, PRD and test
spec exist, with gate evidence as iteration exits. These are handoff hints,
not commands to launch modes or implementation from this planning request.

Team verification path: `team-plan` freezes this plan and allowances;
`team-prd` locks request/identity/quality requirements; `team-exec` performs
bounded packages; `team-verify` runs independent semantic, precision, work-count,
paired latency and locked-quality checks; `team-fix` returns failures to the
owning lane and reruns affected checks. Mark complete only when all applicable
gates pass or the experiment is explicitly rejected and documented with B1
retained. Unsupported GPU/ABI/rater prerequisites are reported as evidence,
never silently converted into a promotion claim.


### Concrete follow-up staffing and launch hints

Keep inherited model settings. Suggested reasoning: high for architecture,
native kernel work and critical verification; medium for bounded capture,
training orchestration and report assembly. Choose two executor lanes only after
WP0/WP1 pass: one exclusively owns WP2B kernel files, the other owns WP2A capture
and offline fitting files. Integration is serialized under one executor; a
verifier remains independent of both implementation owners. A single-owner
Ralph run can execute the same gates sequentially with architect/test-engineer
and verifier reviews at their exits.

The following are future handoff hints, not executed commands:

```text
$ralph Follow .omx/plans/prd-adaptive-region-attention.md and .omx/plans/test-spec-adaptive-region-attention.md; begin at WP0 and stop at failed evidence gates.
$team Use two executor lanes for WP2A and WP2B after WP0/WP1 pass, following deploy/custom_nodes/fasth3_deploy/ADAPTIVE_ATTENTION_PLAN.md; preserve exclusive ownership and independent verification.
omx team 2:executor "Follow the adaptive region attention plan's WP2A and WP2B after WP0/WP1 pass; keep kernel and capture ownership separate, enforce experiment caps, and report evidence before integration."
```

For a team-to-Ralph handoff, the team must supply uniform-policy compatibility,
actual count/work counters, bounded capture artifacts and measured stage timing.
Ralph rechecks those identities before integration and independently verifies
complete-node latency and locked quality before reporting a successful policy.
The final critical review accepted the sampled-work denominator and the
signature-checked fixed-map pilot; neither is a trained-controller result.
