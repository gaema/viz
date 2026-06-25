# real-attention -- finding GPT-2's heads (real weights, Phase 9)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server + a one-time ~548 MB weight download: `python3 -m http.server 8099 --directory viz`

Interactive page: the **real** counterpart to the synthetic attention pages
([`attention-patterns`](../attention-patterns/), [`scaled-dot-attention`](../scaled-dot-attention/)).
It runs a **real GPT-2 (124M)** in your browser and lets you hunt its attention
heads on any sentence you type. **Anchor**: A2 attention. Second increment of
**Phase 9 — real-model grounding** (`../plan/phase9.md`).

## Why a hand-written forward pass

transformers.js **cannot emit attention weights** — the ONNX export drops the
attention probabilities ([Optimum #325](https://github.com/huggingface/optimum/issues/325)
is still open; setting `output_attentions` in an ONNX config throws at session
init). So this page does NOT use transformers.js for inference. Instead
[`gpt2.js`](gpt2.js):

1. fetches the raw GPT-2 **safetensors** weights (HF, CORS-enabled) and parses
   them in vanilla JS, and
2. runs a **minimal GPT-2 forward pass** (token+pos embed → 12 blocks of
   LayerNorm + Conv1D QKV + causal attention + Conv1D MLP with `gelu_new`),
   capturing `A[layer][head] = softmax(QKᵀ/√d + causal mask)` at every layer.

transformers.js is still used — but only for the small **tokenizer** (real
GPT-2 BPE), not the model.

## Verified against PyTorch

"Real" has to mean real, so the JS forward is checked against PyTorch GPT-2
(`output_attentions`, eager) by [`gpt2.test.mjs`](gpt2.test.mjs) over a committed
fixture ([`gpt2-groundtruth.fixture.json`](gpt2-groundtruth.fixture.json)):

- attention matrices at layers 0 / 4 / 5 / 11 match within **max|Δ| ≈ 5e-5**
  (the fixture's 4-dp rounding floor), and
- the JS head detectors independently pick the **same literature heads** PyTorch
  does: previous-token = **L4·H11** (score 1.00), induction = **L5·H5** (0.41),
  attention-sink = **L7·H2** (0.98).

Run it (548 MB weights not committed):
`GPT2_WEIGHTS=~/.cache/huggingface/hub/models--gpt2/snapshots/*/model.safetensors node real-attention/gpt2.test.mjs`

## WebGPU compute backend (optional)

The same forward also runs on the **GPU**: [`gpt2-webgpu.js`](gpt2-webgpu.js)
expresses each step (Conv1D matmul, LayerNorm, causal attention, `gelu_new`) as a
WGSL compute shader, keeps activations resident on-device across the 12 layers,
and reads back only the attention matrices. The embedding gather stays on the CPU
so the 154 MB `wte` never has to go to the GPU. Toggle **“WebGPU compute”** in the
panel; the page picks WebGPU automatically when the browser exposes it.

It is **self-verifying**. On the first GPU run the page also runs the CPU forward
and reports `max|Δ|` over every attention weight in the banner; on any mismatch (a
shader bug, a flaky adapter) it **falls back to the CPU path automatically**. Both
forward timings are shown so the GPU speedup is visible.
[`gpt2-webgpu.test.mjs`](gpt2-webgpu.test.mjs) runs that GPU-vs-CPU check on a tiny
random model under any WebGPU runtime (e.g. Deno) and skips cleanly where
`navigator.gpu` is absent.

> Verification: the WGSL is **author-verified** — `gpt2-webgpu.test.mjs` run under
> Deno (WebGPU via radv on an AMD Radeon 610M) reproduces the CPU forward within
> **worst attn |Δ| = 5.96e-7**. Plus the in-page self-check on a real browser, with
> the PyTorch-pinned CPU forward (`gpt2.test.mjs`) as the ground-truth and the
> automatic fallback. (Headless chromium here doesn't expose `navigator.gpu`, and
> a host with a broken headless Intel GPU wedges Deno's Vulkan adapter init — use a
> clean AMD/radv host for the Deno run.)

## What it shows

Type a sentence (the default `the cat sat on the mat . the cat ran` repeats
"the cat" so an induction head fires). Two linked views:

- **attention heatmap** for the selected `layer`/`head` — a causal (lower-
  triangular) `token × token` grid; hover a cell for the exact weight + the
  query→key token pair.
- **12×12 head-map** — every head, coloured by its role (blue = previous-token,
  green = induction, amber = attention-sink, grey = none), intensity by score.
  Click any cell to inspect that head. This is how you *find* the famous heads.

## Render tier

T2 (Canvas2D heatmap + head-map; the "compute" tier is the GPT-2 forward in JS).

## Wiring

`layout.mount()` + `controls.text('sentence')` + `layer`/`head` steppers + a
"jump to induction head" button + a "WebGPU compute" toggle + "load real GPT-2".
The forward runs on the CPU ([`gpt2.js`](gpt2.js)) or the GPU
([`gpt2-webgpu.js`](gpt2-webgpu.js)); it is async and
degrades to a **labelled idealized synthetic stand-in** offline (the three head
shapes, hand-built on the default sentence) — never blank, headless-verifiable.
A 25 s timeout guards the tokenizer import. Two **challenges** (`?ch=N`): load the
real model, and find an induction head (score ≥ 0.30). Headless hooks: `?ids=`
(inject exact token ids, skip the tokenizer — the real-forward verification
path), `?weights=` (override the safetensors URL, e.g. a local `file://`),
`?layer=`/`?head=`, `?real=0` (synthetic only), `?hover=x,y`. Source:
[`page.js`](page.js), [`gpt2.js`](gpt2.js).
