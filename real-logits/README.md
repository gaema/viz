# real-logits -- what GPT-2 actually predicts next (Phase 9)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server + a one-time ~548 MB weight download: `python3 -m http.server 8099 --directory viz`

Interactive page: the **real** counterpart to the synthetic
[`lm-head`](../lm-head/) + [`sampling`](../sampling/) pages. It runs a **real
GPT-2 (124M)** in your browser to get the actual next-token distribution for a
prefix you type, and lets you reshape it with temperature / top-k / top-p exactly
as a sampler would. **Anchor**: A5 lm_head + the sampling concept. Third (final)
increment of **Phase 9 — real-model grounding** (`../plan/phase9.md`).

## What it shows

Type a prefix. The page runs the verified GPT-2 forward
([`../real-attention/gpt2.js`](../real-attention/gpt2.js)), applies the final
`ln_f`, and projects through the **tied lm_head** (`logits = ln_f(hₙ)·wteᵀ`, no
bias) to get a logit for every one of the 50257 vocabulary tokens. A softmax (at
your chosen **temperature**) turns those into probabilities, drawn as a bar chart
of the most likely next tokens. The default `the cat sat on the mat . the cat
ran` really predicts **up / away / off / out** — a sensibly flat distribution.

The **top-k** and **top-p (nucleus)** controls show how a sampler truncates the
distribution: tokens the sampler would drop go grey, the greedy (argmax) token is
highlighted. This ties the abstract sampling page to a real model's numbers.

## Verified against PyTorch

`GPT2.logits()` is checked against `GPT2LMHeadModel` by
[`../real-attention/gpt2.test.mjs`](../real-attention/gpt2.test.mjs) over the
committed fixture: the **argmax next-token id matches** (510 = `" up"`) and the
top-token logits match within **max|Δ| ≈ 1.1e-4**. The softmax/top-k/top-p display
math is exercised by the page's challenge checks.

## WebGPU lm_head (optional)

The forward — including the **lm_head matmul** (`logits[v] = Σ_d ln_f(hₙ)[d]·wte[v,d]`
over the full 50257-token vocabulary) — also runs on the **GPU** via
[`../real-attention/gpt2-webgpu.js`](../real-attention/gpt2-webgpu.js): the 12
blocks stay resident on-device, then `ln_f` and the tied lm_head run as WGSL
compute dispatches (the lm_head shader reads each `wte[v,:]` row directly — no
transpose), and only the `[V]` logit vector is read back. The 154 MB `wte` is
uploaded to the GPU lazily, only when logits are first requested. Toggle **“WebGPU
compute”**.

Like real-attention it is **self-verifying**: on the first GPU run the page also
runs the CPU `logits()` and reports `max|Δ|` in the banner, falling back to the
verified CPU path on any mismatch. The GPU lm_head path is **author-verified** —
`gpt2-webgpu.test.mjs` (run under Deno on an AMD radv GPU) reproduces the CPU
logits within **worst |Δ| ≈ 2.4e-7** with the same argmax.

## Render tier

T2 (Canvas2D bar chart; the "compute" tier is the GPT-2 forward + lm_head in JS).

## Wiring

`layout.mount()` + `controls.text('prompt')` + `temperature` / `top-k` / `top-p`
controls + a "WebGPU compute" toggle + "load real GPT-2". The forward reuses the verified
[`gpt2.js`](../real-attention/gpt2.js) loader; it is async and degrades to a
labelled **idealized synthetic distribution** offline (so the sampling controls
still teach) — never blank, headless-verifiable. Two **challenges** (`?ch=N`):
load the real model, and use nucleus sampling to keep ≤ 5 tokens. Headless hooks:
`?prompt=`, `?temp=`, `?topk=`, `?topp=`, `?real=0` (synthetic only). Source:
[`page.js`](page.js).
