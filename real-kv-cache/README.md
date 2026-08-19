# real-kv-cache -- what fills up as GPT-2 generates (Phase 9)

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server + a one-time ~548 MB weight download: `python3 -m http.server 8099`

Interactive page: the **real** counterpart to the synthetic
[`kv-cache`](../kv-cache/README.md) page. It shows GPT-2's **actual** KV cache as
it generates. **Anchor**: the KV-cache concept. Part of **Phase 9 — real-model
grounding**.

## What it shows

As GPT-2 generates, every layer caches the **Keys and Values** for all past
positions, so each new token only attends — it doesn't recompute the past. The
cache grows one position per token: the **prompt** fills it in parallel
(*prefill*), then each generated token appends **one column** (*decode*).

`GPT2.cache` (in [`../real-attention/gpt2.js`](../real-attention/gpt2.js))
returns the real per-layer K/V plus the next-token logits from one forward. The
page draws:

- the **actual cached K values** for a chosen layer/head (`positions × 64 dims`),
 with the prefill/decode boundary marked;
- the **real memory** — using GPT-2's true dims (`2 × 12 layers × 768 dim ×
 dtype`) — climbing live as you generate, plus a KV-cache-vs-context-length
 curve (128 → 32K) showing why long context is expensive.

Pick a dtype (fp16 / fp32 / int8), hit generate, and watch both the cache and the
megabytes grow.

## Verified against PyTorch

`GPT2.cache` is checked against `past_key_values` by
`../real-attention/gpt2.test.mjs`: the per-
layer/head K and V tensors match PyTorch within **max|Δ| ≈ 5e-5** (fixture
[`gpt2-groundtruth.fixture.json`](../real-attention/gpt2-groundtruth.fixture.json)).

## Render tier

T2 (Canvas2D K-cache heatmap + memory bars; the "compute" tier is the GPT-2
forward in JS, one per generated token).

## Wiring

`layout.mount` + `controls.text('prompt')` + `layer`/`head` steppers + a `dtype`
select + `gen tokens` + prefill / generate / "load real GPT-2" buttons. Reuses
`gpt2.js` `cache`; async, degrades to a labelled synthetic K cache offline (the
memory math is real either way — the dims are GPT-2's). Two **challenges**
(`?ch=N`): load the real model, and grow the cache by ≥ 8 decode positions.
Headless hooks: `?prompt=`, `?ids=`, `?gen=N`, `?layer=`/`?head=`/`?dtype=`,
`?real=0`. Source: [`page.js`](page.js).
