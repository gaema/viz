# real-neurons -- which MLP units fire (Phase 9)

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server + a one-time ~548 MB weight download: `python3 -m http.server 8099`

Interactive page: real **MLP neuron activations** in GPT-2. Each transformer
block's MLP has 3072 "neurons" — the post-gelu units between the two projections,
where most of a model's learned features live. **Anchor**: the MLP / gated-FFN
concept ([`mlp-gated`](../mlp-gated/README.md)). Part of **Phase 9 — real-model
grounding**.

## What it shows

Type a sentence. `GPT2.mlp` (in [`../real-attention/gpt2.js`](../real-attention/gpt2.js))
returns the post-gelu activations for every layer — `tokens × 3072`. For the
chosen layer the page draws a heatmap of the **most-active neurons** (the columns)
across the sentence (the rows), with the single strongest `(token, neuron)` firing
ringed. Hover any cell for the activation.

The point: **different neurons fire for different tokens** — some are token- or
feature-specific. That selectivity is the raw material of mechanistic
interpretability. Deeper layers fire harder (larger activations).

## Verified against PyTorch

`GPT2.mlp` is checked against a forward hook on `transformer.h[l].mlp.act` by
`../real-attention/gpt2.test.mjs`: the
last-token **top-8 neuron ids and values** at layers 0 / 5 / 11 match PyTorch
(fixture [`gpt2-groundtruth.fixture.json`](../real-attention/gpt2-groundtruth.fixture.json)).

## Render tier

T2 (Canvas2D activation heatmap; the "compute" tier is the GPT-2 forward in JS).

## Wiring

`layout.mount` + `controls.text('sentence')` + `layer` / `top neurons` steppers
"load real GPT-2". Reuses `gpt2.js` `mlp`; async, degrades to a labelled
synthetic gelu-ish stand-in offline. Two **challenges** (`?ch=N`): load the real
model, and find a strongly-firing neuron (peak activation ≥ 4, deeper layers).
Headless hooks: `?text=`, `?ids=`, `?layer=`/`?topn=`, `?real=0`. Source:
[`page.js`](page.js).
