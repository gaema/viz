# vram-budget -- where the GPU memory goes

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **live VRAM math for LLM inference** -- the four things that
consume GPU memory, to scale against a chosen GPU, and *why* a long prompt runs
out of memory. **Anchor**: G (quantization / deployment); Family G ().

## What it shows

A stacked bar (vs the GPU's capacity, red when it OOMs) of:

- **weights** = `params × bytes/weight` -- fixed once you pick the model size and
  the **weight quant** (fp16 = 2 B, int8 = 1, int4 = 0.5).
- **KV cache** = `2 · L · (d/H · n_kv_heads) · seq · batch · kv_bytes` -- caches
  every past token's keys & values, so it grows **linearly with context length
  and batch**. **GQA** (`n_kv_heads ≪ n_heads`, e.g. Llama-70B) and a low **KV
  dtype** shrink it.
- **activations** -- transient forward-pass buffers: tiny during **decode** (one
  token), larger during a big **prefill**.
- **overhead** -- the fixed CUDA context + fragmentation.

The **KV-cache-vs-context curve** plots the KV block as the sequence length grows
and marks where it **crosses the weights line** -- past that point you are
*context-bound*, not *model-bound*, which is the usual reason a long prompt OOMs.

Try **int4** weights (shrinks the blue block), **Llama-70B (GQA)** (shrinks the
orange KV block despite being a bigger model), or **drag the context marker** out
to 128k and watch the KV cache explode past the GPU.

**Drag** the marker on the curve to set the context length; selects for model,
weight quant, KV dtype, phase, and GPU; sliders for context and batch; the OOM
border pulses, a ghost dot sweeps the KV curve; hover a bar segment for its
formula.

## Render tier

T1 (Canvas2D: the stacked VRAM bar + capacity line + the KV-vs-context curve +
breakdown panel).

## Wiring

`layout.mount()` + controls (`model`, `wq`, `kvq`, `context`, `batch`, `mode`,
`gpu`) + `animate` (the OOM pulse + KV ghost sweep) + `onPointer` drag-the-context
+ hover. `?model` / `?wq` / `?kvq` / `?context` / `?batch` / `?mode` / `?gpu` /
`?hover` hooks. Source: [`page.js`](page.js).
