# kv-cache -- the KV cache during decode

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: why autoregressive decode keeps a per-layer cache of past
keys and values, how it fills, and how its memory grows with context.
**Anchor**: A4 KV cache layout (Family B7, the headline attention page; see
`../plan/curriculum.md`).

## What it shows

Three column-aligned bands over the sequence positions: the **K cache**, the
**attention weights** of the current decode token, and the **V cache**. Scrub
the transport to advance the decode token: each step **appends one column** to
K and V (the new token's key/value), the query `q_t` attends over **all
cached keys** (`softmax(q_t·Kᵀ/√d)`), and the **output** is the weight-weighted
sum of the cached value columns. The "future" columns are veiled until reached,
so you watch the cache fill. A live **memory** readout shows
`2·L·H_kv·head_dim·dtype·seq` growing linearly with context, with a **KV dtype**
selector (fp16 / int8 / int4) for the quantized-KV variants — and the reminder
that without the cache each step would recompute K,V for every past token.

## Render tier

T2 (WebGL2 heatmaps for the K/V grids; Canvas2D overlays for weights, query,
output, veil, memory bar).

## Wiring

`layout.mount()` + controls (`head_dim`, `context`, `seed`, `layers`,
`KV dtype`) + a step `Transport` over a per-token attention sequence built in
`page.js` (uses `tensor.dot`/`softmax`/`seededRandn`), drawn with
`render.heatmap`/`cell` + `ctx`. The decode **auto-plays + loops** (the cache
fills column-by-column on load — the headline), hovering any K/V/attn cell
inspects its `(pos, layer, dim)` + value (or when an unfilled cell gets
written), and a **drag handle** scrubs the decode position by hand.
Headless hooks: `?pos=N`/`?drag=N` (filled-column count), `?hover=x,y`,
plus the existing `?step=N`/`?play=1`. Source: [`page.js`](page.js).
