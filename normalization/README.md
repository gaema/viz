# normalization -- RMSNorm vs LayerNorm (and pre/post-norm)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the two normalizations used in transformers, applied per
token (per row), and where the norm sits relative to the residual.
**Anchor**: A8 normalization (Family C4).

## What it shows

An input `[N × D]` (rows = tokens, cols = features). Each row is normalized
**independently**, using statistics computed over its own `D` features:

- **LayerNorm** — `y = (x − μ)/√(σ² + ε)`: subtract the row **mean** μ, divide
  by the row **std** σ (then scale γ + shift β). Centers and scales.
- **RMSNorm** — `y = x/√(mean(x²) + ε)`: divide by the row **RMS** only. No
  mean-subtract, no bias — fewer ops, what Llama-family models use.

The per-row stats (μ and σ, or RMS) are shown as columns next to the input, and
the normalized output on the right. Scrub the transport to walk row by row
(highlighting that token's stats + before/after). Toggle **RMSNorm / LayerNorm**
and **pre / post-norm**:

- **pre-norm** — `out = x + sublayer(norm(x))`: norm sits on the residual
  branch (modern, stable training).
- **post-norm** — `out = norm(x + sublayer(x))`: norm after the add (original
  transformer).

A small diagram shows the placement difference.

## Render tier

T1 (Canvas2D heatmaps + a flow diagram).

## Wiring

`layout.mount()` + controls (`tokens N`, `features D`, `norm`, `pre/post`) + a
per-row `Transport`, drawn with `render.heatmap`/`cell` + `ctx` and
`tensor.seededRandn`. `?step`/`?norm`/`?prenorm` headless hooks. Source:
[`page.js`](page.js).
