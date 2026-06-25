# causal-mask -- why decode only attends to the past

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: the lower-triangular causal mask, and the reason for it —
an autoregressive LM predicting the next token must not peek at it.
**Anchor**: A3 attention pattern (Family B; see
`../plan/curriculum.md`).

## What it shows

An `N×N` grid: rows are **query** positions, columns are **key** positions,
over a real token sequence. A cell `[i,j]` is **visible** when `j ≤ i` and
**masked** (−∞ → 0 weight after softmax) when `j > i` — the lower triangle is
allowed, the upper triangle is masked. Scrub the query position `i`: the
current row lights its visible prefix (tokens `0..i`), the masked future is
darkened, and the readout points out that **query `i` is predicting token
`i+1`**, which sits in the masked region — so it can't cheat by reading the
answer. The key tokens on top dim as they cross into the future; the query
tokens on the left mark the current position. This is the mask step of
[`../scaled-dot-attention/`](../scaled-dot-attention/README.md), made its own
focus.

## Render tier

T1 (Canvas2D — a grid + token labels).

## Wiring

`layout.mount()` + controls (`tokens`) + a per-query-position `Transport` built
in `page.js`, drawn with `ctx` + `render.label`. `?step=N`/`?play=1` headless
hooks as on the other pages. Source: [`page.js`](page.js).
