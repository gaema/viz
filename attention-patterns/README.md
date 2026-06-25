# attention-patterns -- full / sliding-window / hybrid / sink

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the masking schemes that decide which keys each query may
read, and what they cost. **Anchor**: A3 attention pattern (Family B).

## What it shows

An `N×N` grid (query rows × key cols) where attended cells are lit and masked
cells dark. Scrub the transport through four patterns:

- **Full (causal)** — every query attends to all past keys (`j ≤ i`). Attended
  set grows with position → `O(i)` per query, unbounded KV cache.
- **Sliding-window (SWA)** — query attends only to the last `w` keys
  (`i−w < j ≤ i`). Bounded `O(w)` per query / cache (Mistral).
- **Hybrid by layer** — a strip of per-layer grids alternating **local** (SWA)
  and **global** (full): most layers cheap, a few see everything (Gemma 2/3:
  5 local : 1 global).
- **Attention sink** — sliding window **plus** the first `s` tokens always
  attended (the "sink"), which stabilizes very long / streaming contexts
  (StreamingLLM).

A `highlight query i` slider outlines one row and the readout reports its
**attended-key count** under the current pattern — making the `O(i)` vs `O(w)`
cost difference concrete. Sliders for window `w`, sink size `s`, and `N`.

## Render tier

T2 (WebGL2-tier surface; the grids are drawn as Canvas2D cell overlays for the
3-way attended / current-row / masked coloring).

## Wiring

`layout.mount()` + controls (`N`, `window`, `sink`, `query i`) + a per-pattern
`Transport`, drawn with `ctx` + `render.label`. Per the 2026-06-13 interactivity
contract: hover any cell for whether it is attended **and why** for the current
pattern; **drag the `◀` current-query handle** down the rows (the attended band
follows live); **drag the window edge** on sliding/sink to resize `w`; the query
**auto-sweeps down + loops** (generation) and the pattern transport loops.
Headless hooks: `?step=N`/`?play=1`, `?hover=x,y` (attended-reason tooltip),
`?q=N` (current query row), `?win=N` (window size). Source: [`page.js`](page.js).
