# flash-attention -- tiled online softmax (never materialize N×N)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: how FlashAttention computes exact attention without ever
storing the full `N×N` score matrix, by streaming K/V in tiles and keeping a
running ("online") softmax. **Anchor**: A3 attention pattern (Family B9, the
T3 capstone; see `../plan/curriculum.md`).

## What it shows

A block of `Bq` query rows attends over `N` keys split into tiles of `Bk`.
Scrub the transport through the tiles; for each tile the page shows the
online-softmax update, per query row `i`:

- `mᵢ ← max(mᵢ, rowmax(Sⱼ))` — running max (for numerical stability)
- `rescale = exp(mᵢ_old − mᵢ_new)` — shrink the prior contributions when the
  max grows (shown when a new max appears)
- `lᵢ ← rescale·lᵢ + Σ exp(Sⱼ − mᵢ)` — running denominator
- `Oᵢ ← rescale·Oᵢ + Σ exp(Sⱼ − mᵢ)·Vⱼ` — running output accumulator

and at the last tile, `output = O / l`. The score matrix is drawn ghosted with
**only the current tile lit** — that tile is all that exists in SRAM at once,
so memory is `O(N)` instead of the `O(N²)` of materializing the whole matrix
(the
[`../scaled-dot-attention/`](../scaled-dot-attention/README.md) view). The
running `m`, `l`, and `O` panels update tile by tile.

## Render tier

T3 (prefers WebGPU; negotiates down to WebGL2 heatmaps, then Canvas2D — the
overlays/ghosting are Canvas2D regardless).

## Wiring

`layout.mount()` + controls (`keys N`, `tile size`, `seed`) + a per-tile
`Transport` that runs the online-softmax algorithm in `page.js` (uses
`tensor.seededRandn`), drawn with `render.heatmap` + `ctx`. Auto-plays + loops
the tile sweep; hover a score tile cell for its value + the running stats
(`m`, `l`, `rescale`) at that step, or an `O`/output cell for its partial value;
DRAG the `◂▸` handle to scrub the sweep by hand. Headless hooks:
`?pos=N`/`?step=N` set the tile, `?hover=x,y` sets the cursor (both pause the
transport); `?play=1` resumes. Source: [`page.js`](page.js).
