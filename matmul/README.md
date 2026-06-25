# matmul -- C = A · B (the GEMM atom)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: visualizes matrix multiply as the operation every other
mechanism in the zoo reduces to. **Anchor**: foundational primitive (used by
every A1-A12 shape; see `../plan/curriculum.md`
Family A1).

## What it shows

`C[i,j] = Σ_k A[i,k]·B[k,j]`. The three matrices sit in the canonical
corner layout -- **A** bottom-left (m×k), **B** top-right (k×n), **C**
bottom-right (m×n) -- so output cell `C[i,j]` lies at the intersection of
A's row `i` and B's column `j`. Scrub the step transport to watch C fill
in: per output cell (the dot product of a row and a column), or per
multiply-add term with `step per term` on. The active row/column/cell are
outlined live, and the readout strip shows the running computation.

## Render tier

T2 (WebGL2 heatmaps, Canvas2D fallback). The heatmap raster is the only
GPU-touching part; A/B/C and all overlays share one Canvas2D surface.
Colors come from the shared diverging ramp (0 = white), so this page is a
witness for the **Canvas2D-vs-WebGL2 pixel-identity gate** -- see
`../framework/gate-heatmap.html` +
`run-gate.sh`.

## Wiring (framework exercise)

First end-to-end page: `layout.mount()` chrome + `controls` (dim steppers,
seed, per-term toggle) + the step `Transport` walking `tensor.matmulSteps`,
rendered with `render.heatmap`/`cell`/`label`. Source: [`page.js`](page.js).
