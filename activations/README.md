# activations -- GELU (erf / tanh) vs SiLU/Swish vs Mish

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: the smooth activation functions used in transformer MLPs,
plotted side by side. **Anchor**: A7 activation in MLP (Family C3, Phase 3;
see `../plan/curriculum.md`).

## What it shows

Four activations on one axis:

- **GELU (erf, exact)** — `0.5·x·(1 + erf(x/√2))`.
- **GELU (tanh approx)** — `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`; the
  cheaper form most GGUF/llama.cpp models use (`gelu_pytorch_tanh`).
- **SiLU / Swish** — `x·σ(x)`; the gate in SwiGLU.
- **Mish** — `x·tanh(softplus(x))`.

All four are smooth, non-monotonic (a small negative dip), `≈ x` for large
positive `x` and `≈ 0` for large negative `x` — which is why they're nearly
interchangeable. Drag the **x-marker** horizontally (or hover the plot) to read
all four values at a point you control — a vertical guide + a dot on each curve
+ a value tooltip track the cursor; on load the marker **sweeps on its own** and
freezes the moment you grab it. Scrub the transport to highlight each (its curve
bolds and its formula shows); toggle **zoom near 0**, because the differences
only show up around the dip (GELU's is shallowest ≈ −0.17, Mish's deepest ≈ −0.31).

## Render tier

T1 (Canvas2D line plot + axes).

## Wiring

`layout.mount({animate:true, onPointer})` + a `zoom` toggle + a per-activation
`Transport`, drawn with `ctx` and `tensor.silu`/`gelu` (Mish computed inline).
The x-marker is dragged via `onPointer` (horizontal) and read back through the
`pixel↔x` map captured in `draw`; the ambient sweep uses `api.t` and freezes on
drag/hover. Headless hooks: `?x=VALUE` (set marker x, freeze sweep), `?hover=x,y`
(fake cursor for the tooltip + guide), plus the existing `?step=N`/`?play=1`/
`?zoom`. Source: [`page.js`](page.js).
