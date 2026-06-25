# mlp-gated -- the SwiGLU / GeGLU feed-forward block

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the gated MLP used in modern transformers (Llama, etc.) —
two parallel up-projections, an element-wise gate, and a down-projection.
**Anchor**: A6 MLP shape / A7 activation (Family C2).

## What it shows

The block, top to bottom (rows column-aligned so the element-wise step reads
vertically):

1. `gate = x · W_gate` — project `x` up to the intermediate width `I`.
2. `up = x · W_up` — a **second, parallel** projection to the same width.
3. `act(gate)` — apply the activation: **SiLU** (→ SwiGLU) or **GELU**
   (→ GeGLU), toggleable.
4. `hidden = act(gate) ⊙ up` — the **gate**: `act(gate)` modulates `up`
   element-by-element (column `i` of `hidden` = column `i` of `act(gate)` ×
   column `i` of `up`).
5. `out = hidden · W_down` — project back down to `D`.

Scrub the transport through the five stages (auto-plays + loops); rows reveal
as they're computed and the active stage is highlighted. **Drag any input `x`
cell vertically** and gate, up, gated, and out all recompute live — the
element-wise gating made tangible; **hover** a gate / up / gated cell for its
derivation (`hidden[j] = act(gate[j]) · up[j]`). The intermediate width `I = 2·D` here
(real models use ~2.7× for SwiGLU to keep the parameter count of the 3 matrices
comparable to a 4× un-gated FFN).

## Render tier

T1 (Canvas2D heatmaps for the row vectors).

## Wiring

`layout.mount()` + controls (`dim D`, `seed`, `activation`) + a per-stage
`Transport`, drawn with `render.heatmap` + `ctx` and `tensor.silu`/`gelu`/
`seededRandn`. Interactive per the framework contract: `onPointer` drag of `x`
+ `cellAt` hover-to-inspect + autoplay/loop. Headless hooks: `?step=N`, `?play=1`,
`?act=silu|gelu`, `?hover=x,y`, `?drag=i,val` (set input cell `i`, e.g. `?drag=2,1.5`).
Source: [`page.js`](page.js).
