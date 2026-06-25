# transformer-block -- one full pre-norm block

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: a whole transformer block end to end --
`x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> + -> out` -- showing the
**residual spine** and the two sublayer branches that read it (via norm) and add
back. **Anchor**: the block is the unit the per-attribute A1-A12 pages assemble
into; Family C capstone.

## What it shows

A token sequence `x [N×D]` flows along a residual spine. Two sublayers branch
off it and add back:

- **Attention**: `a = Attn(RMSNorm(x))`, then `h1 = x + a`.
- **MLP (SwiGLU)**: `m = MLP(RMSNorm(h1))`, then `out = h1 + m`.

This is **pre-norm** (norm sits on the read branch; the spine itself is never
normalized -- the identity highway from [residual-stream](../residual-stream/README.md)).
Every tensor is really computed (Q/K/V, causal softmax, SwiGLU); the sublayer
internals live in their own pages
([scaled-dot-attention](../scaled-dot-attention/README.md),
[mlp-gated](../mlp-gated/README.md), [normalization](../normalization/README.md)).

Step the transport through the six stages (norm1 -> attn -> add -> norm2 -> mlp
-> add); **drag any x cell** to steer the input and watch the whole block
recompute; **hover** any cell for its value + stage.

## Render tier

T2 (WebGL2 heatmaps for the spine + branch tensors; Canvas2D for the boxes,
arrows, and residual arcs).

## Wiring

`layout.mount()` + controls (`N`, `D`, `seed`) + a 6-stage `Transport` +
`onPointer` drag on `x` + hover tooltips. `?step` / `?drag` / `?hover` headless
hooks. Source: [`page.js`](page.js).
