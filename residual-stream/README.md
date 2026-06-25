# residual-stream -- the residual highway

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the residual stream — the tensor that runs straight through
the transformer, that every block reads from and writes back to by **adding**
(never overwriting). **Anchor**: A8 (Family C5).

## What it shows

For one token, its residual stream traced **through depth**: a `[depth × D]`
grid where each row is the stream after one more block —
`embed → +attn₀ → +mlp₀ → +attn₁ → +mlp₁ → final norm`. Each block reads the
stream (through a norm), computes a contribution, and **adds** it back, so the
stream is a running sum — the "highway". A `‖stream‖` column shows the
per-depth magnitude. Scrub the transport across **tokens** to see each token's
stream; toggle **pre / post-norm**:

- **pre-norm** — `stream ← stream + sublayer(norm(stream))`. The stream itself
  is never normalized inside the stack, so its **magnitude grows with depth**
  (visible in the column); a single **final norm** rescales before the lm_head.
  Modern, stable.
- **post-norm** — `stream ← norm(stream + sublayer(stream))`. The stream is
  re-normalized after **every** add, so the magnitude stays ~1 throughout.
  Original transformer.

A small read-and-add unit diagram shows where the norm sits in each case.

**Direct manipulation:** drag a cell of the `[depth × D]` heatmap to edit that
block's contribution (top row = the embedding `x[t,d]`, the rows below = each
block's delta) and the running sum from that depth **downward** — plus the
`‖rms‖` column — recomputes live, showing one block's write propagating down
the highway. Hover any depth cell for `depth "<block>" dim d = value`, the rms
column for how the magnitude grew from the previous depth, or a flow box for
what that step does.

## Render tier

T1 (Canvas2D heatmap + magnitude column + a flow diagram).

## Wiring

`layout.mount()` + controls (`tokens N`, `features D`, `seed`, `pre/post`) + a
per-token `Transport` (auto-plays + loops), drawn with `render.heatmap`/`cell`
+ `ctx` and `tensor.seededRandn`; `onPointer` hit-tests the depth×D heatmap for
hover + drag. `?step`/`?prenorm`/`?play`/`?hover=x,y`/`?drag=row,col,val`
headless hooks. Source:
[`page.js`](page.js).
