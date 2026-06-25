# qkv-projection -- one embedding → Q, K, V

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a token's embedding is projected into query, key, and
value vectors by three learned weight matrices. **Anchor**: A5 embedding / A2
head_dim (Family B1).

## What it shows

One token embedding `x` (dim `D`), and three weight matrices `W_Q`, `W_K`,
`W_V` (`D×D`). The projections `q = x·W_Q`, `k = x·W_K`, `v = x·W_V` produce
three vectors from the **same** embedding — the token's three roles:

- **Q (query)** — what this token is looking for.
- **K (key)** — what it offers to others.
- **V (value)** — what it carries if attended to.

Scrub the transport through the output dimension `o`: each step highlights
column `o` of all three weight matrices, dots it with `x`
(`out[o] = Σₖ x[k]·W[k,o]`), and fills `q`, `k`, `v` element by element. **Drag
any embedding cell ↕** and all three projections recompute live — the one input
fanning out to three roles — while hovering a `q`/`k`/`v` cell shows its full
dot-product derivation. It is
three matrix-vector products sharing one input — the step before attention
([`../scaled-dot-attention/`](../scaled-dot-attention/README.md)) and the
multi-head split ([`../multi-head/`](../multi-head/README.md)).

## Render tier

T1 (Canvas2D heatmaps for the embedding, weight matrices, and outputs).

## Wiring

`layout.mount({autoplay, onPointer})` + controls (`dim`, `seed`) + a
per-output-dim `Transport` (`loop:true`), drawn with `render.heatmap`/`cell` +
`ctx` and `tensor.seededRandn`. Interactive per the framework contract: hover
(`setTip` + `cellAt`), drag an embedding cell (`onPointer` + re-project), and
autoplay+loop. Headless hooks: `?step=N`/`?play=1` plus `?hover=x,y` (set
pointer + pause) and `?drag=d,val` (set `x[d]`, re-project, pause). Source:
[`page.js`](page.js).
