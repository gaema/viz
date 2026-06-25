# multi-head -- split into heads, attend in parallel, concat

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: how multi-head attention slices the model dimension into H
independent heads, each running its own scaled-dot-attention, then concatenates
the results. **Anchor**: A2 head_dim layout (Family B3; see
`../plan/curriculum.md`).

## What it shows

Q, K, V are `[N×D]`; the D columns are sliced into **H heads** of width
`head_dim = D/H`. Each head `h` attends **only within its own slice**
(`softmax(QₕKₕᵀ/√head_dim)` with a causal mask) — so the **H attention
matrices differ** (different heads learn different patterns). Each head's
output `[N×head_dim]` is written back into its slice, and the slices
**concatenate** into the `[N×D]` output. Scrub the transport to walk head by
head: the current head's Q slice, its `N×N` attention matrix, and its output
slice are highlighted, making the head_dim slicing + parallel-then-concat
structure concrete. The per-head attention auto-plays + loops; **hover** any
per-head attention cell for its derivation (`head h: score[i,j] = qᵢ·kⱼ/√d_head
= value`) or a concat-output cell to see which head + dim it came from; **drag**
a Q cell (in one head's slice) vertically to change that query component and
watch only that head's attention row and its concat-output slice recompute live.
This is single-head attention
([`../scaled-dot-attention/`](../scaled-dot-attention/README.md)) run H times in
parallel.

## Render tier

T2 (WebGL2 heatmaps for the Q/output strips + the per-head matrices; Canvas2D
overlays for head dividers, slice highlight, causal veil).

## Wiring

`layout.mount()` + controls (`tokens`, `heads`, `head_dim`, `seed`) + a
per-head `Transport` built in `page.js` (uses `tensor.dot`/`softmax`/
`seededRandn`), drawn with `render.heatmap`/`cell` + `ctx`. Interaction hooks:
`?step=N`/`?play=1` plus `?hover=x,y` (canvas-space cursor → tooltip) and
`?drag=head,i,c,val` (set `Q[i, head*hd + c]` = `val`, the headless stand-in for
a Q-cell drag in `head`'s slice). Source: [`page.js`](page.js).
