# scaled-dot-attention -- the single-head pipeline

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the full scaled dot-product attention computation over a
sequence, stage by stage. **Anchor**: A3 attention pattern (Family B2).

## What it shows

`Attention(Q,K,V) = softmax(QKᵀ/√d + mask)·V`, with the N×N score matrix as the
centerpiece. The pipeline auto-plays + loops; **drag a Q (query) cell vertically**
to re-steer that query and watch its row of scores, the softmax weights, and the
output recompute live; **hover** a score / weight / output cell for its derivation.
Scrub the transport through the five stages:

1. **QKᵀ** — each cell `[i,j]` is `qᵢ·kⱼ` (query i against key j).
2. **/ √d** — scale, so the softmax doesn't saturate.
3. **causal mask** — query `i` may only see keys `j ≤ i`; the upper triangle
   goes to −∞ (shown veiled).
4. **softmax (rows)** — each row becomes a probability distribution over the
   visible keys (lower-triangular, rows sum to 1).
5. **× V → output** — each output row is `Σⱼ wᵢⱼ·vⱼ`, the weighted sum of values.

Q, K, V are shown as input strips; the output `[N×d]` appears at the final
stage. This is the **prefill** view (all positions at once) — the companion to
[`../kv-cache/`](../kv-cache/README.md), which shows the decode view (one new
query over a growing cache).

## Render tier

T2 (WebGL2 heatmaps for the matrices; Canvas2D overlays for the causal veil,
labels, and stage highlight).

## Wiring

`layout.mount()` + controls (`tokens`, `head_dim`, `seed`) + a 5-stage
`Transport` built in `page.js` (uses `tensor.dot`/`softmax`/`seededRandn`),
drawn with `render.heatmap`/`cell` + `ctx`. Interactivity per the framework
contract (`../design/framework.md`): `autoplay`+`loop` transport, `onPointer` Q-cell
drag with `cellAt` hit-testing + `recompute()`, `setTip` hover derivations.
Headless hooks `?hover=x,y` / `?drag=i,c,val` / `?step=N` / `?play=1` (drag/hover
pause + park on a deterministic stage) as on the dot-product + matmul pages.
Source: [`page.js`](page.js).
