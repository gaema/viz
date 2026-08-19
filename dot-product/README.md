# dot-product -- a · b (projection + cosine)

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the dot product of two vectors, built term-by-term, with the
geometric meaning shown alongside. **Anchor**: foundational primitive (Family
A2; the inner loop of every matmul / attention score / projection).

## What it shows

`a · b = Σₖ aₖ·bₖ`. Two vector strips (a, b) and a product strip `aₖ·bₖ`
that fills in as you scrub the transport; the running sum accumulates with
each term. A geometry panel draws **a** and **b** at their true angle θ
(the real k-dimensional angle, faithfully shown in 2-D) with the scalar
projection of b onto a shaded along a — making `a·b = |a||b|cosθ` and the
cosine similarity `cosθ = a·b / (|a||b|)` concrete. The **alignment**
slider drives b from aligned with a (cosθ→1) through orthogonal to
anti-aligned (cosθ→−1).

## Render tier

T1 (Canvas2D). Small 1×k strips + vector graphics; no GPU needed.

## Wiring

`layout.mount` + controls (dim `k`, `seed`, `alignment`) + the step
`Transport` walking `tensor.dotSteps` (per-term accumulation), drawn with
`render.heatmap`/`cell`/`arrow`/`label`. `?step=N`/`?play=1` headless
hooks as on the matmul page. Source: [`page.js`](page.js).
