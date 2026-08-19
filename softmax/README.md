# softmax -- logits → probabilities

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a vector of logits becomes a probability distribution,
step by step, with temperature and the numerical-stability max-subtract shown.
**Anchor**: A9/A10 (lm_head logits + vocab → sampling; Family A4 foundational;
see `../plan/curriculum.md`).

## What it shows

`softmax(z)ᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)`, over three bar rows — **logits**,
**exp(zᵢ/T − max)**, **probabilities**. Scrub the transport through the
pipeline phases: find the **max** (highlighted; subtracted so `exp` never
overflows), build each **exp** term, **sum** them, then **normalize** each
into a probability (the prob bars fill and sum to 1). The **temperature**
slider shows the effect directly: T→large flattens toward uniform, T→small
sharpens toward a one-hot argmax.

## Render tier

T1 (Canvas2D). Bar charts; no GPU needed.

## Wiring

`layout.mount()` + controls (`k`, `seed`, `temperature`) + the step
`Transport` walking `tensor.softmaxSteps` (max → exp → sum → norm phases),
drawn with `ctx` bars + `render.label`. `?step=N`/`?play=1` headless hooks
as on the other pages. Source: [`page.js`](page.js).
