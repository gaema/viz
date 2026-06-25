# quantization -- fp16 → int4, group-wise

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: **group-wise affine weight quantization** -- how fp16 weights
become low-bit integers (int4, even int3/int2) so a large model fits in a small
amount of memory, and what error that costs. **Anchor**: G (quantization /
deployment); Family G (see `../plan/curriculum.md`).

## What it shows

Weights are chopped into **groups** of `G`. Per group:

```
s = (max − min) / (2^bits − 1)        scale
z = round(−min / s)                   integer zero-point
q = clamp(round((x − min)/s), 0, 2^bits−1)   the stored code
x′ = min + s·q                        dequantized value (what the GPU uses)
```

The weights panel draws, per weight, the **dequantized bar** `x′` with a **black
tick** at the **original** `x` -- the gap between them is the quantization error.
Faint horizontal lines mark each group's `2^bits` reconstruction **levels**, so
you can see the weights snapping onto them. Two knobs trade off:

- **bits** -- more bits = more levels = finer steps = **less error** (but a bigger
  file). The **RMSE-vs-bits curve** (bottom-right) plots this, with the current
  bit-width marked.
- **group size `G`** -- smaller groups adapt `s` to the **local** range, so an
  outlier only coarsens its own small group (**less error**); but the per-group
  `scale + zero-point` overhead grows, **lowering the compression ratio**
  (shown as effective bits/weight and `×` vs fp16).

Pick the **outlier** preset (or **drag** any weight to an extreme) and watch its
group's scale stretch and every neighbour in that group get coarser -- exactly why
small groups and outlier-aware methods (AWQ/GPTQ) exist.

**Drag** a weight ↕ to change it; sliders for **bits** and **seed**; selects for
the weight distribution and **group size**; the group scan animates; hover a
weight to see its `x / q / x′ / err`.

## Render tier

T1 (Canvas2D: the weights bar panel with level lines + the RMSE-vs-bits curve).

## Wiring

`layout.mount()` + controls (`preset`, `bits`, `G`, `seed`) + `animate` (the group
scan) + `onPointer` drag-the-weight + hover. `?preset` / `?bits` / `?G` / `?seed` /
`?drag` / `?hover` hooks. Source: [`page.js`](page.js).
