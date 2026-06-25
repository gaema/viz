# convolution -- kernel sliding over input (stride / padding / dilation)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: 2-D convolution as a small kernel sliding across an input
feature map, each output pixel a **sum of products** over the receptive field --
with the geometry knobs that set the output size: **stride**, **padding**, and
**dilation**. **Anchor**: F (CNN / vision); Family F (see
`../plan/curriculum.md`).

## What it shows

A `k×k` kernel `W` slides over the (zero-padded) input `X`. At each output
position `(oy, ox)` it multiplies its weights with the input cells under it and
sums:

    out[oy, ox] = Σᵢⱼ W[i, j] · Xpad[oy·s + i·dil,  ox·s + j·dil]

- **stride `s`**: how far the kernel jumps between output pixels (bigger stride ⇒
  smaller output, coarser sampling).
- **padding `p`**: a border of zeros around `X` (keeps the output from shrinking;
  "same" padding).
- **dilation `dil`**: gaps between the kernel taps (a larger receptive field for
  the same `k` weights -- atrous convolution).

Output size: `out = ⌊(n + 2p − dil·(k−1) − 1)/s⌋ + 1`. The current receptive
field is outlined on the input with the kernel weights overlaid; the output grid
fills as the kernel slides (auto-plays + loops), and the readout shows the
sum-of-products for the current pixel.

**Drag** any input cell to change its value and watch the affected outputs
recompute; **hover** any input / kernel / output cell for its value (and an
output cell's full Σ). Step to slide the kernel one position at a time.

## Render tier

T1 (Canvas2D: the padded input grid + kernel overlay, the output grid, and the
sum-of-products).

## Wiring

`layout.mount()` + controls (`n`, `k`, `stride`, `pad`, `dilation`, `seed`) + an
output-position `Transport` (slide the kernel) + `onPointer` input-cell drag +
hover. `?step` / `?drag` / `?hover` hooks. Source: [`page.js`](page.js).
