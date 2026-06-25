# dtype-bits -- floating-point & integer bit layouts

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: how a number is stored in fp32 / fp16 / bf16 / fp8 / int8 /
int4, and why fewer bits means more rounding error. **Click any bit to flip it**
and watch the reconstructed value rebuild bit by bit; hover a bit for its field +
place value, or a field label to decode it. **Anchor**: A11 special
pre-quant transforms / precision (Family A3 foundational; see
`../plan/curriculum.md`).

## What it shows

The focused dtype's bits laid out as **sign | exponent | mantissa** cells
(integers: **sign | magnitude**), colored by field. Scrub the transport to
**reveal bits left→right and rebuild the value**: the sign sets ±, the
exponent field sets the scale `2^(e−bias)` (with the implicit leading 1),
and each mantissa bit adds `scale·2⁻ⁱ`. A comparison table encodes the same
chosen value in every dtype and shows each one's total bits, decoded value,
and **rounding error** as a bar — making the bit-width ↔ precision tradeoff
concrete (fp32 ≈ exact, fp8/int4 visibly off).

## Render tier

T1 (Canvas2D). Bit cells + bars; no GPU needed.

## Wiring

`layout.mount()` + controls (`value`, `dtype`) + a step `Transport` over a
page-built bit-reveal sequence (IEEE-style decode lives in `page.js`, not
`tensor.js`), drawn with `render`/`ctx`. Headless hooks: `?step=N`/`?play=1`,
`?flip=i` (toggle bit index i; comma-separated, e.g. `?flip=0,5,9`), and
`?hover=x,y` (canvas-space; bit or field-label tooltip). Source: [`page.js`](page.js).
