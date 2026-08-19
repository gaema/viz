# dtype-bits -- floating-point & integer bit layouts

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page, two views: **how a number is stored** in fp32 / fp16 / bf16 /
fp8 / int8 / int4 (and why fewer bits means more rounding error), and **which
silicon can actually multiply each one**. **Click any bit to flip it**
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

### Hardware support view

A **vendor x dtype matrix**: for every architecture, whether each dtype is a
`native` matrix-engine operand, `vector`/SIMD-only, `emulated` (upconverted to a
wider type first), `none`, or `unknown` -- the last being a real answer, not a
gap, because closed NPU IP often publishes no operand list. Filter by engine
class (GPU matrix engines / GPU vector ALUs / NPUs / audio DSPs / CPU SIMD),
drag to scroll, hover a cell for the level, the one-line why, and the public
vendor document it was read from; click a column header to focus that dtype and
see its tally across the visible rows.

The point it makes that a bit-layout diagram cannot: **a dtype is only cheap
where the silicon has an engine for it.** int8 is nearly universal, bf16 splits
the field, fp8 is a recent GPU/CPU-tile arrival, and int4 -- the format half the
open-weights world ships in -- is a native operand on strikingly few engines,
which is why int4 weights are usually unpacked before they are multiplied.

Data: [`../data/dtype-support.json`](../data/dtype-support.json) (schema +
the public-source rule in `../data/README.md`), fetched at
runtime. Support facts only -- this page carries no performance number, by
design. If the fetch fails (opened over `file://`), the view says so instead of
inventing rows.

## Render tier

T1 (Canvas2D). Bit cells + bars; no GPU needed.

## Wiring

`layout.mount()` + controls (`value`, `dtype`) + a step `Transport` over a
page-built bit-reveal sequence (IEEE-style decode lives in `page.js`, not
`tensor.js`), drawn with `render`/`ctx`. Headless hooks: `?step=N`/`?play=1`,
`?flip=i` (toggle bit index i; comma-separated, e.g. `?flip=0,5,9`),
`?hover=x,y` (canvas-space; bit or field-label tooltip), and for the hardware
view `?view=hardware|bits`, `?engines=<class>`, `?dtype=<key>`, `?hwscroll=N`
(the headless stand-in for dragging the list). Source: [`page.js`](page.js).
