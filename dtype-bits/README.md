# dtype-bits -- the numeric zoo, bit by bit

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page, three views: **how a number is stored** (23 formats -- 10
scalar floats, 6 block-scaled, 7 integers), **what a block-scaled format
actually costs** (the family every "4-bit" model ships in), and **which silicon
can multiply each one**. **Click any bit to flip it** and watch the reconstructed
value rebuild bit by bit; hover a bit for its field + place value, or a field
label to decode it. **Anchor**: A11 special pre-quant transforms / precision
(Family A3 foundational).

## The dtype table

Every column of the hardware matrix and every row of the comparison table is one
of these keys. Bit counts are what the format *costs*, which for a block format
is amortised over its block.

| Family | key | bits | layout |
|---|---|---|---|
| float | `fp64` | 64 | E11M52, bias 1023, IEEE binary64 |
| float | `fp32` | 32 | E8M23, bias 127, IEEE binary32 |
| float | `tf32` | **19 in a 32-bit container** | E8M10, bias 127 -- fp32 range, fp16 precision |
| float | `fp16` | 16 | E5M10, bias 15, IEEE binary16 |
| float | `bf16` | 16 | E8M7, bias 127 -- fp32's exponent, truncated mantissa |
| float | `fp8e4m3` | 8 | E4M3, bias 7 -- **no Inf**; only mantissa-all-ones at max exponent is NaN, so max finite is 448 |
| float | `fp8e5m2` | 8 | E5M2, bias 15 -- IEEE-shaped, has Inf/NaN, max finite 57344 |
| float | `fp6e3m2` | 6 | E3M2, bias 3 -- **no Inf, no NaN**, max finite 28 |
| float | `fp6e2m3` | 6 | E2M3, bias 1 -- **no Inf, no NaN**, max finite 7.5 |
| float | `fp4e2m1` | 4 | E2M1, bias 1 -- **no Inf, no NaN**, 16 codes total, max finite 6 |
| block | `mxfp8` | **8.25** | 32 × fp8 e4m3 + one E8M0 scale (OCP MX v1.0) |
| block | `mxfp6` | **6.25** | 32 × fp6 e3m2 + one E8M0 scale |
| block | `mxfp4` | **4.25** | 32 × fp4 e2m1 + one E8M0 scale |
| block | `nvfp4` | **4.5** | 16 × fp4 e2m1 + one fp8 **e4m3** scale (not a power of two) |
| block | `bfp8` | **8.5** | 16 × (sign + 7-bit magnitude) + one shared 8-bit exponent |
| block | `bfp4` | **4.5** | 16 × (sign + 3-bit magnitude) + one shared 8-bit exponent |
| int | `int32` `int16` `int8` `int4` `int2` | 32/16/8/4/2 | two's complement, × a per-tensor scale |
| int | `uint8` | 8 | unsigned, × scale with a **zero-point** (128 here) |
| int | `int1` | 1 | the bit *is* the sign: 0 → +scale, 1 → −scale |

All integer rows share one absmax, so the comparison is about **code count**,
not about who got a friendlier scale.

## What each view shows

### Bit layout

The focused dtype's bits as **sign | exponent | mantissa** cells (integers:
**sign | magnitude**), coloured by field. Scrub the transport to **reveal bits
left→right and rebuild the value**: the sign sets ±, the exponent field sets the
scale `2^(e−bias)` (with the implicit leading 1), and each mantissa bit adds
`scale·2⁻ⁱ`. A comparison table encodes the same chosen value in every visible
dtype and shows total bits, decoded value, and **rounding error** as a
**log-scaled** bar -- with fp64 and int1 in one table a linear bar renders every
useful row as a single pixel.

Two formats are drawn specially, because their bit strip alone would mislead:

- **`tf32` is not a memory format.** Its 19 meaningful bits are drawn, then the
 **13 unused container bits** of the 32-bit register they ride in, dimmed
 not clickable, with an on-screen line saying so. Nothing stores tf32; it is
 what the tensor core ingests from fp32.
- **A block dtype has no single bit strip.** Focusing one here draws its
 *element* format's strip with a banner naming the real amortised cost
 pointing at the block view.

fp64's 64 cells do fit on one strip -- the cell width shrinks with the format's
width (54 px down to a 6 px floor) and the digit is dropped below 7 px rather
than the row being wrapped, which would break the single-row hit-test the
click-to-flip interaction depends on.

### Block-scaled

The view the low-bit era needs. An MXFP4 value **is not 4 bits**: it is 4 bits of
element plus a shared 8-bit exponent across a block of 32, i.e. **4.25 bits per
element amortised**. The view draws the block itself:

- a **bits-accounting bar** -- `N × elemBits` of elements against the one shared
 scale, ending in `total / N = bits per element`;
- the **shared scale** with its own 8 stored bits and what they mean (E8M0 is
 exponent-only: 8 bits, no mantissa, so the scale can only be a power of two --
 NVFP4's e4m3 scale can land anywhere);
- the **N elements**, each with its stored bit pattern, its reconstructed value
 and the original it was fitted to, coloured by signed magnitude;
- the reconstruction rule, `value[i] = scale × element[i]` (BFP: `(−1)^s · m[i] ·
 2^(sharedExp − mantBits + 1)` -- there is no per-element exponent at all).

The value slider is **element 0** of the block, so the number you are holding
actually rides through the block and pays its rounding. Hover any element for
the full reconstruction and its error; hover the scale for the block accounting.

### Hardware support

A **vendor × dtype matrix**: for every architecture, whether each dtype is a
`native` matrix-engine operand, `vector`/SIMD-only, `emulated` (upconverted to a
wider type first), `none`, or `unknown` -- the last being a real answer, not a
gap, because a closed accelerator often publishes no operand list. A dtype key the
catalogue does not carry **at all** also renders `unknown`; the column set is
owned by the page and the capability facts by the JSON, and neither being ahead
of the other may blank a cell or throw.

23 columns do not fit legibly at once, so there are two filters: **column
family** (all / scalar floats / block-scaled / integers) and **engine class**
(GPU matrix engines / GPU vector ALUs / NPUs / CPU SIMD). Below
~44 px of column width the header keys turn on their side rather than ellipsize
into uselessness. Drag to scroll, hover a cell for the level, the one-line why,
and the public artifact it was read from; click a column header to focus
that dtype and see its tally across the visible rows.

The point it makes that a bit-layout diagram cannot: **a dtype is only cheap
where the silicon has an engine for it.** int8 is nearly universal, bf16 splits
the field, fp8 is a recent GPU/CPU-tile arrival, and the block-scaled formats --
the ones half the open-weights world now ships in -- are native on strikingly few
engines, which is why they are usually unpacked before they are multiplied.

Three things the GPU rows show that a "newer is wider" reading would get
backwards:

- **A vendor can DROP an operand.** The 4-bit and 1-bit INTEGER tensor operands
 are hardware on NVIDIA's Ampere and Ada rows and `emulated` on the Blackwell
 row: the PTX still assembles, but it lowers to a helper that widens to int8.
 The 4-bit operand that generation actually added is FLOAT.
- **One marketing name can be two silicon designs.** `Xe-LPG` gets two rows,
 because the Meteor Lake iGPU has no matrix engine at all while the Arrow Lake
 one has an Alchemist-lineage array issued per sub-group of 8. A probe written
 for the 16-wide form finds nothing on either and concludes wrongly about both.
- **`emulated` is where the mobile GPUs live.** On the Mali and Adreno rows every
 sub-byte integer and every narrow float is `emulated`, because the narrowest
 arithmetic instruction on those parts is an 8-bit dot product. Those formats
 still pay -- they cut weight bytes, and mobile decode is bandwidth-bound -- but
 they buy zero arithmetic, which is the distinction the level is there to make.

Data: [`../data/dtype-support.json`](../data/dtype-support.json) (schema
the public-source rule in `../data/README.md`), fetched at
runtime. Support facts only -- this page carries no performance number, by
design. If the fetch fails (opened over `file://`), the view says so instead of
inventing rows.

## Numerical honesty

The decode is not decorative, so it is cross-checked against implementations
that share no code with it: Python `struct` for fp64/fp32/fp16, `ml_dtypes` for
bf16/fp8 e4m3/fp8 e5m2, a synthesised fp32→10-bit-mantissa reference for tf32,
and an **exhaustive nearest-code search over every code** for fp6/fp4. Two
things that fell out of doing it:

- **No `1 << M` shifts.** JS bitwise operators coerce to int32, so `1 << 52` is
 `1 << 20` and fp64 decoded to nonsense. Every field is extracted with
 `Math.pow(2, k)` arithmetic, exact for the integers involved (a 52-bit
 mantissa is < 2^53) and correct for int32's full-width sign bit too.
- **Rounding is ties-to-EVEN**, which is what every FP unit and every
 fp32→fp8/fp6/fp4 converter implements. `Math.round` is ties-away, and it
 disagrees on exactly the values that sit midway between two codes -- 2.5 into
 fp4 e2m1 is 2.0 under RN-even and 3.0 under `Math.round`, and a low-mantissa
 format is midway between codes constantly.

## Render tier

T1 (Canvas2D). Bit cells, block grid and bars; no GPU needed.

## Wiring

`layout.mount` + controls (`view`, `dtype`, `cols`, `engines`, `value`) + a
step `Transport` over a page-built sequence -- bit reveal, block reveal (scale
then each element), or matrix row reveal, depending on the view. IEEE-style
decode lives in `page.js`, not `tensor.js`. Headless hooks: `?step=N`/`?play=1`,
`?flip=i` (toggle bit index i; comma-separated, e.g. `?flip=0,5,9`),
`?hover=x,y` (canvas-space), `?view=bits|block|hardware`, `?dtype=<key>`,
`?cols=all|float|block|int`, `?engines=<class>`, `?hwscroll=N` (the headless
stand-in for dragging the list). `window.__dtypeNum` exposes the numeric core so
the cross-check above can be run against the real functions. Source:
[`page.js`](page.js).
