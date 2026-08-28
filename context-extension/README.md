# context-extension -- running a rotary model past its trained context length

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: what actually breaks when a model trained with rotary position
embedding is asked for positions beyond the length it was trained on, and how
the five standard responses differ -- shown as a picture rather than a table.

## Complements the `rope` page -- it does not re-teach it

[`../rope/`](../rope/README.md) owns the **rotation mechanism**: a `d`-dim
vector splits into `d/2` pairs, and at position `p` pair `i` turns by
`Δ = p·θᵢ` with `θᵢ = base^(−2i/d)`. That page is where you learn what the
rotation *is*, and this page assumes you already have it -- it never redraws the
rotation planes.

This page starts one level up, from the same `θᵢ` (both pages compute it the
same way, so their numbers agree): each pair has a **wavelength**
`λᵢ = 2π/θᵢ`, and the trained context `L` covers `L/λᵢ` turns of it. A pair that
completes more than a turn inside `L` has shown the model every angle it can
produce. A pair whose wavelength is *longer* than `L` has only ever been seen
over a slice `[0, (L−1)·θᵢ]` of its circle -- so a position past `L` asks it for
an angle that was never in the training data. Everything else on the page
follows from that one asymmetry.

## What it shows

The main view is the **per-pair wavelength spectrum**: one column per dimension
pair, wavelength on a log axis, with a dashed line at the trained context `L`
a dotted line at the current position. Each column carries a hollow dot at the
un-extended `λᵢ`, a filled dot at the rescaled `λ′ᵢ`, and a stem between them --
so *how far the method moved that pair* is a length you can see. The filled dot
is green when the pair's angle at the current position was seen in training
red when it was not.

That makes the difference between the methods visual:

| Method | What it does to the spectrum |
|---|---|
| **naive extrapolation** | moves nothing; the long-wavelength end turns red the moment the position passes `L` |
| **[position interpolation](https://arxiv.org/abs/2306.15595)** | divides every position by `s`, so every column moves by the same amount -- nothing goes out of distribution, and every pair loses the same factor of fine detail |
| **NTK / frequency scaling** | rescales the base (`base·s^(d/(d−2))`), so pair 0 is untouched, the slowest pair is divided by exactly `s`, and the change ramps smoothly in between |
| **[YaRN](https://arxiv.org/abs/2309.00071)** | sorts pairs by wavelength and only moves the ones long enough to justify it (a ramp on turns-inside-`L`, fully interpolated below 1 turn, untouched above 32), plus an attention-temperature term `0.1·ln(s)+1` |
| **[LongRoPE](https://arxiv.org/abs/2402.13753)** | searches a factor *per dimension* instead of deriving one from a rule, so its factor strip is not smooth |

A strip under the spectrum plots the **per-pair rescale factor** `θᵢ/θ′ᵢ`
directly: flat at `s` for interpolation, a smooth ramp for NTK, plateau-ramp-
plateau for YaRN, a jagged staircase for LongRoPE. It is the shortest statement
of what separates them.

> The LongRoPE factors here are a **deterministic illustrative stand-in** with
> the two properties a search result has (non-uniform across dimensions,
> monotone non-decreasing with wavelength). Real factors are per-model search
> outputs; the page says so in its own readout rather than implying otherwise.

## The trade, computed in-page

Every method buys long-range coverage with short-range resolution, and the page
**measures** that rather than asserting it. The bottom-left chart sweeps the
extension factor from `1×` to `32×` and plots, for all five methods at once, the
**mean per-pair angular separation between adjacent positions** as a percent of
the un-extended model's (higher is better; `100%` = untouched). Position
interpolation traces exactly `100/s`; YaRN stays far above it because the fast
pairs it leaves alone keep their full separation; naive extrapolation is flat at
`100%` -- it pays nothing and buys nothing.

The panel beside it is the same five methods scored at the *current* setting:
pairs out of distribution, resolution kept, and the fastest pair's radians per
token. The readout adds the L2 form of the same measure, which is dominated by
pair 0 and therefore flatters YaRN -- both are shown so the reader can see why
the choice of measure matters.

## Interactions

- **Transport** steps the position from inside the trained range out to `s·L`,
 autoplaying and looping on load; the spectrum recolours as the frontier passes
 each pair's wavelength.
- **Drag the extension factor** anywhere on the trade chart -- the marker,
 the spectrum, the factor strip and the comparison panel all follow live.
- **Drag the trained-context line** (the dashed line on the spectrum) up
 down to change `L`; which pairs are "long" is redefined under your hand.
- **Drag the position track** at the top to place the position by hand.
- **Method switch** redraws the same spectrum, so the five responses are
 compared on one widget rather than four.
- **Hover any pair column** for its `θᵢ`, wavelength before and after, turns
 completed inside `L`, the arc of angles ever seen, its angle at the current
 position (raw and mod `2π`), and the in/out-of-distribution verdict. Hover the
 trade chart for every method's resolution at that extension factor.
- **A/B compare** renders position interpolation and YaRN stacked, same axes.

## Render tier

T1 (Canvas2D). The lesson is a few hundred marks and two line charts; nothing
here needs a GPU raster.

## Wiring

`layout.mount` + controls (`method`, `scale`, `ctx`, `d`, `base`) + a
position transport, drawn with `render.label`/`arrow` and the 2D context.
Frequencies come from the same `θᵢ = base^(−2i/d)` the `rope` page uses, so the
two agree. Headless hooks: `?step=N`, `?pos=N` (pin an exact position),
`?hover=x,y` (stand in for a cursor), `?play=1`, `?compare=1`, `?ch=N`, plus the
control keys the framework mirrors into the URL (`?method=yarn&scale=8&ctx=4096`).
Source: [`page.js`](page.js).
