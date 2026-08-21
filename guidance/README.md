# guidance -- classifier-free guidance: two predictions, then extrapolate

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page for the one idea that makes conditional diffusion models follow
a prompt: run the denoiser **twice** per step -- once with the conditioning
once with it dropped -- and combine the two predictions as

```
eps~ = eps_u + w * (eps_c - eps_u)
```

**Companion pages:** how the noise is added and what the network is trained to
predict is `diffusion-noise`; the integrator and the step count are
`diffusion-sampler`. This page owns the conditioning and its amplification only,
and holds the step count fixed (24 deterministic DDIM steps) so the sampler is
not a competing variable.

## The point

Read the formula as a vector, not as a blend. It **starts** at the unconditional
prediction and travels `w` times the difference `Δ = eps_c − eps_u`:

| `w` | where `eps~` lands | what it means |
|---|---|---|
| 0 | on `eps_u` | the prompt is ignored entirely |
| between 0 and 1 | between the two | interpolation -- adherence rises with `w` |
| 1 | exactly on `eps_c` | the conditional prediction, nothing added |
| above 1 | **past** `eps_c`, on the same ray | **extrapolation** -- there is no upper bound |

Everything above `w = 1` is off the end of the segment. Nothing in the arithmetic
stops it at the plausible answer, which is why a large `w` does not merely sharpen
an image -- it over-shoots into blown-out colour and a stiff, stereotyped
composition. The page draws `Δ` as a dashed segment from the tip of `eps_u` to the
tip of `eps_c`, and the amplified copy `w·Δ` continuing along the same ray, with a
tick marking where `w = 1` stops. Past a moderate `w` the `eps~` ray simply runs
off the panel, labelled with its length as a percent of `|eps_c|` -- drawn true to
scale rather than squashed to fit, because the running-off *is* the lesson.

## What is on screen

- **The map** -- a 2D toy distribution: 4 prompt classes (A-D), each of which is
 itself a **2-component** mixture, so a class has internal variety to lose. The
 three predictions are drawn at a **draggable probe point**; drag it and watch
 `Δ` shrink to nothing far from the data and grow as the noise level falls.
- **The diversity strip** -- 12 samples run twice from the **same 12 seeds**: once
 at `w = 1` (the control) and once at your `w`. Each swatch is labelled with the
 sub-mode it landed on and coloured by how *stereotypical* it is (blaring =
 sat down on the sub-mode centre). Pairing the seeds is what makes the
 difference attributable to `w` rather than to luck. Mode collapse is something
 you *see* -- a row that goes uniform -- not something the page asserts.
- **The timeline** -- guidance restricted to an interval of steps, the modern
 refinement ([Kynkäänniemi et al., arXiv:2404.07724](https://arxiv.org/abs/2404.07724),
 who found that limiting guidance to a **middle interval** of the sampling
 trajectory is what improved image quality in practice). Drag either end. In
 this toy the noisy early steps are the ones that decide *which* mode you land
 in, so shrinking the interval hands diversity back, and trimming the quiet tail
 costs nothing at all -- which is also the compute dial, below.

## The trade, and the numbers that show it

Every guided step costs a **second forward pass**: 24 guided steps = 48 passes =
**200% of the 1-pass-per-step baseline**. The readout prints the live pass count
for whatever interval is selected.

What that buys, and what it spends, is measured live in the readout for both arms:
on-prompt fraction, how many of the class's sub-modes are still populated, the
mean pairwise spread (as a percent of the control's), and *typicality* -- mean
distance from the nearest sub-mode centre in units of the data's own standard
deviation. Raise `w` from the default and read them: adherence climbs and then
**saturates**, while spread keeps falling and typicality starts climbing again as
the update over-shoots the mode. Past the saturation point, `w` is buying nothing
and still charging.

**High `w` does not make the output better. It makes it more stereotypically the
prompt.**

## Why the toy is honest about `w > 1`

The data is a Gaussian mixture, so the denoiser's eps-prediction is available in
**closed form** -- every vector and every number on screen is computed, not
staged. That has one consequence worth stating: an *exact* conditional model
already lands on the right class at `w = 1`, so extrapolating past it could only
cost diversity, never buy adherence.

Real conditional models are not exact -- they under-fit the conditioning, which is
the actual reason `w > 1` earns its keep. The page models that with a
**conditioning leak λ**: the fraction of probability mass the "conditional" model
still assigns to the other classes. At `λ = 0` it is the exact conditional score
and `w > 1` buys no adherence at all; raise λ and adherence at `w = 1` falls,
and `w > 1` measurably rescues it. Both regimes are one slider apart.

## Render tier

T1 (Canvas2D). Everything is vectors, dots, swatches and a timeline; there is no
large grid to justify WebGL2.

## Wiring

`layout.mount` + controls (`w`, `cls`, `leak`, `gStart`, `gEnd`, `seed`) + a
24-step transport (autoplay, loop) + `onPointer` for the probe drag and the two
interval handles + hover-to-inspect on the three vector tips, the probe, every
swatch and every timeline step. A/B compare is wired to `w` (1 vs 12).

**URL hooks** (all of `w`, `cls`, `leak`, `gStart`, `gEnd`, `seed`, `step` are
mirrored into the query string, so the copy-link button reproduces the exact
frame): `?px=`/`?py=` place the probe (the headless stand-in for dragging it),
`?hover=x,y` fakes the cursor, `?compare=1` opens the A/B panes, `?ch=N` opens a
challenge. Source: [`page.js`](page.js).

## Source

- Ho & Salimans, *Classifier-Free Diffusion Guidance* -- <https://arxiv.org/abs/2207.12598>
- Kynkäänniemi et al., *Applying Guidance in a Limited Interval Improves
 Sample and Distribution Quality in Diffusion Models* -- <https://arxiv.org/abs/2404.07724>
