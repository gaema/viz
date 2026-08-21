# diffusion-noise -- what a diffusion model is TRAINED to do

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **training objective** of a diffusion / flow model --
forward corruption under a schedule, and the regression the network is asked to
perform on the corrupted sample. **Anchor**: the generative-model family (DiT /
diffusion), alongside the emerging-architecture pages.

This page is **training only**. Integrating the learned field into a picture --
step counts, solvers, samplers -- is the **diffusion-sampler** page. Conditioning
and guidance scale are the **guidance** page. Both are named on screen
neither is built here.

## What it shows

One training step, drawn end to end:

```
x_t = a(t)·x₀ + s(t)·ε corrupt a clean sample by a known amount
ε ~ N(0, I) fresh noise every step
loss = ‖ target(x₀, ε, t) − net(x_t, t) ‖²
```

Two interpolants, side by side, switched by the **interpolant** control (both
equations stay on screen so the pair is always visible):

| | interpolant | coefficients | usual target |
|---|---|---|---|
| 🟢 | **diffusion** (DDPM) | `a = √ᾱ_t`, `s = √(1−ᾱ_t)`, so `a² + s² = 1` | `ε` |
| 🟢 | **rectified flow** | `a = 1−t`, `s = t`, a straight line from data to noise | velocity `ε − x₀` |

Rectified flow is what production image and video models train, so it is a
**mode of this page, not a sequel to it**: a reader who only ever sees `ᾱ` will
not recognise the code they meet in the wild. Flow matching's Gaussian
probability paths subsume the diffusion paths as special cases -- see the
sources below.

Six panels across the top, all 16×16 and all live:

| panel | what it is |
|---|---|
| `x₀` | the clean training sample (a template plus per-pixel spread) |
| `ε` | this step's noise draw |
| `x_t` | the corrupted sample -- the only image the network ever sees |
| `x̂₀` | **the model's current guess of the clean sample** |
| `ε̂` / `v̂` / `x̂₀` | the prediction, in whichever target you selected |
| `\|x̂₀ − x₀\|` | what the guess lost |

**The fourth panel is the point of the page.** At high noise the model's guess of
the clean image is the blurry average of everything it was trained on -- it
cannot tell which sample it is looking at, and the posterior bar under the panel
shows the weights nearly uniform. As `t` falls, the posterior collapses onto one
sample, and texture returns. Scrub the transport and watch that transition; it
is the single clearest picture of what the network has actually learned.

### Where `x̂₀` comes from (no weights are downloaded)

The toy data distribution is deliberately simple enough to be solved exactly:
pick one of `K` synthetic templates, then add per-pixel spread `sd`. Both stages
are Gaussian, so the Bayes-optimal denoiser -- the function a trained network
approximates -- is available in closed form and is computed in-page, per frame:

```
w_k ∝ exp(−‖x_t − a·μ_k‖² / 2(a²sd² + s²)) which template is this?
E[x₀ | x_t, k] = (a·sd²·x_t + s²·μ_k) / (a²sd² + s²) what did it look like?
x̂₀ = Σ_k w_k · E[x₀ | x_t, k]
```

The second line *is* the blur-to-texture effect in one expression: at high noise
`s²` dominates and the estimate is the template mean; at low noise the `x_t`
term dominates and the un-templated detail comes back. Every number the page
prints is this arithmetic, so hovering a pixel shows a true derivation rather
than a caption.

The **data spread** slider is load-bearing, not decoration. Set it near zero
the training set becomes exactly memorisable: the optimal denoiser is then
perfect at every noise level, the loss is zero everywhere, and the whole
`ε`-vs-`x₀` trade below disappears. Real data always carries detail no label
predicts, and that is what makes the objective non-trivial.

## The trade the page is built around

Pick the **predict** control and watch the loss curve move:

| target | loss at HIGH noise | image-space cost of a fixed error |
|---|---|---|
| `ε` | ~0 -- `x_t` is almost all `ε`, so the network can nearly copy its input | `s/a = 1/√SNR` -- **blows up** |
| `v` / velocity | moderate | `s` |
| `x₀` | large -- the clean sample is genuinely unknowable | `1` -- constant |

So `ε`-prediction is **easiest exactly where mistakes are most expensive**,
that inversion is not a detail of the parameterisation -- it is the reason a
loss weighting exists at all. The middle panel plots, versus `t`: the measured
loss in the chosen target space, the weighting `w(t)`, and their product. That
product is where the training signal actually lands, so **the weighting you pick
IS a decision about where image quality goes**. Try `uniform` against `SNR`
against `min-SNR-5` and watch the budget slide between the noisy and clean ends
of the axis.

And the reason any of this matters downstream: at inference the model never sees
clean data. Its own output becomes the next input, so an error at one noise
level is fed back in at the next -- which is the hook into the sampler page.

## Interactions

- **Transport** over 32 noise levels, autoplaying and looping; `?step=N` selects one.
- **Drag the schedule panel** left/right to set the noise level by hand -- every
 panel, the posterior, and the arithmetic card recompute under the cursor.
- **Drag a pixel of `x₀`** ↕ to edit that training sample's template, and watch
 the model's guess follow it (the model only knows what is in its training set).
- **Hover any pixel** of any panel for its value and the arithmetic that produced
 it -- `x_t = a·x₀ + s·ε` with the actual numbers, the posterior's template
 share, or `ε̂ = (x_t − a·x̂₀)/s`.
- **Controls**: interpolant, schedule shape, prediction target, loss weighting,
 training-set size, data spread, sample, noise seed.
- **A/B compare** puts the diffusion interpolant and rectified flow side by side.
- **Challenges**: drive the posterior into the blurry-mean regime; find a noise
 level where a fixed `ε`-error is amplified more than 3× in image space.

### Schedule shape

`linear` / `cosine` / `quadratic` set the `ᾱ` schedule in diffusion mode. In
rectified-flow mode the same control picks the **timestep shift** (`τ = shift·t /
(1 + (shift−1)·t)`, the reparameterisation production flow models use to spend
more or less of the axis at high noise): `linear → 1`, `cosine → 3`,
`quadratic → 0.5`. Same meaning either way -- how fast signal is destroyed as
`t` runs 0 → 1. The linear `ᾱ` schedule visibly destroys the signal by `t ≈ 0.7`,
which is exactly the complaint the cosine schedule was introduced to fix; the
`a`/`s` curves show it directly.

## URL hooks

Every handle is addressable, so any view is reproducible and screenshot-verifiable:

| param | effect |
|---|---|
| `?step=N` | seek the transport to noise-level bin `N` (0-31) and pause |
| `?t=0.42` | seek to the nearest bin to that `t` |
| `?mode=` | `diffusion` \| `flow` |
| `?sched=` | `linear` \| `cosine` \| `quadratic` |
| `?target=` | `eps` \| `v` \| `x0` |
| `?weight=` | `uniform` \| `snr` \| `minsnr` |
| `?K=`, `?sample=`, `?seed=`, `?spread=` | training-set size, which sample, noise draw, data spread |
| `?pix=N` | which pixel the arithmetic card reads |
| `?paint=i,v;i,v` | set template pixels directly (the headless stand-in for the `x₀` drag) |
| `?hover=x,y` | place the cursor (the headless stand-in for a hover) |
| `?play=1` | start the transport playing |
| `?compare=1`, `?ch=N`, `?theme=` | framework hooks (A/B compare, challenge, theme) |

## Sources

- Ho et al., *Denoising Diffusion Probabilistic Models* -- <https://arxiv.org/abs/2006.11239>
- Lipman et al., *Flow Matching for Generative Modeling* -- <https://arxiv.org/abs/2210.02747>
 (its Gaussian probability paths "subsume existing diffusion paths as specific instances")

Everything on the page is synthetic and computed in-page: no model weights are
fetched, and the page works offline over any static server.
