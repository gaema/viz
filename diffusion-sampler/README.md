# diffusion-sampler -- integrating a learned field

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: what a diffusion / flow-matching model does at **inference**
time -- integrating a learned velocity field from a noise sample to a data
sample -- and why the **step count is a discretisation dial**, not a quality
setting. **Anchor**: generative sampling (diffusion / flow matching).

**Scope.** This page is the *sampler* only. The training objective and the
noising schedule are the [`diffusion-noise`](../diffusion-noise/README.md) page;
conditioning and guidance scale are the [`guidance`](../guidance/README.md)
page.

## What is on screen

A 2D toy distribution -- two moons, as an equal-weight mixture of narrow
Gaussians -- with:

| Element | What it is |
|---|---|
| faint green dots | the data manifold (the mixture's modes) |
| violet arrows | the velocity field `u(x, t)` **at the current t**, so the field visibly turns as the run proceeds |
| solid blue lines | the trajectory each particle actually walks, revealed step by step |
| hollow circles | each particle's start (its noise sample) -- **drag one** |
| dashed gold curve | the *conditional* path `x_t = α(t)·x_end + σ(t)·x_start` -- what the interpolant toggle straightens |
| right-hand panels | the **same seed and the same field** integrated at 4 / 10 / 50 steps |

Nothing is a neural network and nothing is faked. For a Gaussian mixture the
probability-flow velocity is closed form, so every arrow, every particle update
and every number in the readout is arithmetic the page just did.

## The math it runs

With the interpolant `x_t = α(t)·x_data + σ(t)·x_noise`, `t: 0 → 1`:

```
diffusion path α = sin(½πt) σ = cos(½πt) curved, schedule-shaped
rectified flow α = t σ = 1 − t straight conditional path

E[x_data | x_t = x] = (1 − c·α)·M(x) + c·x c = α·var / τ², τ² = α²var + σ²
u(x, t) = α'·E1 + (σ'/σ)·(x − α·E1) the probability-flow velocity
∇ log p_t(x) = −(x − α·E1) / σ² the score

ODE step x ← x + u·h h = 1/N
SDE step x ← x + (u + ½g²·∇log p_t)·h + g·√h·z g = churn·σ
```

`M(x)` is the responsibility-weighted mean of the mixture modes -- a softmax over
squared distances, which is why the field points at whichever moon you are
nearest and blends between them where you are not.

The SDE drift carries the `½g²·score` correction, so **both samplers have the
same marginals**. The toggle therefore isolates exactly one variable: whether
noise is re-injected along the way.

## The trade the page is built to show

Three measured numbers, recomputed live:

| Metric | Meaning |
|---|---|
| `off` | mean distance from an endpoint to the nearest data mode -- how far off the manifold the run landed. Its floor is the data's own mode width (0.06), not zero |
| `Δref` | mean distance from an endpoint to the same particle's endpoint under a 200-step deterministic reference |
| `path/chord` | mean trajectory length ÷ start-to-end distance; `1.00` is a straight line |

Read across the 4 / 10 / 50 ladder:

- **Fewer steps = larger integration error.** It shows up in `off` first --
 points drifting *near* the manifold rather than onto it. In a real image model
 that is lost fine texture, not an obviously broken picture, which is why
 low-step output can look fine at a glance and thin under inspection.
- **Deterministic (ODE)**: `Δref` shrinks steadily with the step count -- the
 same seed always produces the same endpoint, and the map is invertible. The
 cost is diversity: one noise sample maps to exactly one output, so per prompt
 you get less variety.
- **Stochastic (SDE)**: the score term pushes a step that lands off-manifold back
 onto the data, so few-step runs stay on-distribution -- it *self-corrects*
 accumulated discretisation error. But `Δref` stays large no matter the step
 count, because the sample is no longer pinned by the seed. Reproducibility is
 what you pay.

## On "straighter ⇒ fewer steps"

The page separates two things the folk version of this claim conflates:

1. The **conditional** path (a specific noise sample to a specific datum) *is*
 exactly a straight segment under the linear interpolant. That is the dashed
 ghost, and it is a definition, not a result.
2. The **marginal** trajectory the sampler actually integrates is a different
 object. On this toy, switching to the linear interpolant does **not**
 straighten it -- the page prints `path/chord` for both so you can check
 rather than take it on faith. Genuine straightening of the marginal is what
 reflow buys.

And whether straightness is what does the work at all is disputed:
<https://arxiv.org/abs/2410.07303> argues it is not.

## Interactions

| Interaction | How |
|---|---|
| step the integration | transport play / pause / step / scrub; auto-plays and loops |
| re-integrate under your hand | **drag** a hollow start marker -- that particle's whole trajectory, and its reference endpoint, recompute live |
| inspect the field | **hover** an arrow for `u`, `E[data\|x]`, the score, and how far one step of `h` moves; hover a particle for its position and the full update arithmetic; hover a ladder panel for its metrics |
| change the path | interpolant select: diffusion (curved) vs rectified flow (straight conditional path) -- same field formula, same integrator, same particles |
| change the sampler | ODE / SDE toggle + churn slider (`g = churn·σ`) |
| resize the problem | step count, particle count, seed |

### URL hooks

Every control is restorable from the query string, so one URL replays one exact
run -- the randomness is seeded per `(seed, particle, step)`.

```
?step=3 pause the transport at integration step 3
?steps=50&path=linear 50 steps on the rectified-flow interpolant
?sde=1&churn=1.2 stochastic sampler at a given noise level
?seed=7&P=16 which noise samples, and how many
?drag=0,1.4,-1.1 move particle 0's start (headless stand-in for a drag)
?hover=300,220 place the cursor (headless stand-in for a hover)
?guide=0 hide the dashed conditional paths
```

## Sources

- Karras, Aittala, Aila, Laine, *Elucidating the Design Space of Diffusion-Based
 Generative Models* -- <https://arxiv.org/abs/2206.00364>. The separation this
 page's controls are built on: schedule, scaling and sampler are orthogonal
 knobs, not one "quality" setting.
- Liu, Gong, Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data
 with Rectified Flow* -- <https://arxiv.org/abs/2209.03003>.
- *Rectified Diffusion* -- <https://arxiv.org/abs/2410.07303>, on why
 straightness is not what does the work.
