# few-step-distillation -- changing the model so four steps are enough

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a fifty-step sampler becomes a **one-to-four-step** one --
by training a **different model**, not by sampling the same one more coarsely --
and the specific thing that costs. **Anchor**: few-step generative sampling
(consistency distillation; adversarial / distribution-matching distillation).

## Scope -- and how this differs from `diffusion-sampler`

[`diffusion-sampler`](../diffusion-sampler/README.md) owns step count as a
**discretisation dial on a fixed model**: the same learned field, integrated with
fewer, larger steps, so the whole story is integration error. This page is the
other lever entirely -- **the model is changed** so that a handful of steps is
enough. Nothing here is a coarser integration of the teacher; the student is a
separate map, fitted from the teacher, with its own failure mode.

Read them in that order if you can. The training objective and the noising
schedule are [`diffusion-noise`](../diffusion-noise/README.md); conditioning
guidance scale are [`guidance`](../guidance/README.md).

## What is on screen

| Element | What it is |
|---|---|
| left panel | the **teacher**: N small Euler steps of the exact field, revealed by the transport |
| faint violet arrows | the velocity field `u(x, t)` at the current `t` |
| green dots | the data manifold (the toy mixture's modes) |
| right panel | the **student**: 1..8 long jumps from the same start, same seed |
| hollow green rings (right) | where the teacher's many steps ended -- the student's jump should land there |
| dotted grey line (right) | the residual between the student's answer and the teacher's |
| dashed gold ties (consistency mode) | the **training signal**: several points of one trajectory, each tied to that trajectory's single shared endpoint |
| red rings + arrows (adversarial mode) | the **training signal**: the raw regression output, and the correction that puts it back on the data manifold |
| two rails in the right panel | **drag them** -- jump count `K` (1..8) and distillation strength |
| bottom strip | the seed sweep: everything held fixed, one column per noise sample, teacher row above, student row below |

## The math it runs

Nothing on the page is a neural network and nothing is faked. The toy
distribution is two moons as an equal-weight mixture of narrow Gaussians, so the
probability-flow velocity is closed form and the teacher trajectory is **exact**
-- which is what makes the student's error a measurement rather than a drawing.

```
interpolant x_t = α(t)·x_data + σ(t)·x_noise, α = sin(½πt), σ = cos(½πt)
E[x_data|x_t] = (1 − c·α)·M(x) + c·x, c = α·var/τ², τ² = α²var + σ²
u(x, t) = α'·E1 + (σ'/σ)·(x − α·E1)
teacher step x ← x + u·h, h = 1/N (N of these)
```

The **student is fitted in the page**, by distillation from that teacher:

```
1. draw M noise samples z_j, run the teacher to convergence: y_j
2. f(x, t) = Σ_j w_j·y_j / Σ_j w_j
 w_j = exp( −|x − (α(t)·y_j + σ(t)·z_j)|² / 2h(t)² )
 h(t) = h₀·(σ(t) + 0.06)
3. K-step sampling: x̂ ← f(x, t_k); re-noise x ← α(t_{k+1})·x̂ + σ(t_{k+1})·z; repeat
```

Two things are worth stating plainly about step 2, because they are what make
the page measure the cost instead of asserting it:

- The regression targets are indexed by **where the trajectory is at time `t`**.
 That *is* the consistency training signal -- every point on one trajectory must
 map to that trajectory's endpoint -- and it is what the gold tie-lines draw.
- The bandwidth is the student's **capacity**. It is scaled by the noise level
 `σ(t)` because that is what sets how hard the jump is: from pure noise the
 student must guess the whole endpoint, while from a mostly-denoised point it
 barely has to move. The first version of this page used a bandwidth fixed in
 `x`-space, and the page's own spread number then said more jumps made diversity
 *worse* -- which is not what shipped few-step samplers do. The noise-scaled
 form is both the physically right one and the one whose measurements behave.

## The cost, measured live

Hold everything fixed and **sweep only the seed**. That is the bottom strip: one
column per noise sample, the teacher's answer above and the student's below, each
cell tinted by the sample's own position so a repeated sample is a repeated
colour. Four numbers, recomputed every frame:

| Metric | Meaning |
|---|---|
| `spread` | RMS distance of the sample set from its own centroid, reported as a **percent of the teacher's** (lower = LESS variety; 100% = teacher parity) |
| `repeats` | share of samples that have a near-identical twin elsewhere in the row (within 0.08) |
| `modes` | how many distinct data modes the 16 samples reach, student vs teacher |
| `ESS` | effective sample size -- how many teacher endpoints each student answer is a weighted blend of. This is the mechanism: a low-capacity student averages, and an average of many endpoints is the same answer whatever the seed |

Alongside them, the accuracy axis: mean `|student − teacher|` over the swept
seeds, and mean off-manifold distance for both (its floor is the data's own mode
width, 0.06, not zero).

**The jump rail carries a paired control.** Whenever `K > 1` the page re-runs the
*same fitted student on the same seeds at one jump* and prints both readings, so
"more jumps helps" is never a slogan here -- a re-noise puts randomness back
then the averaging is applied again, and which one wins depends on the strength.
Read the two numbers.

**The mode switch is a real trade, not a free win.** The adversarial /
distribution-matching term pulls each raw output most of the way to the nearest
data mode -- the stand-in for a discriminator (ADD) or a distribution-matching
loss (DMD2). It is aimed at off-manifold error and it moves that number hard.
What it does to the *diversity* numbers is measured, not claimed: switch the
training signal back and forth and compare the spread percent and the repeat
count.

## Interactions

| Interaction | How |
|---|---|
| step the run | transport play / pause / step / scrub; auto-plays and loops. The teacher advances one Euler step at a time; the student's jumps are revealed alongside |
| **drag the student's step count** | grab the `jumps K` rail in the right panel (1..8) -- or the matching slider |
| **drag the distillation strength** | grab the `strength` rail in the right panel; the kernel widens under your hand and the strip drains |
| **drag a starting point** | grab a hollow start marker in the teacher panel: that trajectory, its converged teacher endpoint, and the student's jumps from it all recompute |
| switch the training signal | consistency (self-consistency ties) vs adversarial / distribution-matching (manifold correction arrows) -- the drawing changes with the objective |
| hover for arithmetic | a teacher particle gives `x`, `u(x,t)`, `h` and `x + u·h`; a student jump gives the top kernel weights, the ESS, the raw output, the manifold correction and the re-noise; a strip cell gives the sample, the other model's sample from the same noise, and whether it is a repeat |
| resize the problem | teacher step count, trajectories shown, seed |

### URL hooks

Every control is restorable from the query string, and the re-noise draws are
seeded per `(seed, sample, jump)`, so one URL replays one exact run.

```
?step=3 pause the transport at teacher step 3
?K=4 student jump count
?bw=0.6 distillation strength (0 = high capacity, 1 = one answer for all)
?mode=adversarial training signal: consistency | adversarial
?tsteps=50&seed=4 teacher step count, and which noise samples
?P=6 trajectories drawn in the panels
?signal=0 hide the training-signal drawing
?drag=0,1.9,-1.4 move trajectory 0's start (headless stand-in for a drag)
?hover=560,470 place the cursor (headless stand-in for a hover)
```

## Sources

No number from any paper is reproduced on the page; everything on screen is
measured in-page on the toy. These are cited for the claims the toy cannot
measure -- what the shipped families actually train against.

- Song, Dhariwal, Chen, Sutskever, *Consistency Models* --
 <https://arxiv.org/abs/2303.01469>. Self-consistency as the training signal,
 and multistep consistency sampling (jump, re-noise, jump).
- Luo et al., *Latent Consistency Models* -- <https://arxiv.org/abs/2310.04378>.
 The same idea in a latent space, which is how it reached image models people
 run.
- Sauer, Lorenz, Blattmann, Rombach, *Adversarial Diffusion Distillation* --
 <https://arxiv.org/abs/2311.17042>. The discriminator term, and the one-to-four
 step regime this page is named for.
- Yin et al., *Improved Distribution Matching Distillation* (DMD2) --
 <https://arxiv.org/abs/2405.14867>. Distribution matching in place of a pure
 regression, and the diversity problem it is aimed at.
