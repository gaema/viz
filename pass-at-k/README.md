# pass-at-k -- sharpening buys the first sample and sells the tail

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: why reinforcement learning on a verifiable reward makes a model
**sharper** rather than **larger**, and why that makes the pass@k curves of the RL
model and its base model **cross**. **Anchor**: the reasoning-time companion to
[softmax](../softmax/README.md) (sharpening a distribution with a temperature
exponent is the same arithmetic) and [sampling](../sampling/README.md) (where
the truncation/renormalize machinery lives).

## What it shows

At k = 1 the RL model wins comfortably. Let both models draw more candidates
the **base** model overtakes it. The page's claim is about the mechanism, not
about anyone's leaderboard: RL concentrates probability mass onto reasoning paths
the base model could already produce, and it pays for that with the tail.

Two distributions over candidate answers are built in-page and everything else is
derived from them:

| symbol | what it is |
|---|---|
| `p` | base model's probability of each candidate answer -- **broad** |
| `q ∝ p^(1+α) · e^(β·reinforced)` | RL model's -- **peaked**; `α` sharpens, `β` reinforces |
| `reinforced` | correct **and** `p ≥ floor` -- RL only reinforces paths it actually sampled during training |
| `s` | per problem, `Σ p` (or `Σ q`) over the ✓ candidates = probability one draw is right |
| `pass@k` | `mean over problems of 1 − (1 − s)^k`, for k i.i.d. samples |

**Why a crossing is possible at all** is the part worth sitting with, and the page
is built so you can discover it rather than be told it. For **one** problem,
`1 − (1 − s)^k` is monotone in `s`: whichever model has the larger `s` wins at
**every** k, and the curves can never cross. The crossing lives in the **average
over problems**. Sharpening drives `s` up where the model's favourite answer
happens to be right, and drives it toward **0** where it is wrong -- and a problem
with `s = 0` is never solved, at any k, however many samples you spend. Coverage
is what the average is made of, and coverage is what gets sold.

The **discovery floor** is where the honesty of the trade lives: RL can only
reinforce a correct path it actually sampled often enough to see. Correct
candidates below the floor are marked with a **struck ✓** -- they were right, they
were never reinforced, and sharpening pushes their mass toward zero along with
everything else that is not the favourite.

Both regimes are reachable, so the lesson is found rather than staged:

| configuration | result |
|---|---|
| low `α`, floor at 0 | sharpening **helps at every k** -- nothing correct was pruned, no crossing anywhere |
| high `α`, floor raised | RL wins k = 1 by a wide margin and the base overtakes it within a handful of samples |
| `α = 0, β = 0` | the two models are the same distribution -- nothing bought, nothing traded |

The closing readout is live: **the k at which the curves cross**, how far ahead RL
is at k = 1, and how far behind it is at its worst. The chart also carries the
practical consequence -- **k = 1 is what ships**. With no verifier there is
nothing to pick *with*, so every point right of k = 1 assumes a judge good enough
to recognize the right answer among the k you drew. That assumption is what
best-of-N selection buys, and it is why this trade is usually worth taking even
though the curves cross.

## Where the empirical result comes from

The page teaches the **mechanism** and computes its own numbers; it does not
reproduce anyone's measurements. The empirical finding it explains -- that an
RL-trained model's pass@k is overtaken by its own base model at large k, on
verifiable reasoning benchmarks -- is:

> Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity
> in LLMs Beyond the Base Model?*, arXiv:2504.13837 (v1 April 2025; latest v5,
> November 2025). <https://arxiv.org/abs/2504.13837>

Nothing on the page is a transcription of that paper's figures, and none of the
numbers it shows should be cited as a measurement of any real model.

## Interactions

| # | Contract item | Here |
|---|---|---|
| 1 | scrub / step | transport steps k over the 11 powers of two from 1 to 1024, autoplays and loops; `?step=N` |
| 2 | hover-to-inspect | a candidate bar → `p`, the weight `p^e · e^β` that produced it, and the resulting `q`; a curve point → both models' `1 − (1 − s)^k` written out term by term; a problem row → its two success probabilities and what each reaches at k = 1024 |
| 3 | direct manipulation | drag the **α track** on the canvas, and drag **any candidate bar** ↕ to change that answer's base probability -- both curves and the crossover recompute under the hand |
| 4 | live animation | the transport autoplays and loops across k, sweeping the k-marker along both curves |
| 5 | resize the problem | candidates per problem (3-14), problems in the benchmark (1-10), and which problem is shown |

**A/B compare** renders weak sharpening (α = 0.3) against strong (α = 3.2).
**Challenge mode** asks first for a crossing at k ≤ 8, then for a sharpening that
helps at *every* k -- the two ends of the trade.

## Render tier

T1 (Canvas2D: probability bars, a per-problem strip, and two curves on a log-k
axis; no heatmap and no large tensor, so nothing above T1 would teach more).

## Wiring

`layout.mount` + controls (`sharp`, `tilt`, `floor`, `n`, `m`, `focus`, `seed`)
+ `transport` over the 11 k values (autoplay, loop) + `onPointer` for the α-track
drag (routed through `controls.set`, so the slider and the deep link track it),
the candidate-bar drag, and click-to-focus on the problem strip + hover. URL
hooks: `?step`, `?sharp`, `?tilt`, `?floor`, `?n`, `?m`, `?focus`, `?seed`,
`?drag=i,logit`, `?hover=x,y`, `?play=1`, `?ch=N`, `?compare=1`, `?theme=`.
Problems come from `tensor.js` `seededRandn(seed,...)` / `rng(seed)`, so one URL
replays one exact picture.

Two implementation notes worth keeping:

- The pass@k tables are **memoized** on every input that can move them (including
 a drag counter). They span 1024 k-values for both models and the page redraws on
 hover, so recomputing per frame would spend the whole frame budget on arithmetic
 nothing asked to change. Within a table, `(1−s)^k` is built incrementally from
 `(1−s)^(k−1)` -- no `pow` in the loop.
- A crossing must open a gap of at least `5e-4` to count. Both curves saturate at
 1.0 once k is large enough to find any answer with `s > 0`, so a bare `q ≤ p`
 test fires on a floating-point **tie** at the top of the chart and reports a
 "crossing" worth 0.0% -- which is not a crossing, it is two curves finishing
 together.

Source: [`page.js`](page.js).
