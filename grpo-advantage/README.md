# grpo-advantage -- the group is the baseline

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8460 `

Interactive page for the one piece of GRPO a reader can actually manipulate: the
**group-relative advantage**. Sample a group of rollouts for one prompt, score
each with a verifiable reward, and set each rollout's advantage to its reward
**minus the group mean**. There is no learned value network anywhere in that --
the other samples of the same prompt *are* the baseline.

**Anchor**: the training-time companion to
[pass-at-k](../pass-at-k/README.md). That page is about what RL *did* to a
finished model -- it sharpens the distribution, buying the first sample
selling the tail -- and it takes the training as a given. This page is one
gradient step earlier: where the signal that does the sharpening comes from,
when there is none. Read together, `pass-at-k` shows the *outcome* of pushing
mass onto rollouts that already worked, and `grpo-advantage` shows the
*arithmetic* that decides which rollouts get pushed. Both compute everything
in-page from a seeded synthetic group; neither reproduces anyone's measurements.

## What it shows

| symbol | what it is |
|---|---|
| `G` | rollouts sampled for this one prompt |
| `r_i` | the verifier's reward for rollout `i` -- correct/wrong, plus an optional partial-credit spread |
| `μ` | `(1/G) Σ r_i` -- the **group mean**, and the whole baseline |
| `σ` | `sqrt((1/G) Σ (r_i − μ)²)` -- how much the group disagrees |
| `A_i` | `r_i − μ`, optionally `÷ (σ + 1e−4)` |
| `ρ_i` | the ratio between the new and old policy on rollout `i` |
| clipped term | `min(ρ_i·A_i, clip(ρ_i, 1−ε, 1+ε)·A_i)` |
| gradient signal | mean `|A|` over the rollouts still inside the clip window |

The clipped objective bounds how far one step may move the policy. When the
clipped branch wins, the term is **constant in `ρ`**, so it has no gradient at
all -- and that happens exactly when the step would push further in the direction
the advantage already likes:

| condition | what happens |
|---|---|
| 🟢 `1−ε ≤ ρ_i ≤ 1+ε` | the `ρ·A` branch wins -- this rollout still moves the policy |
| 🔴 `A_i > 0` and `ρ_i > 1+ε` | clamped: the term stops depending on `ρ`, gradient zero |
| 🔴 `A_i < 0` and `ρ_i < 1−ε` | clamped, same reason on the other side |

The asymmetry is the point: a rollout can be far outside the window and still
carry gradient, if the step is pulling it *back* toward 1.

## The failure the page is built around

**If every rollout in the group earns the same reward, every advantage is exactly
zero.** All correct, or all wrong -- it does not matter which. Every `r_i` equals
`μ`, so every `r_i − μ` is `0`, and the group contributes no gradient whatsoever.
Dividing by the std does not rescue it: `0 / (0 + 1e−4)` is still `0`.

It is reachable by hand in two moves -- click each rollout's ✓/✗ tag until they
agree, or drag every reward bar to the same place -- and the gradient signal
readout then reads `0.000` in the alarm colour. Two URLs that land on it
directly:

| URL | reading |
|---|---|
| `index.html?pass=1&spread=0&g=8&step=7` | every rollout correct, `μ = 1.000`, `σ = 0.000`, `A = [+0.000 × 8]`, **gradient signal = 0.000** |
| `index.html?pass=0&spread=0&g=6&step=5` | every rollout wrong, `μ = 0.000`, same collapse |
| `index.html?g=4&rewards=0.4,0.4,0.4,0.4&step=3` | the drag stand-in: four rewards set by hand to the same value, `μ = 0.400`, **0.000** |

That third one matters: agreement at an *intermediate* reward collapses the
signal just as completely as agreement at 1.0. The group has no idea whether
0.400 is good; it only ever sees differences.

This is why **dynamic sampling** exists -- filter out the all-correct
all-wrong groups and sample replacements until the batch is full of groups that
disagree. Without it, a batch spends its forward passes on groups that cannot
contribute, and the effective batch size silently shrinks as the model gets
better at the easy prompts.

## Two more things the page lets you find

- **A small group is a noisy baseline.** With `G = 2` the mean is one other
 sample, and the two advantages are forced to be `+d` and `−d` -- the readout
 prints `σ/√G` beside it so the noise is visible rather than asserted.
- **Dividing by the std is not free.** Turn normalisation on with a group that
 barely disagrees (`σ ≈ 0.02`) and watch tiny advantages get amplified back to
 full size by `1/(σ + 1e−4)`. A prompt the model has nearly mastered then
 carries the same weight as one it is genuinely learning from. That bias,
 the related response-length bias, are what Dr. GRPO objects to.

## What is NOT shown, and why

**No policy update. No weights. No training curve.** That is a scoping decision,
not an omission: a weight update over a transformer is not something a reader can
watch, and animating one would be a fiction dressed as a mechanism. What *is*
live math is the quantity the update is built from -- and it is the part where
the interesting failure lives -- so the page shows that and says so on the canvas.
The policy ratios `ρ` are seeded synthetic values standing in for "how far a
candidate step moved this rollout's probability"; they are an input to the clip
arithmetic here, never a claim about a real optimizer's behaviour.

Nothing on this page is a transcription of any paper's figures, and none of its
numbers should be cited as a measurement of any real model or training run.

## Where the mechanism comes from

> Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
> Open Language Models* (which introduces GRPO), arXiv:2402.03300 (2024).
> <https://arxiv.org/abs/2402.03300>

> Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*
> (Dr. GRPO) -- the std-normalisation and response-length biases,
> arXiv:2503.20783 (2025). <https://arxiv.org/abs/2503.20783>

## Interactions

| # | Contract item | Here |
|---|---|---|
| 1 | scrub / step | the transport scores the group **one rollout at a time**, so the baseline visibly moves as each arrives; autoplays and loops; `?step=N` |
| 2 | hover-to-inspect | a reward bar → the reward and the full `μ` sum written out; an advantage bar → `r − μ`, the division if normalisation is on, and why it is or is not clamped; a ratio marker → `ρ`, the window, and the value of `min(ρ·A, clip(ρ)·A)`; the gauge → the mean-`|A|` sum term by term |
| 3 | direct manipulation | **drag any reward bar ↔** and the mean, every advantage, the clipped fraction and the gradient signal recompute under the hand; **click a ✓/✗ tag** to flip that rollout between reward 1 and 0 |
| 4 | live animation | the transport autoplays and loops across the group |
| 5 | resize the problem | group size `G` (2-16), the fraction the verifier marks correct, the partial-credit spread, the clip window `ε`, the policy drift, and the seed |

**A/B compare** puts a group that disagrees next to one where every rollout is
correct -- the second pane is all zeros. **Challenge mode** asks first for an
exactly-zero gradient signal, then for a configuration where at least half the
group is clipped while the survivors still carry signal above 0.15.

## Render tier

T1 (Canvas2D: G reward bars, G signed advantage bars, a ratio scatter against a
shaded clip window, and one gauge -- no heatmap and no large tensor, so nothing
above T1 would teach more).

## Wiring

`layout.mount` + controls (`g`, `pass`, `spread`, `norm`, `eps`, `drift`,
`seed`) + `transport` over the G rollouts (autoplay, loop) + `onPointer` for the
reward drag and the verdict click. URL hooks: `?step`, `?g`, `?pass`, `?spread`,
`?norm`, `?eps`, `?drift`, `?seed`, `?r=i,value` (one reward -- the headless
stand-in for a bar drag), `?rewards=a,b,c,…` (the whole group at once),
`?hover=x,y`, `?play=1`, `?ch=N`, `?compare=1`, `?theme=`. The group comes from
`tensor.js` `rng(seed)` / `seededRandn(seed, …)`, so one URL replays one exact
picture.

Three implementation notes worth keeping:

- **The transport reads `G` off the live control state, not off a page global.**
 `compute` runs during the transport's rebuild, which is *before* the next
 `draw`, so a global written in `draw` is one edit stale and the step list
 ends up sized for the previous group.
- **Rebuild before seeking.** `?g=10&step=9` changes the step count and then
 seeks; against the stale step list the seek clamps to the old last index
 lands on "8 of 10". `rebuildIfDirty` first.
- **The clip window gets one centred label, not one per edge.** At a small `ε`
 the `1−ε` and `1+ε` labels collide into an unreadable run of digits.

Source: [`page.js`](page.js).
