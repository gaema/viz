# verifier-selection -- three ways to spend the same test-time budget

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: you are allowed **N generations per problem**. There are three
well-known ways to spend that budget, they do not agree about which is best,
one of them gets **worse** the more you spend. **Anchor**: the selection-time
companion to [pass@k](../pass-at-k/README.md). pass@k asks whether the model can
produce a right answer at all among many tries; this page asks the question that
comes immediately after -- **how do you CHOOSE among those tries when you cannot
check the answer?** pass@k is an upper bound that assumes a perfect oracle picks
for you. Here the picker is fallible, and that changes the shape of the curve.

## The three spends

| strategy | what it does with the budget | needs a verifier? |
|---|---|---|
| **best-of-N** | sample N complete attempts, score each, keep the top one | yes -- an outcome verifier / reward model |
| **majority vote** (self-consistency) | sample N complete attempts, return the most common **answer** | no |
| **beam / stepwise search** | spend the budget on **breadth at intermediate steps**, keeping the b best prefixes each step | yes -- a **process** reward model, scoring partial work |

## The result that makes it a page

**Best-of-N optimises whatever the verifier scores, which is not the same thing
as correctness.** Give the verifier an accuracy below 1 and best-of-N **turns
over**: it climbs while extra samples are mostly buying coverage, peaks, and then
declines, because every further sample is another chance to turn up a wrong
answer the verifier likes better than the right one. The verifier's score for an
answer is *fixed* -- sampling the same wrong answer again does not lower it -- so
more budget is a wider search for the verifier's own mistakes.

**Majority vote cannot fail that way, because it never consults a verifier** --
there is nothing there to game, so the flawed-verifier slider moves the blue curve
and leaves the violet one exactly where it was. It fails a *different* way,
the distinction is worth keeping sharp: majority vote converges on the model's
**most common** answer, so when the model is confidently and *consistently* wrong,
extra votes only make the wrong consensus firmer. Raise **wrong-answer
consistency** toward 3 and majority vote ends **lower than it started** -- measured
at the page's defaults, `36.8% at N = 1 → 34.6% at N = 64`, with the verifier left
at 0.95. So "majority vote never declines" would be an overclaim; the true
statement is that it never declines *for the verifier's reason*.

**Beam search sits between them and has its own failure.** It gets more out of the
same budget when the process reward model is good, because it re-uses partial
progress instead of restarting; and when the process reward model is bad it prunes
correct prefixes early and never recovers, since a discarded prefix cannot come
back. It also cannot use a budget smaller than the number of steps.

Drag the **verifier-accuracy** track down and watch exactly this: the best-of-N
curve rolls over while the majority curve does not move at all.

## The model, in full

Everything on screen is computed in-page from the model below. **No number here
is transcribed from a paper**, and none of them should be cited as a measurement
of any real model or benchmark.

| symbol | what it is |
|---|---|
| `d_j` | problem *j*'s difficulty = benchmark mean + spread × a seeded offset |
| `s_j` | `sigmoid(6·(a − d_j))` -- probability one sample from a model of ability `a` is right |
| `p` | the answer distribution: `p[correct] = s_j`; the remaining mass spreads over the wrong answers with concentration `g` (large `g` = one wrong answer the model keeps giving) |
| `v_i` | the verifier's **fixed** score for answer *i* = `z_i + μ·[i correct]`, `z_i` seeded noise |
| `μ` | `√2 · Φ⁻¹(q)`, so the slider `q` is exactly *P(the verifier scores a random correct answer above a random wrong one)* |
| `r` | beam's per-step correctness, `s_j^(1/D)` -- so beam and the other two share one underlying ability |

All three strategies are evaluated **exactly**, not by simulation, so the curves
are smooth and the crossings are arithmetic rather than sampling noise:

- **best-of-N** returns the highest-scoring answer *present* among the N draws, so
 `P(pick i) = C_i^N − (C_i − p_i)^N` where `C_i` is the total probability of the
 answers scoring at or below `v_i`. One line, exact, no Monte Carlo.
- **majority vote** is `P(the correct answer is the plurality)`, ties split
 uniformly, by a dynamic program over the wrong answers' count vectors --
 `Σ_k P(correct drawn k times) × P(every other answer drawn fewer, weighted by
 1/(1+ties))`.
- **beam** is a dynamic program over the number of still-on-track beams. Each step
 generates `c = floor(N/D)` continuations and keeps `b`; the process reward model
 ranks perfectly with probability `2q−1` and uniformly at random otherwise, which
 makes the survivor count a mixture of a truncation and a hypergeometric draw.

### The baseline, and a caveat that is load-bearing

Every percentage on the page is reported as a **percent of the greedy single
sample** -- one generation, no budget, no verifier, take the model's most likely
answer. Higher is better; **100% = parity with spending nothing**. That is the
comparison a reader actually needs: a strategy that costs 64× the compute
lands at 45% of the baseline has spent the budget to become *worse than not
having it*, which the raw accuracy alone does not make obvious.

The caveat, stated on the page as well: **in this model** majority vote's *limit*
as N → ∞ is exactly the greedy baseline, because sampling is i.i.d. over whole
answers, so the most common answer converges on the most likely one. At a
**finite** N it can sit on either side of that limit -- a correct answer that is a
close second still wins some pluralities -- so majority vote reading above 100% of
the baseline at N = 64 is convergence in progress, not a contradiction. A real
language model does not work this way at all: greedy decoding is per-token argmax
and need not produce the most likely *answer*, which is why self-consistency does
beat greedy decoding in practice. The page's limit is a property of its own
simplification, and saying so is cheaper than quietly implying a result the model
cannot support.

## Where the empirical claims come from

The page teaches the mechanism and computes its own numbers. The published work
it explains:

> Cobbe et al., *Training Verifiers to Solve Math Word Problems*,
> arXiv:2110.14168 (2021). <https://arxiv.org/abs/2110.14168> -- sample many
> solutions, rank them with a trained verifier.
>
> Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language
> Models*, arXiv:2203.11171 (2022). <https://arxiv.org/abs/2203.11171> -- majority
> vote over sampled reasoning paths, no verifier required.
>
> Lightman et al., *Let's Verify Step by Step*, arXiv:2305.20050 (2023).
> <https://arxiv.org/abs/2305.20050> -- process supervision: reward the
> intermediate steps, not only the final answer.
>
> Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective
> than Scaling Model Parameters*, arXiv:2408.03314 (2024).
> <https://arxiv.org/abs/2408.03314> -- which strategy wins depends on problem
> difficulty and on the budget; none of them dominates.

That last one is the reason this page has three curves instead of a
recommendation. Move the difficulty and budget sliders and the winner changes,
which is the finding.

## Interactions

| # | Contract item | Here |
|---|---|---|
| 1 | scrub / step | transport steps the budget N over 12 values from 1 to 64, autoplays and loops; `?step=N` (0-based index into the ladder) |
| 2 | hover-to-inspect | a curve point → all three accuracies at that N **and** the selection arithmetic written out for the shown problem (`C^N − (C−p)^N` with its numbers, the plurality probability, the beam's `c`/`b`/`r`); a candidate bar → its `p`, its verifier score decomposed into noise + separation, and each strategy's probability of picking it; a problem row → per-strategy accuracy and whether the verifier's top-scoring answer is the right one; the beam strip → the per-step on-track trace |
| 3 | direct manipulation | drag the **verifier accuracy**, **model accuracy** and **problem difficulty** tracks on the canvas -- all three curves recompute under the hand; click any problem row to open it |
| 4 | live animation | the transport autoplays and loops across the budget ladder, sweeping the N-marker along all three curves |
| 5 | resize the problem | candidate answers per problem (3-7), problems in the benchmark (4-12), reasoning steps D (2-6), beam width b (1-6) |

**A/B compare** renders an accurate verifier (0.95) against a flawed one (0.62)
-- the same budget, the same benchmark, opposite verdicts about whether spending
it helps. **Challenge mode** asks first to make best-of-N give back ≥ 5 points
after its peak, then to make *majority vote* end lower than it started with the
verifier left accurate -- the two failure modes, one at a time, so neither can be
mistaken for the other.

## Render tier

T1 (Canvas2D: probability bars, a per-problem strip, a beam-survival strip,
three curves on a log-N axis; no heatmap and no large tensor, so nothing above T1
would teach more).

## Wiring

`layout.mount` + controls (`vacc`, `macc`, `diff`, `spread`, `wcon`, `depth`,
`beam`, `cands`, `probs`, `focus`, `seed`) + `transport` over the 12 budgets
(autoplay, loop) + `onPointer` for the three canvas tracks (routed through
`controls.set`, so the sliders and the deep link track them) and click-to-focus on
the problem strip. URL hooks: `?step`, `?vacc`, `?macc`, `?diff`, `?spread`,
`?wcon`, `?depth`, `?beam`, `?cands`, `?probs`, `?focus`, `?seed`, `?hover=x,y`,
`?play=1`, `?ch=N`, `?compare=1`, `?theme=`. Problems come from `tensor.js`
`seededRandn(seed,...)` / `rng(seed)`, so one URL replays one exact picture.

Three implementation notes worth keeping:

- The problem set stores only **latents** (difficulty offset, which answer is
 correct, the wrong-mass shape, the verifier's per-answer noise). Everything the
 sliders control is applied on top of them each frame, so dragging a slider never
 rebuilds the benchmark under the reader -- only `seed`, the problem count and the
 answer count do.
- The whole 12-budget sweep is **memoized** on every input that can move it. It is
 twelve budgets × M problems of exact multinomial DP and the page redraws on every
 hover, so recomputing per frame would spend the frame budget on arithmetic
 nothing asked to change. Measured cost of one full sweep at the defaults: tens of
 milliseconds, which is why the exact DP is affordable at all and no sampling is
 needed.
- Best-of-N's response to the verifier slider is **piecewise constant**, and that
 is correct rather than a bug: `argmax` over a fixed set of scores only changes
 when the *ranking* changes, so nudging the separation does nothing until it
 reorders two answers. A verifier is a ranking, not a number, and the page shows
 that honestly instead of smoothing it.

Source: [`page.js`](page.js).
