# speculative-decoding -- draft k tokens, verify them in one pass

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a small **draft** model buys tokens from a big **target**
model without changing what the big model would have said. **Anchor**: the
decode-time companion to [prefill-vs-decode](../prefill-vs-decode/README.md)
(which shows why one forward per token is the bottleneck)
[sampling](../sampling/README.md) (whose accept/reject test this one extends).

## What it shows

Plain decode emits one token per forward pass of the big model, one after
another -- and a single token is a skinny matrix-vector product, so the hardware
spends most of its time waiting on memory rather than doing math. Speculation
spends that idle width instead:

- **propose** -- the cheap draft model runs `k` times and guesses the next `k`
 tokens.
- **verify** -- the target runs **ONE** forward over all `k+1` positions. They
 batch exactly the way a prompt does, so the extra positions are close to free;
 the pass yields the target's own distribution `p` at every one of them.
- **accept test** -- walk the proposals left to right and accept proposal `x`
 with probability `min(1, p(x)/q(x))` against a drawn uniform, where `q` is the
 draft's own probability for the same token. Stop at the **first**
 disagreement.
- **commit** -- keep the accepted prefix. At the rejection, the target emits its
 **own** token, drawn from the residual `normalize(max(0, p − q))` -- which is
 what makes the whole scheme produce exactly the target's distribution. If
 every proposal survived, the extra position the same forward already covered
 yields one **free bonus** token.

So a round always yields **at least 1 and at most k+1** tokens. Accepted
proposals are green, the rejected one is red, and the tail after it is struck
through and thrown away untested -- that discarded draft compute is the price of
the trade.

The economics panel is the point of the page: **mean accepted length per round
is everything**. It shows tokens produced, target forwards spent, and the
resulting **tokens per target forward** against the baseline of `1.00` (plain
decode), plus the **net** rate once each drafted token is charged a fraction of
a target forward. Drag the agreement down and the cost up and the verdict flips
to *SLOWER than not speculating* -- a bad draft is worse than no draft.

**Drag** the draft-agreement and draft-cost handles on the canvas; the counters,
accepted lengths and verdict move under your hand. **Hover** any proposal for
its draft probability `q`, its target probability `p`, the ratio, and the exact
comparison that decided it. The run is generated from a **seeded** generator, so
one URL replays one exact run.

## Interactions

| # | Contract item | Here |
|---|---|---|
| 1 | scrub / step | transport steps the four phases of a round (propose · verify · accept test · commit), autoplays and loops; `?step=N` |
| 2 | hover-to-inspect | any proposal chip → `q`, `p`, `p/q`, the drawn uniform, the verdict and **why**; the discarded tail says it was never tested |
| 3 | direct manipulation | canvas handles for **draft agreement** and **draft cost**, with the accepted lengths and the tokens-per-forward counter live under the drag |
| 4 | live animation | the verify phase sweeps the one-forward bracket across all `k+1` positions each frame |
| 5 | resize the problem | `k` (draft length) and `rounds` steppers |

**A/B compare** renders a weak draft (agreement 0.25) against a strong one
(0.95). **Challenge mode** asks for >2.00 tokens per target forward, and then
for a configuration where speculation actually loses.

## Render tier

T1 (Canvas2D token chips, probability bars, and the tokens-per-forward gauge; no
heatmap needed).

## Wiring

`layout.mount` + controls (`k`, `quality`, `cost`, `rounds`, `seed`)
`transport` over `rounds × 4` phase records (autoplay, loop) + `animate` (the
verify sweep via `api.t`) + `onPointer` handle drags (routed through
`controls.set`, committed non-silently on release so the deep link syncs)
hover. URL hooks: `?step`, `?k`, `?quality`, `?cost`, `?rounds`, `?seed`,
`?hover=x,y`, `?play=1`. All randomness comes from `tensor.js` `rng(seed)` /
`seededRandn(seed,...)`, so a given URL reproduces the same run exactly.
Source: [`page.js`](page.js).
