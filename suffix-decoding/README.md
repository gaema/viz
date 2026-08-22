# suffix-decoding -- speculate from text you have already seen, with no draft model

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: speculative decoding **without a second model**. The proposal
comes from the text the system has already seen -- the prompt, the conversation,
this generation's own output -- indexed in a depth-bounded **suffix tree**.
**Anchor**: the companion to
[speculative-decoding](../speculative-decoding/README.md), which owns the
verify-and-accept mechanism this page reuses and deliberately does **not**
re-teach. Contrast it with [radix-attention](../radix-attention/README.md),
whose tree is a *prefix* tree: there one path is inserted per request, from its
start; here one path is inserted per **position**, so a node's count is the
number of positions that produced it.

Source: **SuffixDecoding**, <https://arxiv.org/abs/2411.04975>.

## What it shows

- **match** -- take the most recent tokens and find the longest suffix of them
 that the tree knows. Not merely the longest suffix *present*: a span seen
 exactly once has one continuation with count 1 and no evidence behind it, so
 the match backs off to the longest span that has **recurred**. The page shows
 when it backs off and by how much.
- **propose** -- walk on from the match, taking the most frequent continuation
 at each step, re-anchoring on the longest recurring suffix every token. No
 model runs; this is a tree lookup.
- **verify** -- one forward pass of the big model covers all the proposed
 positions at once, exactly as it would a draft model's proposal.
- **commit** -- keep the agreeing prefix plus the model's own next token,
 index the new tokens straight back into the tree, so the structure grows as
 the generation goes.

**The property that makes it interesting** is that this gets *better with
repetition*, precisely where a draft model gains nothing: an agent loop
re-emitting a similar tool call, a document being quoted back, structured output
with a fixed skeleton. On genuinely novel prose there is nothing to copy and it
degrades to ordinary one-token-per-forward decoding. Both regimes are reachable
from the repetitiveness handle, and the accepted-length curve moves across them.

## Real vs modelled -- stated on the page too

| | |
|---|---|
| 🟢 **REAL** | the corpus, the suffix tree and every node count, the longest-recurring-suffix match, the occurrence positions, the proposal, and the accepted length -- a literal longest-common-prefix against the reference continuation |
| 🟡 **MODELLED** | that the reference continuation is what the big model would emit. Corpus and reference come from one seeded process whose repetitiveness is the page's knob |
| 🟡 **MODELLED** | the draft-model baseline curve -- an analytic geometric accept model, drawn **dashed** and labelled |

The draft baseline is nearly flat in repetitiveness on purpose: a draft model's
per-token agreement is a property of the model *pair*, not of the text.

## The trade, against the draft model

Both curves share one axis. No draft weights and no draft forward passes -- but
the proposal is only as good as what has been seen before, so its mean accepted
length swings far harder with the input than a draft model's does. Measured on
the page (4 seeds × 12 rounds per point, proposal length 6, tree depth 6):

| repetitiveness | suffix tree, mean accepted | draft model, mean accepted (modelled) |
|---|---|---|
| 0.00 | 0.29 | 2.21 |
| 0.30 | 0.75 | 2.34 |
| 0.60 | 2.00 | 2.47 |
| 0.80 | 2.73 | 2.56 |
| 1.00 | 4.23 | 2.65 |

The suffix tree moves **14.6×** across the range while the draft moves 1.20×,
and the two curves cross between repetitiveness 0.75 and 0.80 (2.27 vs 2.54,
then 2.73 vs 2.56). That crossing is the whole point:
which one wins is a property of the *text*, not of the method.

## Interactions

| # | Contract item | Here |
|---|---|---|
| 1 | scrub / step | transport steps the four phases (match · propose · verify · commit), autoplays and loops; `?step=N` |
| 2 | hover-to-inspect | a tree child → its continuation, how often it was seen, and what share of the context's occurrences it accounts for; a proposed token → the match length and count that produced it, its rival candidates, and whether it survived; the occurrence map → what the node's count actually counts |
| 3 | direct manipulation | canvas handles for **repetitiveness** and **proposal length**, plus an **editable corpus text** field -- paste the line the page offers (the matched context **plus** the continuation, which is what a repeat actually is) and the accepted length jumps: measured, round 1 goes from 1 accepted to 6. Pasting the continuation *without* its context does nothing, because one sighting hanging off nothing loses to better-attested continuations -- the page's hint gives you the form that works |
| 4 | live animation | the match sweeps along the context each frame; the verify phase sweeps the one-forward bracket |
| 5 | resize the problem | tree depth, lines already seen, rounds, proposal length |

**A/B compare** renders novel text (repetitiveness 0.05) against an agent loop
(0.90). **Challenge mode** asks for a mean accepted length above 3.0, and then
for a configuration where the draft model wins.

## Render tier

T1 (Canvas2D token chips, the tree's continuation bars, the occurrence map,
the two-curve axis; no heatmap needed).

## Wiring

`layout.mount` + controls (`rho`, `plen`, `depth`, `corpus`, `rounds`,
`draftA`, `seed`, `extra`) + `transport` over `rounds × 4` phase records
(autoplay, loop) + `animate` + `onPointer` handle drags (routed through
`controls.set`, committed non-silently on release so the deep link syncs)
hover. URL hooks: `?step`, `?rho`, `?plen`, `?depth`, `?corpus`, `?rounds`,
`?draftA`, `?seed`, `?extra`, `?hover=x,y`, `?play=1`. All randomness comes from
`tensor.js` `rng(seed)`, so a given URL reproduces the same run exactly; the
swept curve is also exposed as `window.__suffixCurve` so a headless check can
read the whole curve, which a screenshot cannot. Source: [`page.js`](page.js).
