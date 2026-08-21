# attention-sinks -- the position everything attends to and nothing reads

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8451`

Interactive page: why a head parks most of its attention on the first token or
two **regardless of what those tokens say**, why deleting them wrecks the
output, and the two shipped remedies that pull in opposite directions and both
work. **Anchor**: A3 attention pattern / softmax normalization (Family B).

## The mechanism, in one line

Softmax must sum to 1. A head that has nothing it wants to read still has to put
its whole probability mass somewhere, and the only positions every query can see
are the earliest ones -- so it dumps the mass there. Those positions become
**attention sinks**: enormous attention weight, almost no contribution to the
output vector. Named and measured in StreamingLLM
([arXiv:2309.17453](https://arxiv.org/abs/2309.17453)), which reports that
keeping a couple of early tokens resident is what stops perplexity exploding
under a sliding-window cache.

## The trade, made measurable in-page

A sink costs a cache slot that is **never evicted** and contributes **almost
nothing** to the output. The page puts both numbers side by side, because the
"aha" is that they disagree wildly:

| Card | What it is |
|---|---|
| **attention mass on the sinks** | Σ over the sink positions of `a[q][i]` -- what the softmax hands them |
| **their share of the output** | Σ over the sink positions of `aᵢ‖vᵢ‖ / Σⱼ aⱼ‖vⱼ‖` -- what they actually contribute |
| **cost of deleting the sinks** | `‖o′ − o‖ / ‖o‖`, averaged over heads -- how far the output vector moves when they are gone (lower is better; 0% = deleting them costs nothing) |

On the default view the first two are roughly an order of magnitude apart,
the third is enormous: the head never wanted those tokens' *content*, it wanted
somewhere to put the mass -- and taking that somewhere away forces the mass onto
tokens that do carry value.

## The two remedies, which are opposites

| Remedy | Where it acts | What it does |
|---|---|---|
| 🟢 **learned per-head sink logit** | the softmax **denominator** | appends an extra logit `ℓ` with no value vector behind it -- an explicit no-op option, so spare mass has a legitimate home. The "off-by-one softmax" idea (Miller, *Attention Is Off By One*), shipped as a learned per-head sink logit in current open-weight models |
| 🟢 **per-head output gate** | the head's **output** | lets the head scale its own contribution down instead, so it never needs the attention distribution to mean "nothing here". Gated Attention ([arXiv:2505.06708](https://arxiv.org/abs/2505.06708)) reports head-wise output gating removes the sink outright |

One adds a place for the mass to go; the other removes the need for the mass to
mean anything. **Both free the pinned cache slot**, and both cost a trained
parameter the base model does not have -- which is the honest price, and the
reason a base model cannot simply be patched into either.

## The inputs are synthetic, and shaped on purpose

There is no model behind the page, so the inputs are generated and deliberately
shaped, and everything downstream is then computed honestly from them.

| | |
|---|---|
| **derived in-page** | the softmax (including the extra denominator term), the attention weights, the output vector, the contribution shares, the drift when the sinks are deleted, every number in the readout and in the tooltips |
| **shaped inputs** | a content-independent positional bias decaying from position 0 (the learned sink bias), per-position value vectors whose norm is tiny at the sink positions -- which is the reported shape: high attention, near-zero value -- and per-head variation in sink strength |
| **modelled, and labelled as such on screen** | both remedies shrink that learned bias in proportion to how much no-op capacity they hand the head. That relationship is the reported result -- a trained model with somewhere legitimate to put spare mass does not learn the big bias in the first place -- and it is not something one forward pass can derive |

The `seed` slider redraws the token content without touching any of those shapes.

## How this differs from the sibling pages

| Page | Subject |
|---|---|
| [`attention-patterns`](../attention-patterns/README.md) | the **mask**: which keys a query is *allowed* to read, per pattern. "Sink" appears there as a pattern over an `N×N` grid -- who may attend to whom, not why the mass goes there |
| [`kv-eviction`](../kv-eviction/README.md) | the **policy**: with a capped cache, which entries get thrown away and what that costs in retained attention mass. It shows that keeping 2 sink tokens takes retained mass from 15.8% to 92.6% |
| **`attention-sinks`** (this page) | **why the sink exists at all**, what it is worth on the two axes that disagree, and the two trained mechanisms that make it unnecessary |

`kv-eviction` answers *what should the cache keep*. This page answers *why is
that particular slot the one you must never drop* -- and then removes the reason.

## Render tier

T2 (WebGL2-tier surface; the four bar rows, the no-op column, the per-head strip
and the score cards are drawn as Canvas2D overlays).

## Wiring

`layout.mount` + controls (`remedy`, `sink logit ℓ`, `output gate g`, `delete
the sinks`, `sink tokens S`, `window W`, `heads H`, focused head, `N`, `seed`)
a per-query-position `Transport` that auto-plays and loops. Per the
interactivity contract:

- **Direct manipulation** -- drag the `◂▸` no-op column up/down to set the sink
 logit `ℓ`, or the `◂▸` gate handle left/right to set `g`; either grab also
 switches the remedy to the one being dragged, and every number recomputes
 under your hand (the simulation re-runs, it is not interpolated).
- **Delete the sinks** -- one toggle removes the pinned positions from the cache
 and the distribution visibly collapses onto the tokens behind them.
- **Hover** any position for its logit split into content + learned bias, its
 attention weight with the arithmetic, its value norm, its share of the output,
 and -- for a sink -- how many times bigger it is in attention than in output.
 Hover the no-op column for what the extra denominator term is worth.
- **Transport** across the query positions, autoplay + loop.
- **A/B compare** -- base model vs learned sink logit at identical settings.
- **Challenge mode** -- read the gap, break it by deleting, then free the slot.

Headless hooks (a screenshot has no pointer, so every handle has a URL twin):
`?step=N`, `?play=1`, `?remedy=none|sinklogit|gate`, `?sinkl=F`, `?gate=F`,
`?del=0|1`, `?s=N` (sinks), `?w=N` (window), `?heads=N`, `?head=N`, `?n=N`,
`?seed=N`, `?hover=x,y`. Source: [`page.js`](page.js).
