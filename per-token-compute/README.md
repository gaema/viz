# per-token-compute -- not every token needs the same amount of work

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8557 `

Interactive page: a standard transformer spends **identical compute on every
token**, however trivial -- `N` tokens × `L` blocks, one uniform rectangle. This
page draws what happens when that rectangle is allowed to go **ragged**,
what the ragged shape costs to schedule.

**Scope**: this page is about **how much** compute a token receives. Which
*expert* a token picks is [`../moe-routing/`](../moe-routing/README.md),
per-expert load, capacity and drops are
[`../moe-balance/`](../moe-balance/README.md) -- routing is not re-taught here.

## The four mechanisms (modes of one widget)

| Mode | Mechanism | Shape it makes |
|---|---|---|
| **Mixture-of-Depths** | A router picks a capacity-limited subset of tokens to process in each block; the rest skip it via the residual ([arXiv:2404.02258](https://arxiv.org/abs/2404.02258)) | Interior gaps, but a **fixed** count per block |
| **Zero-computation experts** | An MoE where some "experts" are the identity, so routing to one costs nothing and the model learns which tokens deserve real experts | Each token decides for itself, so the batch width **wanders** block to block |
| **Mixture-of-Recursions** | Shared layers applied a variable number of times per token, so a hard token loops more ([arXiv:2507.10524](https://arxiv.org/abs/2507.10524)) | A clean skyline, contiguous from the bottom |
| **Early exit** | The simplest form: a token stops once a confidence threshold is met | A skyline too, driven by a confidence trace rather than a budget |

The picture is the same in all four: **sequence on one axis, depth on the
other**, each token's actual compute path drawn as a ragged height. Easy tokens
(punctuation, predictable continuations) are short; hard ones are tall.

## The cost that makes it honest

Ragged compute is a **scheduling** problem. GPUs want dense, uniform work, so a
batch whose tokens take different paths either **wastes capacity on padding** or
needs **gather/scatter** to compact itself -- and the router that makes the
decision is charged on every token at every block, including the ones it skips.

Both counters are reported against one named baseline, **the uniform-depth
model** (every token through every block, dense, no router), as a percent with
the direction stated -- these are lower-is-better axes, so `>100%` is worse:

- `useful FLOPs … % of the uniform-depth model` -- what the ragged shape saves.
- `machine time … % of the same baseline` -- what the machine actually runs:
 `Σ_blocks (executed rows + router)`, where executed rows are
 `ceil(active / tile) · tile` under **pad to tile**, or `active + gather_cost · N`
 under **gather / scatter**.
- `utilisation` -- useful ÷ machine.

Both are computed live from the reader's settings, and there are reachable
configurations on **both** sides: a real saving, and a net loss where the ragged
shape costs more than it saves.

## Interactions

- **Transport** steps through the blocks (autoplay + loop); tokens visibly drop
 out as they finish, dimming with a ✓ at their exit depth.
- **Direct manipulation**: drag any token's difficulty bar, drag the **capacity**
 knob and the **spread** knob -- the profile and both counters move under your
 hand.
- **Mode switch** across the four mechanisms.
- **Hover** any token for its per-block path, its exit depth, and its share of
 the batch's useful compute; hover a block bar for active / padding / router.
- **URL hooks**: every control is mirrored into the query string,
 `?step=N` (seek a block), `?dif=i:v,…` (forced per-token difficulty -- the
 headless stand-in for a drag), `?hover=x,y`, `?play=1`. The sequence is
 **seeded**, so a link reproduces the picture.

## Render tier

T1 (Canvas2D): the profile grid, the skyline, the per-block width bars and the
two baseline meters are all vector work -- nothing here needs a GPU tier.

## Wiring

`layout.mount` + controls (`mode`, `N`, `L`, `cap`, `spread`, `sched`, `tile`,
`gs`, `seed`) + an `L`-step transport (one step per block) + `onPointer` drags
for the difficulty strip and the two knobs + hover. Source: [`page.js`](page.js).
