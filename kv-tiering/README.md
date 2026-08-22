# kv-tiering -- the cache spilled downhill, and the race on every hit

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: a KV cache that does not end at the accelerator. Blocks pushed
out of the top tier land in host DRAM, then on a local SSD, instead of being
destroyed -- and when one is wanted again the runtime faces a genuine choice
with two different physics behind it: **fetch the block back up the hierarchy,
or recompute it from the tokens**. Both costs are computed for every hit and the
winner is drawn, never asserted. **Anchor**: A3 attention pattern / KV cache
(Family J, serving-time).

## The mechanism

| Quantity | What sets it | Consequence |
|---|---|---|
| `fetch_ms` | `block_bytes ÷ tier_bandwidth` | grows with the block's **size**, shrinks with the tier's **bandwidth** -- so it gets worse the further down the hierarchy you look |
| `recompute_ms` | the model's arithmetic over the block's tokens | grows with the **model** and with the block's **position** in the sequence, shrinks with **spare arithmetic** |

Because the two are priced by unrelated resources, the winner **flips**. It
flips with distance down the hierarchy (a slower tier favours rebuilding), with
the machine's arithmetic headroom (a fast, idle machine rebuilds almost for
free), with how fat each token's KV is, and -- the part that is easy to miss --
with **where the block sits**. A block of `L` tokens at offset `pos` has to be
attended against everything before it, so its recompute cost carries an
`L × (pos + L/2)` term: the same block is cheap to rebuild at the start of a
conversation and dear to rebuild 3,000 tokens in.

That term is also what makes a **break-even block size** exist rather than
cancel out. Fetch is linear in `L`; recompute is not. Setting them equal solves
to

```
L* = 2 × (fetch_per_token − linear_per_token − attn_per_pair × pos) ÷ attn_per_pair
```

which the page prints live, per tier. Blocks larger than `L*` are cheaper to
fetch; smaller ones are cheaper to rebuild. When `L*` comes out non-positive
there is no crossing at all and fetching wins at every size -- the page says
that instead of printing a number that does not exist.

## What is on screen

- **The hierarchy** -- three tiers, each with a capacity strip (one cell per
 slot, so empty capacity is visible) and a pipe carrying its bandwidth and its
 measured busy fraction. Blocks are coloured by conversation; the ones this
 request needs are outlined.
- **The race chart** -- both costs against block size, on log axes, with the
 crossings marked. Fetch is a straight line through the origin; recompute bends
 upward. A second, dashed recompute curve shows the same block deep into a
 conversation, which is the position effect made visible.
- **The verdict strip** -- for the current request, one column per block: a
 fetch bar, a recompute bar, and the one that was actually paid drawn solid.
- **The occupancy timeline** -- one compute unit and one link per tier
 boundary, all serial. This is where the page charges the thing that is easy to
 hand-wave: **a deeper tier is not free even on a hit.** Every fetch occupies
 its link, and so does every *spill* -- the transfer that happens when a block
 is pushed down to make room, which nobody asked for.
- **The policy comparison** -- always fetch, always recompute,
 cheaper-of-the-two, each scored against the baseline.

## The baseline

Everything is reported as a percent of **GPU-only caching**: the same top-tier
capacity, nothing below it, so an evicted block is simply gone and any later hit
is a full recompute. Lower is better, `100%` is parity. The readout states the
percent, the absolute means, and the link traffic that bought the difference --
because tiering is not free, and at a thin enough link (or a cheap enough
recompute) it **loses**, which the page will say in as many words.

## Interaction

| Control | What it does |
|---|---|
| transport | steps the arrival stream, one request per step; blocks migrate between tiers as it is served. Auto-plays and loops. |
| drag a capacity strip | resize that tier, live |
| drag a pipe | change that link's bandwidth (log scale) |
| drag the ↕ handle on the race chart | change the arithmetic rate, moving every crossing |
| hover | any block (tier, size, age, and BOTH candidate costs), any timeline segment, any policy bar, the chart, a pipe, a capacity strip |
| policy switch | always fetch / always recompute / whichever is cheaper |
| A/B compare | always-recompute vs cheaper-of-the-two, side by side |

Every control is mirrored into the query string, and `?step=N` plus `?seed=`
mean one URL replays one arrival stream at one moment:
`?step`, `?policy`, `?cap0`, `?cap1`, `?cap2`, `?bw1`, `?bw2`, `?rate`,
`?params`, `?kvkb`, `?blk`, `?nreq`, `?gap`, `?seed`, `?hover=x,y`.

Two URLs where the same question gets opposite answers, both on the host-DRAM
tier, same block size, same page:

| URL | host-DRAM verdict |
|---|---|
| `?step=3&policy=cheaper&bw1=55&params=3&rate=400&kvkb=128&blk=256` | **fetch** -- 0.61 ms of link time beats 3.88 ms of arithmetic |
| `?step=3&policy=cheaper&bw1=4&params=1&rate=1000&kvkb=512&blk=256` | **recompute** -- 33.6 ms of link time loses to 0.58 ms of arithmetic |

## The numbers are yours, and they are nominal

There is no machine behind this page. Bandwidths, capacities, model size, KV
per token and the arithmetic rate are all reader-set sliders whose defaults are
**public nominal figures** of the right order (a PCIe-class host link, a
consumer NVMe sequential read rate, a small dense model at a plausible effective
arithmetic rate). They set the *shape* of the tradeoff, which is the lesson.
Everything downstream -- residency, verdicts, queueing, the means -- is then
simulated honestly from those inputs, so the picture and the number can never
disagree.

## How this differs from the sibling pages

| Page | Subject |
|---|---|
| [`paged-attention`](../paged-attention/README.md) | **paging inside the top tier**: fixed-size blocks plus a per-sequence block table, versus one contiguous slab. Nothing leaves the accelerator. |
| [`kv-eviction`](../kv-eviction/README.md) | **which blocks to drop** under a fixed budget, and what that costs in retained attention mass. The dropped block is gone. |
| **`kv-tiering`** (this page) | what happens to a block **after it leaves the top tier** -- where it goes, what it costs to get back, and which of the two ways of getting it back is cheaper. |

The three compose in that order: `paged-attention` says a cache is made of
blocks, `kv-eviction` says which block leaves, and this page picks the story up
at the moment it leaves. Neither of the other two is re-taught here.

## Sources

Public material describing KV offloading across a memory hierarchy and the
fetch-versus-recompute decision on a prefix-cache hit:

- The vLLM project's public documentation on prefix caching and KV connectors /
 KV offloading (`docs.vllm.ai`).
- The LMCache project's public documentation and papers on storing KV outside
 accelerator memory (DRAM and disk tiers) and reusing it across requests
 (`lmcache.ai`).
- "Efficient Memory Management for Large Language Model Serving with
 PagedAttention" (arXiv:2309.06180) for the block-structured cache this tiering
 is layered on top of.
