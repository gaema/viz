# kv-eviction -- a constant-size KV cache over a growing context

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the context keeps growing, the KV cache must not. Three
eviction policies decide what to forget, all holding **the same number of
slots**, and the page scores each one by how much of the real attention mass its
survivors still carry. **Anchor**: A3 attention pattern / KV cache (Family B).

## What it shows

A row of token slots is the cache. Above it, this step's full-cache attention
distribution -- the truth every policy is graded against. Below it, each token's
accumulated attention, which is what makes some of them worth keeping. Evicted
slots are struck through and visibly gone; the mass they were carrying turns
grey in the bar above.

| Policy | Keeps | Why it is here |
|---|---|---|
| **Sliding window** | the last `B` tokens | the obvious policy, constant memory, and the one that fails -- the first tokens go first |
| **Sinks + sliding window** | the first `S` tokens forever + the last `B−S` | StreamingLLM (arXiv:2309.17453): the same budget, rescued by a two-token change |
| **Heavy-hitter eviction** | the top `B` tokens by accumulated attention, wherever they sit | H2O (arXiv:2306.14048): evict the lowest accumulated score when the cache fills |

The score panel shows all three at once, so the ranking is visible on load
rather than after three clicks: **kept attention mass, as a percent of the
full-cache baseline** (100% = the policy discarded nothing this step wanted).
Cache size cannot separate the policies -- it is constant and identical for all
three -- so the score is the only thing that does.

The point the page is built to land: **the naive policy is the one that looks
most reasonable and is the one that breaks.** Attention concentrates on the
first few positions regardless of their content, so a window that evicts them
loses a large share of the distribution while looking perfectly well-behaved.
Dragging the sink bracket from 0 to 4 hands most of it back at the same budget.

### The attention is synthetic, and shaped on purpose

There is no model behind the page. The distribution is built from three additive
logit terms that stand in for the reported behaviour: a strong,
**content-independent** pull toward the first positions (the attention-sink
phenomenon -- this is exactly why dropping token 0 is catastrophic), a recency
term, and a fixed per-token content salience drawn from the seed (which is what
plants a heavy hitter in the middle where no window can see it). Everything
downstream -- the eviction order, the scores, the tooltips -- is then computed
honestly from that distribution, so the picture and the number can never
disagree. The `seed` slider redraws the content without touching the shape.

## How this differs from the sibling pages

| Page | Subject |
|---|---|
| [`attention-patterns`](../attention-patterns/README.md) | the **mask**: which keys a query is *allowed to read*, per pattern. Sliding-window and sink appear there as attention patterns over a full `N×N` grid; nothing is ever deleted. |
| [`kv-cache`](../kv-cache/README.md) | the cache **filling**: K/V columns appended per decode step, and the memory that grows with them. Append-only -- no policy, no eviction. |
| **`kv-eviction`** (this page) | what happens when the cache is **capped**: which entries are *thrown away*, in what order, and what that costs in retained attention mass. |

The distinction is worth stating because it is the one readers collapse: a
sliding-window *mask* and a sliding-window *eviction policy* look alike on a
diagram, but only the second one destroys state it can never get back. This page
starts where `kv-cache` stops growing.

## Render tier

T2 (WebGL2-tier surface; slots, bars and handles are drawn as Canvas2D overlays
for the kept / sink / heavy-hitter / evicted colouring).

## Wiring

`layout.mount` + controls (`policy`, `budget B`, `sinks S`, `N`, `seed`) + a
per-token `Transport` that auto-plays and loops, so the cache visibly fills,
hits its cap and starts evicting. Per the interactivity contract:

- **Direct manipulation** -- drag the `◂▸` budget handle to resize `B` and the
 `◂▸` sink-bracket handle to set `S`; all three policy scores recompute under
 your hand (the simulation re-runs, it is not interpolated).
- **Hover** any slot for its attention this step, its accumulated attention,
 whether it survives **and why** (in-window / sink / heavy-hitter rank /
 evicted, with the step it was evicted at).
- **Transport** across the sequence, autoplay + loop.
- **A/B compare** -- sliding window vs sinks+window at identical budget.
- **Challenge mode** -- break StreamingLLM by dragging `S` to 0, then rescue it.

Headless hooks (`--screenshot` has no pointer, so every handle has a URL twin):
`?step=N`, `?play=1`, `?policy=window|sink|h2o`, `?w=N` (budget), `?sinks=N`,
`?n=N`, `?seed=N`, `?hover=x,y`. Source: [`page.js`](page.js).
