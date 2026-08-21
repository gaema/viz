# radix-attention -- one KV cache shared across requests

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a serving runtime avoids recomputing the same prompt
prefix for every request by keeping prompts in a **radix tree** of token
sequences, so a shared prefix is stored -- and attended over -- exactly once.
**Anchor**: A4 KV cache layout, one level up from [`../kv-cache/`](../kv-cache/README.md)
(which shows the cache of a *single* sequence).

## What it shows

Three stacked bands. On top, the **incoming requests**: one row per prompt,
one box per token, drawn on a shared token grid. Below them, the **radix tree**
built by inserting those prompts in arrival order -- each node owns a *span* of
tokens plus the KV for exactly that span, drawn on the same grid, so a node's
horizontal extent is literally the token range it covers. A run of tokens with
no branch is path-compressed into one node; when a new request agrees with only
part of a node's span, the insert **splits** it. Node fill deepens with the
number of requests sharing it, and the `×N` badge names that count.

Stepping the transport admits the next request: it walks the tree, takes the
**longest matched prefix**, and lights those tokens green (**reused** -- their
keys and values are already resident, so prefill skips them) while the
unmatched suffix goes orange (**computed**) and hangs off a new node. The
running **tokens reused vs computed** bar underneath is the whole point of the
structure -- with a long shared system prompt most of every later request is
already in the tree.

The tree is finite: a **capacity** control caps how many tokens it may hold,
and when an insert overflows it the **least-recently-used leaf** is evicted
(ghosted with a dashed outline, and its span released). An interior node with
live children is never evictable, because its children's KV is only reachable
through the path that runs over it.

## Interactions

- **Transport** — one step per arriving request, auto-playing and looping:
 match, reuse, compute the suffix, grow the branch, evict if over capacity.
- **Direct manipulation** — drag the ▽ **divergence handle** on any request row
 to move where that request stops agreeing with the row above it. The tree
 re-shapes and the reuse counter moves under your hand; drag it right
 computed tokens turn into reused ones. The `edit divergence of request` /
 `…diverges at token` steppers do the same thing numerically and stay in sync
 with the handles.
- **Hover-to-inspect** — a tree node reports its token span, how many requests
 share it, its KV footprint (with the `2 · layers · kv_heads · head_dim ·
 dtype` derivation) and when it was last touched (the LRU key); an evicted
 node says why it was the one to go; a request row traces its whole path
 through the tree and its reused/computed split; the counter bar reports the
 recompute saved.
- **Resize the problem** — `requests`, `tokens per request`, `cache capacity`
 and `layers` (which scales the KV bytes per token).
- **A/B compare** — first request (cold tree, everything computed) beside the
 last (warm tree, prefix reused).
- **Challenge mode** — make one request a total cache hit, and push overall
 reuse past half of all tokens.

## Render tier

T1 (Canvas2D). The lesson is structure -- spans, branches, one path per shared
prefix -- not pixel throughput, so nothing here needs a GPU tier.

## Wiring

`layout.mount` + controls (`reqs`, `plen`, `req`, `diverge`, `capacity`,
`layers`) + a step `Transport` whose `compute` inserts every request into a
freshly built radix tree and snapshots it per step, so scrubbing back and forth
replays exact tree states. Drawing is plain `ctx` plus `render.label`.
Divergence points live in one `shares` state key (`"6,4,6"`), which the
framework's deep-link sync mirrors into the URL. Headless hooks: `?shares=6,4,6`
(the stand-in for a handle drag), `?hover=x,y`, plus the usual `?step=N` /
`?play=1` / `?theme=`. Source: [`page.js`](page.js).
