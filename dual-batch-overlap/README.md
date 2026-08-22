# dual-batch-overlap -- hiding an expert all-to-all behind another microbatch

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: a mixture-of-experts layer served with **expert parallelism**
has to route. Every token is sent to whichever device holds the expert it
picked, and the results come back -- two all-to-all collectives per layer,
while they are in flight the arithmetic units have nothing to do. Split the
batch in two and interleave the halves: while microbatch A is in its all-to-all,
microbatch B computes, and vice versa. With balanced halves the communication
disappears into the arithmetic entirely; with unbalanced halves a visible bubble
remains. **Anchor**: A4 KV cache / runtime shape (Family J, serving time).

**Scope.** [parallelism](../parallelism/README.md) owns *what* expert
parallelism puts on the wire and why the volume is set by the router rather
than the topology. This page does not re-derive any of that: it is about
**hiding that cost in time**, and the communication cost here is a single
duration you set with a slider or drag directly. Sibling scheduling pages:
[chunked-prefill](../chunked-prefill/README.md) packs one queue's work into
steps; [continuous-batching](../continuous-batching/README.md) packs many
requests into one step; this one packs **two lanes** against each other.

## What it shows

- **Two lanes, one scheduler.** A compute lane and a communication lane, each a
 single serial resource, with real dependency edges between a microbatch's
 phases: attention + router → all-to-all dispatch → expert FFN → all-to-all
 combine → the next layer. Blocks are placed by a list scheduler
 (`schedule` in [`page.js`](page.js)), each task starting at
 `max(predecessor end, lane free)`.
- **The bubble is a consequence, not a drawing.** Step time, overlap fraction
 and bubble are read off the finished schedule. Nothing is computed from a
 closed-form expression, so a configuration where splitting *hurts* falls out
 on its own rather than being asserted.
- **A named baseline.** Every number is quoted as a percent of **no overlap,
 one batch (M = 1)** -- which is the same simulator with a single microbatch,
 so nothing can overlap and the compute lane idles through both all-to-alls.
 Step time is lower-is-better, so >100% of the baseline is **worse**.
- **Both regimes are reachable, and the failure is the interesting one.**
 Overlap pays only when a microbatch's compute is at least as long as the
 communication it has to hide. Each split halves that compute *and* pays the
 fixed per-invocation costs again -- a launch on the compute lane, a link
 latency on the communication lane -- so past a point splitting makes the step
 **slower than not splitting at all**. The split chart re-simulates all three
 microbatch counts and draws them against the baseline rule, so the point where
 4 loses to 2, and where 4 loses to 1, is something you find rather than read.

The millisecond figures come from a deliberately simple illustrative cost model
-- a per-microbatch share of the work plus a fixed per-invocation overhead on
each lane -- chosen so the *shape* of the trade is readable. They illustrate the
mechanism; they are not a measurement of any particular model, interconnect or
accelerator.

## Where the mechanism comes from (public sources)

| Source | What it contributes |
|---|---|
| DeepSeek-V3 Technical Report, <https://arxiv.org/abs/2412.19437> | The DualPipe schedule and its computation-communication overlap: dispatch/combine all-to-all of one chunk hidden behind another chunk's compute |
| MegaScale, <https://arxiv.org/abs/2402.15627> | Overlapping collectives with computation at scale, and what stops it being free |
| Lina, <https://arxiv.org/abs/2210.17223> | MoE serving work that decomposes and pipelines the all-to-all against expert compute |
| Tutel, <https://arxiv.org/abs/2206.03382> | Adaptive MoE dispatch, including splitting the all-to-all so it can be pipelined |

## Interactions

| Interaction | What it does |
|---|---|
| **Transport** | Play / pause / step / scrub the schedule; phases fill in on both lanes in the order the scheduler placed them. Autoplays and loops. |
| **Drag a compute block** | Stretches the compute cost -- the whole lane re-packs under your hand and the bubble grows or vanishes. |
| **Drag a comm block** | Stretches the communication cost, the same way. |
| **Drag across the split chart** | Re-splits the batch (1, 2 or 4 microbatches); each bar is a complete re-simulation at that count. |
| **Hover any phase block** | Its duration, what it waits on (its own previous phase, or the lane being busy), what it overlaps on the other lane, and how much of a transfer was actually hidden. Hover a bubble for what was still in flight. |
| **Resize the problem** | Layers (1-3) and microbatch count, live. |
| **Challenges** | Get the overlap to 100%; then find a setting where four microbatches are slower than not splitting at all. |

URL hooks: every control is mirrored into the query string (`?C=` compute ms,
`?V=` all-to-all ms, `?M=` microbatches, `?L=` layers), plus `?step=N`,
`?hover=x,y` and `?play=1` -- so a dragged state is reproducible headlessly.

Two worth opening directly:

- Append `?C=6&V=4&M=2&L=2` to the page URL -- overlap pays: **68.8% of
 the no-overlap one-batch baseline** (lower is better), 87.5% of the
 communication hidden, 1.34 ms residual bubble.
- Append `?C=1&V=1&M=4&L=2` -- **splitting makes it
 worse**: 121.9% of the same baseline, because each microbatch's arithmetic is
 now too small to cover anything while the fixed launch and link costs are paid
 eight times each.

## Render tier

T1 (Canvas2D throughout: two lane strips, a baseline ghost strip, a bubble
hatch, and a re-simulated split chart).

## Wiring

`layout.mount` + controls (compute cost, all-to-all cost, microbatches,
layers) + a Transport over the scheduled tasks (autoplay + loop) + `onPointer`
block-stretch and split-chart drag + hover-to-inspect + two challenges + an A/B
compare between one batch and two microbatches. Source: [`page.js`](page.js).
