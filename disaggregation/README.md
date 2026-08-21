# disaggregation -- prefill pool, decode pool, KV cache on the wire

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **deployment** consequence of the two inference regimes.
Prefill and decode want opposite machines, so a serving fleet can stop asking
one machine to do both -- run a **prefill pool** and a **decode pool**, and ship
each request's KV cache from one to the other. **Anchor**: A4 KV cache / runtime
shape (Family J, serving time). Companion to
[prefill-vs-decode](../prefill-vs-decode/README.md), which shows *why* the two
regimes have different shapes, and to
[chunked-prefill](../chunked-prefill/README.md)
[continuous-batching](../continuous-batching/README.md), which mitigate the same
interference **inside** a single pool.

## What it shows

- **The two phases want opposite hardware.** Prefill is one big parallel GEMM
 over the whole prompt: the math units saturate and the cost scales with
 prompt length (compute-bound). Decode is one skinny step per token that
 re-reads the weights and the whole cache: the math units idle and the cost is
 set by memory bandwidth. On this page a prefill machine runs **one** prompt at
 a time, while a decode machine advances **a whole batch** in one step --
 because when you are bandwidth-bound the weight read is paid once for the
 batch.
- **Why sharing one pool hurts.** The lower timeline is the baseline: every
 machine does both phases. The moment a prompt lands on a machine, that
 machine goes to the prefill and its resident batch emits *nothing* until the
 prefill ends -- the hatched gaps. That is head-of-line blocking, and it is
 what welds time-to-first-token and per-output-token latency into one number.
- **The split.** The upper timeline runs the identical workload on a prefill
 pool plus a decode pool, each sized independently. No prefill can ever land on
 a decode machine, so nothing stalls mid-generation.
- **The new cost, charged honestly.** Between the two pools is a link, and the
 KV cache is real bytes on it: K and V for every layer for every prompt token
 (this page prices it at 0.5 MB per token). The transfer is proportional to
 prompt length, it is drawn as its own timeline lane and as its own segment of
 each request's latency, and it lands **after** the first token (which the
 prefill machine emits) and **before** all the rest -- so it is charged to
 inter-token latency, not to time-to-first-token. Throttle the link and the
 page will tell you, in the readout, that disaggregation has become a loss.

Both deployments are simulated in-page tick by tick from the same request list,
so every counter, tooltip and readout number is derived rather than annotated.
The readout reports throughput, TTFT, TPOT and end-to-end latency for both, each
ratio as a percent of the shared-pool baseline with its direction named.

**The sweet spot is measured, not asserted.** The strip above the allocation bar
re-simulates the disaggregated deployment at *every* possible split and plots
the resulting end-to-end latency, with the best one marked. Both failure modes
are the two tall ends: too few prefill machines and prompts queue for compute;
too few decode machines and prefilled requests queue for a slot.

The honest headline is the one the production runtimes give: disaggregation buys
**latency decoupling** -- a first-token target and a per-token target that can be
hit independently -- and not automatically raw throughput, because a shared pool
can always put every machine on whichever phase is short of capacity.

## Interaction

Press play (or scrub) and both deployments advance together against one shared
time axis. **Drag** the allocation bar to move a machine between the pools, or
drag the interconnect pipe up and down to widen or throttle the link -- the
simulation, the sweep and every counter re-run under your hand. **Hover** a
request, on either timeline or in the per-request breakdown strip, for its phase
breakdown with the KV bytes and the transfer time; hover a worker for what it is
doing and, when it is idle, which pool starved it; hover the sweep strip for the
latency of every split.

## Render tier

T1 (Canvas2D throughout -- the lesson is schedule and dataflow geometry, not
pixel scale).

## Wiring

`layout.mount` + controls (`M`, `split`, `bw`, `nreq`, `gap`, `plen`, `seed`),
a `max(makespan)`-tick `Transport` both timelines read, `onPointer` handles for
the allocation-bar and pipe drags, hover-to-inspect, two challenges (land on the
sweet spot; throttle the link until it loses), and an A/B compare over the split.
URL hooks: `?step`, `?M`, `?split`, `?bw`, `?nreq`, `?gap`, `?plen`, `?seed`,
`?hover=x,y` (the headless stand-in for a drag or a hover).
Source: [`page.js`](page.js).
