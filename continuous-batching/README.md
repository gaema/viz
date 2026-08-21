# continuous-batching -- refill the slot the moment a sequence stops

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a serving runtime schedules a batch of generation
requests, and why it admits a waiting request the instant a slot frees instead
of at the end of the batch. **Anchor**: A4 KV cache / runtime shape (Family J,
serving time). Companion to
[prefill-vs-decode](../prefill-vs-decode/README.md), which shows the two regimes
this page schedules, and [kv-cache](../kv-cache/README.md), which shows what a
slot actually holds.

## What it shows

Two Gantt charts of the **same workload** -- same arrivals, same output lengths,
same number of batch slots -- sharing one x axis (x = decode step, y = batch
slot, one coloured bar per request):

- **Static batching**: the batch is formed once and every slot in it is held
 until the **longest** sequence in that batch finishes. A sequence that emits
 its stop token early keeps its slot and emits padding, and nothing may join a
 batch in flight. Those are the hatched cells -- slot-steps the runtime paid
 for and got nothing back.
- **Continuous batching**: the scheduler re-admits every step. The moment a
 sequence stops, the head of the queue takes that slot mid-flight, so the
 hatched dead area mostly disappears and the queue drains sooner.

Three counters recompute live for each scheduler: **steps to drain the queue**,
**slot utilisation** (busy slot-steps ÷ slots × steps) and **per-request
latency** split into wait + generate. A queue-depth strip under each timeline
shows the backlog draining, and a per-request latency strip at the bottom makes
the honest tradeoff visible: the *generate* segment is identical under both
schedulers. Continuous batching buys throughput and queueing delay -- it does
not make any single sequence produce one token faster.

The schedules are really simulated in-page from the request list, so every
number in the readout and every tooltip is derived, not annotated.

Press play (or scrub) and both schedulers advance together, so the step where
they diverge is visible; **drag** any bar sideways to restretch that request's
output length -- one long sequence is ruinous on the static timeline and nearly
free on the continuous one; **hover** a bar for its arrival / wait / output
length / occupied steps, or a hatched cell for which sequence that slot is
waiting on.

## Render tier

T1 (Canvas2D throughout -- the lesson is schedule geometry, not pixel scale).

## Wiring

`layout.mount` + controls (`slots`, `nreq`, `gap`, `seed`, and a `lens`
free-text field holding per-request output-length overrides) + a
`max(makespan)`-step `Transport` that both timelines read, `onPointer` bar-drag
for the output length, hover-to-inspect, two challenges, and an A/B compare over
batch-slot count. URL hooks: `?step`, `?slots`, `?nreq`, `?gap`, `?seed`,
`?lens=id:len,id:len` (the headless stand-in for a drag), `?hover=x,y`.
Source: [`page.js`](page.js).
