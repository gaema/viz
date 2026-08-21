# prefix-routing -- the load balancer that knows which replica holds your prefix

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a serving runtime picks **which replica** serves a
request when several replicas of one model each keep their own prefix cache --
and why a router that only follows the cache is worse than round-robin.

**Anchor**: A4 KV cache / runtime shape (Family J, serving time).

## The layer this page sits on

[radix-attention](../radix-attention/README.md) shows prefix reuse **inside one
replica**: two requests that begin with the same tokens share one KV path in a
radix tree, so the second one prefills only its suffix. This page is the layer
**above** it -- reuse **across replicas**, where the router has to choose.

The reuse radix-attention buys is per replica, so it only exists if the request
arrives at the replica that already holds the prefix. Round-robin does not know
that: a request whose long system prompt is already computed on replica 2 lands
on replica 3 and recomputes every block of it, and the tree on replica 2 ages
out unused. A prefix-aware router hashes the prompt's leading blocks and sends
the request where the matched prefix is longest. Same mechanism, one level up.

## What it shows

A seeded stream of requests arriving at N replicas. Each request is a run of
fixed-size blocks: the leading blocks are a shared **system prompt** drawn from
a skewed popularity distribution, the trailing blocks are that request's own
text and are shared with nobody. Each replica has a FIFO queue and an LRU block
cache. Three panels:

- **Replica cards** -- live queue depth, utilisation, and which system-prompt
 prefixes that replica currently holds (one bar per prompt, height = how many
 of its blocks are cached).
- **Arrival lanes** -- one lane per replica, one dot per request in arrival
 order, coloured by system prompt. Fill = the fraction of the shared prefix
 that was served from cache; an orange ring means nothing was reused and the
 whole prompt was prefilled.
- **The trade-off chart** -- the same arrival stream re-simulated at 21 router
 weights, from pure load balance to pure cache affinity. Cache hit rate (green)
 and load balance (violet) cross; mean time-to-first-token relative to
 round-robin (dashed) bottoms out near the crossing, with a ★ on the best
 weight in the sweep.

## The failure, which is the point

The routing score is

```
score(replica) = w · matched-prefix-fraction − (1 − w) · relative-backlog
```

and `w` is a knob the reader drags. Both ends are reachable and both are bad:

- **w = 0, pure load balance.** Requests scatter, so every replica sees every
 system prompt, and with a finite cache they evict each other's prefixes.
 Balance is excellent, hit rate collapses, and the prefill nobody avoided shows
 up in latency.
- **w = 1, pure cache affinity.** This is a *pinning* function. Every request
 that shares one popular system prompt scores highest on the single replica
 that cached it, so they all queue there while the siblings sit idle. Hit rate
 is the best on the page and throughput is the worst on the page -- at a high
 popularity skew a replica can end the run having served **nothing**.
- **In between**, a router that will break affinity for a busy replica beats
 both, by a wide margin.

The **popularity skew** is the second draggable, and it is what decides whether
affinity is worth anything at all: with many equally popular distinct prompts
there is no hot prefix to follow, the green curve flattens, and affinity only
costs balance. With one dominant prompt affinity pays the most *and* pins the
hardest.

Every number -- hit rate, mean TTFT, tail (p95) TTFT, throughput, per-replica
utilisation -- comes out of the simulated run and is recomputed from scratch
whenever a control moves. Ratios are reported as a percent of the **round-robin
baseline**, which is the same arrival stream routed `k mod N` ("TTFT 71% of
round-robin", lower is better; "throughput 103% of round-robin", higher is
better). Nothing on the page is annotated.

## Sources for the mechanism

Both public:

- **SGLang** -- RadixAttention plus its cache-aware load balancer across
 workers: *Efficiently Programming Large Language Models using SGLang*,
 <https://arxiv.org/abs/2312.07104>
- **vLLM production-stack** -- the KV-cache-aware routing logic in its router:
 <https://docs.vllm.ai/projects/production-stack/en/latest/>

## Render tier

T1 (Canvas2D throughout -- the lesson is a scheduling decision, not pixel scale).

## Wiring

`layout.mount` + controls (`replicas`, `w`, `skew`, `prompts`, `capacity`,
`arrivals`, `seed`) + a one-step-per-arrival `Transport` (autoplay, loop),
`onPointer` drag on the ◆ weight marker and the ▲ skew handle,
hover-to-inspect on replica cards / arrival dots / the chart / the popularity
histogram, three challenges, and an A/B compare over `w = 0` against `w = 1`.
URL hooks: every control (`?w`, `?skew`, `?replicas`, `?prompts`, `?capacity`,
`?arrivals`, `?seed`) plus `?step=N`, `?hover=x,y`, `?play=1` -- seeded, so one
URL replays exactly one arrival stream at exactly one router setting.
Source: [`page.js`](page.js).
