# parallelism -- what actually crosses the wire when one model is split across GPUs

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: four different things get called "parallelism" when one model
is served across several GPUs, and readers conflate them. They are not variants
of each other -- they move different bytes, at different moments, for different
reasons. One transformer layer is drawn across an N-GPU strip and you choose a
strategy **per sublayer**, watching what leaves each device. **Anchor**: A4 KV
cache / runtime shape (Family J, serving time). Companion to
[moe-routing](../moe-routing/README.md), which is the router this page's
all-to-all obeys, [mla](../mla/README.md), which is the latent cache that makes
tensor-parallel attention a bad trade,
[disaggregation](../disaggregation/README.md), which splits a fleet by *phase*
rather than a model by *tensor*.

## What it shows

| Strategy | What is split | What crosses the wire | The trade |
|---|---|---|---|
| **Tensor parallel** | every matmul, across all N | an **all-reduce** of activations, once per sublayer, in every layer | lowest latency, but it wants a fat link -- the collective is on the critical path of every sublayer, so it degrades once the group spans a slower fabric |
| **Pipeline parallel** | whole **layers** | one **point-to-point** activation send per stage boundary | the cheapest comms on the page and it does not grow with layer count -- but the pipeline **bubbles**, and a single decode token can only be in one stage, so it does nothing for single-token latency |
| **Expert parallel** | the MoE **experts** | an **all-to-all of TOKENS**, out to their experts and back | expert memory scales with GPU count, but the volume is set by the **router**, not the topology: it is load-imbalanced by construction and the step waits for the busiest rank |
| **Data-parallel attention** | nothing -- attention is **replicated**; each rank owns its own requests and KV | **nothing at all**, for attention | N copies of the attention weights, and the ranks must still line up at the MoE boundary -- a rank with no requests of its own runs a **dummy forward pass** so the collective does not deadlock |

Two counters recompute live, per rank, from the controls -- nothing on the page
is annotated:

- **bytes crossing the wire per decode step** -- split into the attention half
 and the MoE half, given both fleet-wide and per GPU, and converted to
 milliseconds against the link speed you set. A ring all-reduce of a `P`-byte
 payload is charged `2(N−1)/N × P` out of every GPU; a stage boundary is
 charged one `P`; an all-to-all is charged from the routing matrix.
- **per-GPU memory** -- attention weights, expert weights, KV cache
 workspace, stacked against a device budget you set. Whatever crosses the
 budget line is hatched, so a configuration that cannot be deployed says so.

Two cliffs are worth finding on purpose, because they are the reason real
deployments pair DP-attention with expert parallelism rather than tensor-
parallelling everything:

- **A tensor-parallel KV cache only shards as far as it has KV heads to give
 away.** Switch the cache to its latent (single-head) form and tensor parallel
 has nothing to split at all -- every rank keeps a whole copy of the very cache
 the latent form exists to shrink. Data-parallel attention shards it by
 *request ownership* instead, and crosses nothing.
- **Replication is a memory bill.** DP attention duplicates the attention
 weights N times; the readout reports the resident multiple, and on a small
 device the strip goes over budget.

You are meant to be able to build a **bad** configuration and see why it is bad:
a two-GPU tensor-parallel split of a latent cache on a long context, a five-stage
pipeline at one micro-batch, a heavily skewed router over few ranks.

**Out of scope, in one line:** *context* parallelism -- splitting a single
**sequence** across ranks so one long prompt's attention is computed
cooperatively -- is a different axis from all four of these and is not drawn
here.

## Render tier

T1 (Canvas2D throughout -- the lesson is topology and volume, not pixel scale).

## Wiring

`layout.mount` + controls (`gpus`, `attn`, `moe`, `kv`, `d`, `layers`,
`experts`, `topk`, `reqs`, `seq`, `skew`, `micro`, `link`, `budget`,
`overhead`) + an 8-stage `Transport` walking one layer's sublayers in order
(input → attention math → attention collective → router → dispatch → expert math
→ combine → output) so the collectives animate where they happen; `onPointer`
drags the strip horizontally for the GPU count and vertically for the router
skew, and clicks a strategy chip to switch that sublayer; hover-to-inspect on
every GPU (what it holds and why) and every arrow (what is crossing, with the
formula); three challenges; and an A/B compare over tensor- versus
data-parallel attention. URL hooks: `?step`, `?gpus`, `?attn`, `?moe`, `?kv`,
`?d`, `?layers`, `?experts`, `?topk`, `?reqs`, `?seq`, `?skew`, `?micro`,
`?link`, `?budget`, `?overhead`, `?hover=x,y`.
Source: [`page.js`](page.js).

## Sources

The mechanisms and the DP-attention-plus-expert-parallel pairing follow the
public write-ups: the vLLM parallelism-and-scaling serving documentation, Meta's
engineering post on tensor / context / expert parallelism for LLM inference,
the vLLM large-scale-serving ("wide expert parallelism") post. The model
dimensions on the page are yours to set; the defaults are round numbers chosen
to make the shapes legible, not a description of any particular deployment.
