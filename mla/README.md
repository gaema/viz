# mla -- Multi-head Latent Attention (cache one latent, not K and V per head)

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: why caching a single small **latent** per token, instead of a
full key and value for every head, is what makes long context affordable --
what it costs. **Anchor**: A2 head_dim layout / A4 KV cache (Family B).

## Where it sits

[`kv-cache`](../kv-cache/README.md) sets up the problem: the KV cache is the one
term that grows with context, so at long context it, not the weights, is what
you run out of. [`gqa-mqa`](../gqa-mqa/README.md) gives the earlier answer --
**sharing**: let several query heads read one key/value head, and the cache
shrinks by the sharing ratio. This page gives the next answer --
**compression**: stop caching per-head K and V at all, cache one low-rank latent
per token and rebuild the per-head K,V from it whenever attention needs them.
Sharing is bounded by the head count; compression is bounded by the latent rank
you are willing to pay arithmetic for, which is a different (and much looser)
budget.

## What it shows

**1. The baseline.** Standard multi-head attention caches K and V for every
head: `2 · n_h · d_h` elements per token per layer. At the page's default shape
(`n_h = 128`, `d_h = 128`) that is 32,768 elements per token per layer -- and it
is charged again for every layer and every token in the context.

**2. The compression.** MLA down-projects the token to one latent `c_KV` of
width `d_c` and caches that, plus a small decoupled RoPE key `k_rope` of width
`d_R`: `d_c + d_R` elements per token per layer. At `d_c = 512`, `d_R = 64` that
is 576 -- **1.76% of the MHA footprint, 56.9× smaller**. The two per-token
footprints are drawn as squares whose **area is proportional to the element
count under one shared unit**, so the collapse is a picture rather than a
number, and the decode tape below draws each appended token at the same scale
(a full-height block for MHA, a hairline for MLA).

**3. The catch, stated on the page.** The latent has to become per-head K and V
again before attention can use it. At decode that re-expansion is normally
*absorbed* into the query and output projections, which then run at **latent
width `d_c` instead of head width `d_h`** -- a bigger matmul on every step. The
page prices both sides per token per layer:

| Side | Formula | At the defaults |
|---|---|---|
| MHA projections | `4 · d · n_h · d_h` (Q, K, V, output) | 469.8M MAC |
| MLA projections | `2 · d · n_h · d_c + d · (d_c + d_R)` (absorbed Q and output at latent width, plus the two down-projections) | 943.7M MAC |

That is **201% of MHA's projection arithmetic** (lower is better; 100% =
parity) bought with **1.76% of its cache**. The "the trade" panel shows both
bars, and dragging the latent dimension moves them in opposite directions under
one hand -- which is the actual shape of the design decision. Note that `d_c` is
a **rank budget**, not a free knob: drive it far below the production value
the arithmetic does drop below MHA's, but so does how much of K and V the latent
can still represent. The page says so on the rail's tooltip rather than letting
the bars imply a free lunch.

**4. Why it matters where it matters.** The weights are a fixed cost; the cache
is the term that scales with context. The growth panel plots total cache against
context length for both schemes: two straight lines out of the origin whose
slopes differ by the compression factor, so the absolute gap keeps widening the
longer the context gets. At 8K tokens over 61 layers in fp16 that is 30.50 GB
against 549.0 MB.

Published figures for the shape the defaults use (DeepSeek-V2/V3/R1
hyperparameters: `kv_lora_rank = 512`, `qk_rope_head_dim = 64`, 128 heads,
`qk_nope_head_dim = 128`) are reported in the DeepSeek-V2 paper
(arXiv:2405.04434): a 56.9× element-count reduction versus the multi-head
predecessor, and 5.76× higher generation throughput at equivalent batch sizes on
the authors' hardware. Those are the paper's reported numbers, not measurements
taken here; everything the page displays is computed live in the browser from
the control values.

## Render tier

T1 (Canvas2D -- to-scale area blocks, a flow chain, tapes, bars and a line
plot).

## Wiring

`layout.mount` + controls (`latent dim d_c`, `decoupled RoPE dim d_R`,
`attention heads`, `head_dim`, `hidden dim`, `layers`, `context`, `cache dtype`,
`decode steps`) + a per-token decode `Transport` built in `page.js`, drawn with
`ctx` + `render.label`. Direct-manipulable per the framework interaction
contract:

- **Transport** -- each step decodes one token and appends its cache entry to
 both tapes; auto-plays and loops.
- **Drag** -- the latent-dimension rail, or sideways on the MLA footprint square
 itself (its area *is* `d_c + d_R` drawn to scale). Cached bytes and the
 absorbed projection matmul move in opposite directions as you drag.
- **Hover** -- every cached block, projection block, tape row and bar reports
 what it is, its shape, and the formula its size came from with the current
 numbers substituted.
- **Steppers** resize the problem (heads / head_dim / hidden / layers /
 context / decode steps).
- **A/B compare** contrasts a barely-compressed latent against an aggressive
 one.
- **Challenge mode** has two goals that deliberately pull against each other:
 get the cache under 2% of MHA, then get the arithmetic back under 150% of it.

Headless hooks: `?step=N` / `?play=1`, `?hover=x,y` (canvas px), and a URL hook
for every control handle -- `?dc=`, `?dR=`, `?heads=`, `?hdim=`, `?hidden=`,
`?layers=`, `?ctxk=` (context in units of 1024 tokens), `?ntok=`, `?kvdtype=`.
Any state the page can be dragged into is therefore reproducible without a
pointer. Source: [`page.js`](page.js).
