# ring-attention -- one sequence too long for any single device, attended to without any device ever holding all of it

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: shard the sequence across N devices. Each device keeps its own
**queries** permanently; the **KV blocks rotate** around the ring. At each of N
steps every device attends its local queries against whichever KV block it is
currently holding, folds the result into a running online-softmax state (max
`m`, sum `l`, accumulator `O`), and passes the block on to its neighbour. After
N steps every query has met every key -- and no device ever stored more than a
couple of blocks at a time.

It is [flash-attention](../flash-attention/README.md)'s online softmax with the
**tiling distributed across machines** instead of across on-chip SRAM: the same
`rescale = exp(m_old − m_new)` recurrence, the same never-materialize-the-N×N
guarantee, one ring hop where flash-attention has an SRAM tile load.

**Not the same axis as [parallelism](../parallelism/README.md).** That page
splits a **model** -- tensors, layers, experts -- across devices, and the
sequence is whole on every rank. This page splits the **sequence** itself,
the model is whole on every device. They compose; they are not variants of each
other.

## The point is the overlap

The send of the next KV block is **issued while the current block's attention is
still computing**. That single fact decides whether the ring scales:

| | grows as | consequence |
|---|---|---|
| attention compute per block | **block²** (`4 · block² · width` FLOP) | doubling the block quadruples the compute |
| KV block on the wire | **block** (`2 · block · width · 2 B`) | doubling the block only doubles the bytes |

So a **big enough block hides the wire completely** -- the transfer finishes
inside the compute window and the ring costs `N × compute`. A block that is too
small, or a link that is too slow, leaves every device **idle waiting on bytes**,
and the step costs `N × transfer` instead. Both regimes are reachable from the
controls, and the stall is drawn as **real idle time** on the per-device
timeline (hatched, per step, per device) rather than asserted in prose.

The counter-intuitive half is that adding devices at a fixed sequence length
makes the blocks *smaller* -- which is good for memory and bad for overlap.
That trade is the page.

## Numbers, computed live from your settings

Nothing on the page is annotated; every figure recomputes from the sequence
length, device count, model width, link bandwidth and device throughput you set.

- **per-device memory** -- two KV block buffers (the one being attended and the
 one arriving) plus the local queries plus the fp32 accumulator. **This is the
 headline: it is set by the BLOCK, not by the sequence.** Grow the sequence
 add devices in step and the number does not move.
- **bytes crossing the ring per step** -- per wire and ring-wide.
- **compute time vs transfer time per block**, and the verdict: hidden, or
 stalled by a stated number of milliseconds.
- **total idle** -- absolute and as a percentage of wall time.

**Baseline: single-device attention**, which needs the whole KV resident. It is
reported as a **memory requirement, not a runtime** -- past some sequence length
it does not fit on one device at all, and quoting a time for a run that cannot
start would be a fiction. The bar for it is hatched for exactly that reason.

**Scope of the model, stated so the arithmetic can be checked**: one attention
layer, all heads, one micro-batch, KV in 16-bit, non-causal. A causal mask would
leave half the block pairs empty and turn the rotation into a load-balancing
problem -- a real and separate topic, not drawn here. The device throughput
link bandwidth are yours to set; the defaults are round numbers chosen to make
both regimes reachable, not a description of any particular machine.

## Two layers of number, deliberately different scales

The **accumulator values** (`m`, `l`, `O`, the rescale factor) are a **real
online softmax** run in-page over a toy shard -- 3 queries, 3 keys, 4 dims per
device, seeded so every reload shows the same picture. Hover a device and the
numbers you see were computed, not authored. The **bytes and milliseconds** come
from the full-scale settings above. The page never mixes them.

## Render tier

T1 (Canvas2D throughout -- the lesson is topology, packing and volume, not pixel
scale).

## Wiring

`layout.mount` + controls (`n`, `seq`, `link`, `flops`, `hidden`, `seed`) + an
N-step `Transport` (autoplay + loop) walking the rotation, so the blocks visibly
move around the ring; an ambient animation clock carries the in-flight block
along each wire between steps. `onPointer` drags the **ring** horizontally for
the device count and vertically for the sequence length, and the **timeline**
horizontally for link bandwidth -- the packing re-flows and the stalls appear or
vanish under your hand. Hover-to-inspect on every device (what it holds, which
queries it owns, its running max/sum/accumulator), every wire (the bytes in
flight and whether they are hidden), and every timeline bar (that step's compute
/ transfer / idle). Three challenges: hide the communication, then break it, then
hold a 256K-token sequence under 512 MB per device. URL hooks: `?step`, `?n`,
`?seq`, `?link`, `?flops`, `?hidden`, `?seed`, `?hover=x,y`, `?play=1`.
Source: [`page.js`](page.js).

## Sources

Liu, Zaharia and Abbeel, *Ring Attention with Blockwise Transformers for
Near-Infinite Context*, <https://arxiv.org/abs/2310.01889> -- the rotation
schedule and the overlap argument. The online-softmax recurrence it distributes
is Dao et al.'s FlashAttention, <https://arxiv.org/abs/2205.14135>, which the
[flash-attention](../flash-attention/README.md) page draws at the SRAM-tile
scale.
