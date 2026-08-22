# context-parallelism -- pass the KV, or pass the queries?

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **serving decision** that sits on top of a sequence-sharded
attention. The sharding, the blocks rotating around a ring, the online-softmax
accumulation and the compute-vs-transfer overlap are the mechanism -- and they
are the neighbouring [ring-attention](../ring-attention/index.html) page's
subject, not re-taught here. This page asks the one question that mechanism
leaves open:

> Every (query, key) pair has to meet, and exactly one of the two operands has to
> travel to make that happen. **Which one do you move?**

**Pass-KV** circulates the keys and values and keeps each device's queries
pinned. **Pass-Q** circulates the query blocks against resident KV. Both compute
the **same attention output** -- the arithmetic is identical and neither is an
approximation of the other. Only the traffic differs, and which one is cheaper
flips between phases. **Anchor**: A4 KV cache / runtime shape (Family J,
serving-time mechanisms).

## What it shows

- **The two payloads, priced.** Every byte figure is computed live from the head
 counts, sequence length, device count and phase you set -- nothing is a stored
 constant, so each hover tooltip's arithmetic can be checked by hand.
 - *pass-KV* per hop = `shard × 2 (K and V) × 2 B × H_kv × d`. It is **flat** in
 how many queries are live: the resident cache shard moves whatever else is
 happening.
 - *pass-Q* per hop = `q × H_q × (2·d [Q] + 4·d [partial output] + 8 [the two
 softmax statistics])`. It is **linear** in the live query count, because a
 query that moves has to take its unfinished answer with it.
- **The curves CROSS, and the crossing is draggable.** On a log-log plot of
 bytes-per-hop against query tokens live per device, a flat line and a linear
 one meet at
 `q* = (shard × 2 × 2 × H_kv × d) / (H_q × (2·d + 4·d + 8))`.
 **Drag the ✕ itself** and the page re-solves that expression for the KV-head
 count that would put the crossing where you dropped it, snapping to a grouping
 that divides the query heads evenly -- which is precisely why grouped-query
 attention changes this decision at all.
- **The phase decides which side of the crossing you are on.** Prefill puts a
 whole prompt shard of queries in flight at once; decode puts one query token
 per sequence against a cache holding the entire context. Those land on
 opposite sides of `q*`, so a real system picks differently per phase. The
 verdict card names the winner and says how far you are from the crossing.
- **The comparison is reported with its direction stated.** pass-Q's traffic is
 given as a **percent of pass-KV's** on a lower-is-better axis (`100%` =
 parity, `>100%` = pass-Q moves *more*), never as a bare multiplier.
- **Persisting the shards removes the flip.** Turn on *KV shards persist*
 only the newly produced K,V has to move each step. Both costs then become
 linear in the query count, the curves go parallel, they never cross, and one
 pattern wins at every operating point -- which is the honest reason a system
 would rather place the cache once than re-send it.

Head dimension is held fixed at 128 so the head *counts* stay the story; Q, K
V are 16-bit and the circulating partial output and its softmax statistics are
fp32, as an online softmax requires.

The bytes are an exact consequence of the stated shapes and dtypes. They are not
a measurement of any particular interconnect, engine or accelerator -- there is
no wire time on this page, only what has to cross it.

## Two URLs, opposite verdicts

Same model, same devices, same grouping; only the phase differs:

| | URL | Verdict |
|---|---|---|
| 🔵 | `index.html?phase=prefill&seqPow=17&devices=8&qheads=32&kvheads=8` | **pass-KV** -- 64.0 MiB/hop vs pass-Q 388.0 MiB/hop |
| 🟠 | `index.html?phase=decode&seqPow=17&devices=8&qheads=32&kvheads=8&batch=1` | **pass-Q** -- 24.3 KiB/hop vs pass-KV 64.0 MiB/hop |

## Interactions

Press play (or scrub) to step the exchange hop by hop -- the ring draws whichever
operand is the one that moves under the current pattern, and the other stays
pinned in its box. **Drag** anywhere on the plot to move the operating point;
**drag the ✕** to re-solve the crossing for a KV-head grouping. **Hover** a
device for what it holds, what it is sending and to whom; hover either curve for
its byte arithmetic term by term; hover the ✕ for the crossing expression. The
steppers and sliders resize the problem live (sequence length, devices, query
heads, KV heads, decode batch), and the phase select flips the verdict.

## Render tier

T1 (Canvas2D throughout: the device ring, the log-log cost plot, the verdict
card).

## Wiring

`layout.mount` + controls (sequence length, devices, query heads, KV heads,
phase, decode batch, shard persistence, which pattern the ring draws) + a
Transport over the ring hops (autoplay + loop) + `onPointer` drags for the
operating point and the crossing + hover-to-inspect + two challenges + an A/B
compare across the two phases. URL hooks: every control is mirrored into the
query string (`?phase=` `?seqPow=` `?devices=` `?qheads=` `?kvheads=` `?batch=`
`?persist=` `?show=`), plus `?step=N`, `?opq=N` and `?crossq=N` (headless
stand-ins for the two drags), `?hover=x,y` and `?play=1`. Source:
[`page.js`](page.js).

## Sources

- Liu, Zaharia, Abbeel, *Ring Attention with Blockwise Transformers for
 Near-Infinite Context* -- https://arxiv.org/abs/2310.01889
- Liu et al., *World Model on Million-Length Video and Language with Blockwise
 RingAttention* -- https://arxiv.org/abs/2402.08268
- Grattafiori et al., *The Llama 3 Herd of Models* --
 https://arxiv.org/abs/2407.21783 -- describes long-context serving that
 gathers the key/value tensors rather than the queries, on the grounds that
 grouped-query attention makes K,V much smaller than Q.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from
 Multi-Head Checkpoints* -- https://arxiv.org/abs/2305.13245
- Brandon et al., *Striped Attention: Faster Ring Attention for Causal
 Transformers* -- https://arxiv.org/abs/2311.09431
