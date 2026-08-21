# paged-attention -- a KV cache in blocks, not slabs

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: why a serving runtime stores the KV cache in fixed-size
**blocks** with a per-sequence **block table**, instead of one contiguous slab
per sequence -- and how two sequences that start with the same prompt end up
pointing at the same physical blocks.
**Anchor**: J1 serving-time KV-cache memory management (the paged-attention /
block-table idea popularized by vLLM), sitting on top of the
[`kv-cache`](../kv-cache/README.md) page's per-token cache math.

## What it shows

Two strips over the **same 96-cell memory pool**, so the counter under each is
directly comparable.

- **Contiguous.** Every sequence reserves one slab the size of the longest
 answer it might produce. Cells past its current length are hatched: reserved
 and dead. The leftover at the end of the pool is hatched too when it is
 smaller than one reservation -- it cannot host another sequence, which is
 fragmentation you can point at. A sequence that outgrows its slab is flagged:
 a full slab has nowhere to grow.
- **Paged.** The same sequences in fixed-size blocks handed out from a free
 list, so they are **not adjacent**, and only the **last** block of each
 sequence is partly empty. Below, one **block table** per sequence maps logical
 block → physical block, with an arrow into the block it names.
- **Shared blocks.** Two of the sequences are answers to the same prompt, so
 their first block-table entries point at the **same** physical blocks (marked
 `⇄`, with both arrows converging). Only whole blocks can be shared; the block
 where the two sequences first diverge is private to each -- copy-on-write.
- The counter that matters, live under each strip: **wasted cells / total
 cells**, split into reserved-but-unused vs fragmented tail on the contiguous
 side, and last-block slack on the paged side, plus how many cells prefix
 sharing stored once instead of twice.

## Interactions

| | |
|---|---|
| **Transport** | Auto-plays + loops a decode. Each step appends one token to one sequence; a new physical block is allocated **only when the current one fills**, and the step label says which. |
| **Direct manipulation** | Drag the `◂▸` grip on the **block-size** bar (2 → 8 tokens) and drag a **reservation boundary** in the contiguous strip. The allocation, the block tables, the arrows and both waste counters recompute under your hand. |
| **Hover** | Any pool cell → which sequence and logical position it holds, or "reserved by S*i*, empty", or free / fragmented tail. Any paged block → its owner, logical block and slot, its slack, or "free -- any sequence can take it". Any block-table entry → the physical block it points at and whether that block is shared. |
| **Resize the problem** | Sequence count (2-4), block size, reservation size, prefix sharing on/off. |
| **A/B compare** | Coarse 8-token blocks vs fine 2-token blocks, side by side. |
| **Challenges** | Drive paged waste to ≤ 3 cells; get 2+ shared blocks. |

## Render tier

T1 (Canvas2D). The lesson is allocation topology, not pixel throughput -- a few
hundred cells and a fan of arrows, so the floor tier teaches it and nothing is
gained by reaching higher.

## Wiring

`layout.mount` + controls (`blocksize`, `seqs`, `reserve`, `share`) + a step
`Transport` whose `compute` simulates prefill and decode and snapshots the
allocation after every token (`lens[i]`, `alloc[i][L] = physical block`).
Everything drawn with `render.label`/`arrow` + `ctx`, coloured with
`theme.js` tokens only (`categorical` per sequence, `T.violet` for shared,
`T.bad` hatching for waste), so it reads in light and dark.
Headless hooks mirroring the drag state:
`?blocksize=8&seqs=3&reserve=20&share=0`, plus `?step=N`, `?hover=x,y`,
`?play=1`. Source: [`page.js`](page.js).

## Data

None fetched -- the allocation is simulated in-page from a fixed pool of 96
token slots, a deterministic free list, and per-sequence prompt lengths. Same
numbers on every reload, no download, no weights.
