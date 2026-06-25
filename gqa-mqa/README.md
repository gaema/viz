# gqa-mqa -- KV-head sharing (MHA / GQA / MQA)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how query heads share key/value heads, and why that shrinks
the KV cache. **Anchor**: A2 head_dim layout / A4 KV cache (Family B4).

## What it shows

`n_q` query heads on top, `n_kv` key/value heads below, with a line from each
query head to the KV head it reads. Scrub the transport through the schemes
(the divisors of `n_q`, descending):

- **MHA** (`n_kv = n_q`) — every query head has its own K,V head.
- **GQA** (`1 < n_kv < n_q`) — query heads are grouped; each group of
  `n_q / n_kv` query heads shares one KV head (e.g. Llama-2/3-70B: 64 Q → 8 KV).
- **MQA** (`n_kv = 1`) — all query heads share a single KV head (PaLM, Falcon).

Query heads are colored by their group, so the grouping is obvious. A
**KV-cache memory bar** tracks `n_kv / n_q` of the MHA size — because the cache
stores K,V **per KV head**, fewer KV heads means a proportionally smaller cache,
which is the decode-time memory/bandwidth bottleneck.

## Render tier

T1 (Canvas2D — boxes, connecting lines, a memory bar).

## Wiring

`layout.mount()` + controls (`query heads`) + a per-scheme `Transport` built in
`page.js` (the divisors of `n_q`), drawn with `ctx` + `render.arrow`/`label`.
Direct-manipulable per the framework interaction contract: **drag the handle on
the KV-cache bar** to morph MHA → GQA → MQA (snaps to the divisors of `n_q`);
**hover** a Q head for its KV group + sibling heads, or a KV head for the Q heads
that read it. Auto-plays the schemes + loops. Headless hooks: `?step=N`/`?play=1`
plus `?hover=x,y` (canvas-px) and `?kv=N` (set #KV heads → nearest divisor,
pauses). Source: [`page.js`](page.js).
