# prefill-vs-decode -- two inference regimes

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the two phases of autoregressive inference, side by side --
**prefill** (process the whole prompt in one parallel pass) vs **decode**
(generate one token at a time). **Anchor**: A4 KV cache / runtime shape (Family
C). Companion to
[kv-cache](../kv-cache/README.md), which shows the cache itself filling.

## What it shows

- **Prefill**: all `N` prompt tokens go through at once -- one big
  matrix-matrix GEMM, the full `N×N` causal attention. Lots of FLOPs done in
  parallel ⇒ **compute-bound** (the GPU's math units are saturated).
- **Decode**: one new token per step. A skinny `1×D` query (a GEMV, not a GEMM)
  that **re-reads the entire KV cache** `[(N+t)×D]` to attend over every past
  position, emits one token, appends one cache row, repeats. Little compute,
  big reads ⇒ **memory-bandwidth-bound** (the math units sit mostly idle).

The same attention is really computed in both (causal `softmax(QKᵀ/√d)`); the
difference is the *shape* of the work: a parallel triangle once vs a growing
single row every step. The cost readout contrasts the FLOPs/bytes and why
prefill is fast-per-token while decode is the throughput bottleneck.

Press play (or scrub) to run prefill then the decode loop; **drag** the timeline
handle to move through the steps; **hover** any cell for its value + which phase
filled it.

## Render tier

T2 (WebGL2 heatmaps for the prompt / attention / cache; Canvas2D for the
timeline, brackets, arrows, and cost panel).

## Wiring

`layout.mount()` + controls (`N`, gen `G`, `D`, `seed`) + a `(1 + G)`-stage
`Transport` (prefill then G decode steps) + `onPointer` timeline drag + hover.
`?step` / `?pos` / `?hover` hooks. Source: [`page.js`](page.js).
