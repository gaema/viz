# embedding -- token-id → row lookup (and the tied lm_head)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: how a token id becomes a vector (a row lookup in the
embedding table), and how the same table is reused as the output projection
when weights are tied. **Anchor**: A5 embedding shape (Family C1, Phase 3
transformer block; see `../plan/curriculum.md`).

## What it shows

An embedding table `E` of shape `[V × D]` (one row per vocabulary token).
Each input token is just an **id** — an index — so "embedding" is a pure row
lookup: `embed(id) = E[id]`. Scrub the transport through the input sequence;
each step highlights the token's **row** in `E` and pulls out its `D`-dim
vector. No arithmetic — just indexing.

The **lm_head** (output projection, hidden state → vocab logits) is shown on
the right with a **tied / untied** toggle:

- **tied** — `lm_head = Eᵀ`: the same `[V × D]` weights serve both the input
  lookup and the output logits. Costs `V·D` parameters total.
- **untied** — a separate `[V × D]` matrix. Costs `2·V·D` parameters.

For large vocabularies `V·D` is huge (e.g. 256k × 4096 ≈ 1B), so tying saves a
lot — the page shows the parameter count for each.

## Render tier

T1 (Canvas2D heatmaps for the tables + the pulled vector).

## Wiring

`layout.mount()` + controls (`dim D`, `seed`, `tie weights`) + a per-token
`Transport`, drawn with `render.heatmap`/`cell` + `ctx` and `tensor.seededRandn`.
Per the framework interaction contract: hover any cell for `E[token,dim]` (or
the fetched-row value); click/drag a row of E (or the input-token strip) to pin
which token id is looked up; the sweep auto-plays + loops. Headless hooks:
`?step=N`/`?play=1`/`?tied=0|1` as on the other pages, plus `?hover=x,y` (fake
cursor, pauses) and `?tok=N` (pin the looked-up token id to row N of E). Source:
[`page.js`](page.js).
