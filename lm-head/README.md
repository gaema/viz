# lm-head -- hidden state → vocab logits

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: the output projection that turns a hidden state into a logit
per vocabulary token, plus the three tricks around it — tied weights, logit
soft-capping, and decode-time logit slicing. **Anchor**: A9 lm_head structure /
A10 vocab (Family C6, Phase 3; see `../plan/curriculum.md`).

## What it shows

The last hidden state `h` (a `D`-vector) is projected against every row of the
unembedding matrix `W_lm` (`[V × D]`): `logit[v] = h · W_lm[v]`. Scrub the
transport across the vocabulary to fill the logit per token (each = a dot
product with that token's row). The argmax is the predicted next token.

Toggles:

- **tie weights** — `W_lm = E` (reuse the embedding table) vs a separate matrix.
  Tying saves `V·D` parameters (huge at real vocab sizes); the `W_lm` heatmap
  then equals the embedding table.
- **soft-cap** — `logit ← cap · tanh(logit / cap)`, which bounds logits to
  `±cap` (Gemma 2 caps the final logits at 30). The bars show raw vs capped and
  the `±cap` lines; large logits get squashed toward the cap.
- **slice last token** — in **decode** you only need the next-token
  distribution, so you slice the hidden states to the **last** position and
  compute `[1 × V]`, not the full `[N × V]`. The annotation shows the compute
  saved (`N×` fewer logit rows).

The hidden states `H [N×D]` are shown with the sliced row highlighted.

## Render tier

T2 (WebGL2 heatmaps for `H` and `W_lm`; Canvas2D for the logit bars + cap lines).

## Wiring

`layout.mount()` + controls (`seed`, `tie`, `soft-cap`, `cap`, `slice`) + a
per-vocab `Transport` (auto-plays + loops), drawn with `render.heatmap`/`cell` +
`ctx` and `tensor.seededRandn`. **Interactive** (2026-06-13 contract): drag any
`h` cell ↕ to steer the hidden state and watch every logit + the argmax recompute
live; hover a logit bar for its full derivation (`h · W_lm[v] = Σ_d h[d]·W[v,d]`),
an `h` or `W_lm` cell for its value. `?step`/`?tie`/`?cap0`/`?slice0`/`?play`
headless hooks, plus `?hover=x,y` (fake cursor) and `?drag=d,val` (set `h[d]`,
e.g. `?drag=2,2.5`). Source: [`page.js`](page.js).
