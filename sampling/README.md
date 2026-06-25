# sampling -- greedy / top-k / top-p / temperature

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how the next token is actually chosen from the lm-head's
logits -- temperature, then a truncation strategy (greedy, top-k, top-p), then a
draw. **Anchor**: the decode-time companion to [lm-head](../lm-head/README.md)
(which produces the logits); Family C.

## What it shows

The logits become probabilities via `softmax(logits / T)`:

- **temperature `T`**: `T<1` sharpens (more peaked, more greedy), `T>1` flattens
  (more random). `T→0` is greedy.
- **greedy**: take the argmax -- deterministic, always the top bar.
- **top-k**: keep only the `k` highest-probability tokens, renormalize, sample
  from those.
- **top-p (nucleus)**: keep the smallest set of top tokens whose cumulative
  probability first reaches `p`, renormalize, sample.

Bars are drawn in descending probability; the **kept** set is colored, the
**cut** tail is greyed. The cumulative curve + the `p` line show the nucleus
(top-p); a divider shows the `k` cut (top-k). A live sampler draws tokens from
the kept, renormalized distribution and tallies them -- watch the empirical
frequency converge to the probabilities (greedy always picks the top).

**Drag** the `k` divider or the `p` line to resize the kept set; the slider
tracks it. **Hover** any bar for its token / logit / prob / cumulative.

## Render tier

T1 (Canvas2D bars + cumulative curve + the live sampler; no heatmap needed).

## Wiring

`layout.mount()` + controls (`temp`, `strat`, `k`, `p`, `seed`) + `animate`
(the live sampler via `api.t`) + `onPointer` threshold drag (routes through
`controls.set` so the slider syncs) + hover. `?temp`/`?strat`/`?k`/`?p`/`?hover`
hooks. Source: [`page.js`](page.js).
