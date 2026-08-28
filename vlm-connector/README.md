# vlm-connector -- how the vision encoder reaches the language model

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **the VLM connector** -- the piece between a vision encoder
a language model, and the reason image resolution is a **token-budget decision
rather than a quality dial**. **Anchor**: G (multimodal / deployment); Family G.

## What it shows

A vision encoder turns an image into a grid of patch embeddings. Those cannot
enter the language model as they are -- they are the **wrong width** (the
encoder's dim, not the LLM's embedding dim) and the **wrong number** (a grid that
grows with the square of the resolution). A **connector** fixes both. Width is a
projection either way; the interesting choice is *how many tokens come out*.

| Connector | Shape | Tokens per image | What it costs |
|---|---|---|---|
| **linear / MLP projector** | one projection per patch | `(H/p)·(W/p)` -- every patch becomes a token | nothing structural, and therefore everything: the token count *is* the resolution. The most expensive shape ([LLaVA](https://arxiv.org/abs/2304.08485)) |
| **pixel-shuffle / patch-merge** | a `k×k` block of neighbouring patches is concatenated along the **channel** axis, then projected once | `⌊(H/p)/k⌋·⌊(W/p)/k⌋` | spatial detail: `k²` patches now share one token, and a grid side that is not a multiple of `k` leaves a remainder outside the merged grid — which is why the count floors rather than divides ([Qwen2-VL](https://arxiv.org/abs/2409.12191), [InternVL](https://arxiv.org/abs/2312.14238)) |
| **resampler (perceiver-style cross-attention)** | a **fixed** number of learned queries attend to all the patches | `Q`, **constant at any resolution** | a fixed bottleneck: past a point, extra pixels have nowhere to go, and the output tokens carry no spatial layout at all ([Flamingo](https://arxiv.org/abs/2204.14198)) |

**The trade is arithmetic, and the page computes it live rather than tabulating
it** -- `tokens = (H/patch)·(W/patch)/merge²`, with the language model's attention
work growing with the **square of the total sequence length**. So doubling the
resolution quadruples a projector's tokens and multiplies the attention work by
roughly sixteen: the cost moves with the square, twice over. The closing readout
gives **tokens per image** and **the percentage of a fixed context** they consume,
so a high resolution visibly eats the conversation.

## Relationship to `multimodal-inject`

[`multimodal-inject`](../multimodal-inject/README.md) shows **that** media tokens
get spliced into the text sequence and that self-attention then treats them like
any other column. This page is the complement: **how many** of those tokens there
are, **from where** in the image each one came, and **at what cost** in context
and attention. Read that page for the splice; read this one for the bill.

## Interactions

- **Transport** stepping the pipeline `patchify → encode → connect → splice`,
 autoplaying and looping; each stage reveals its panel. `?step=N` freezes a
 stage for capture.
- **Direct manipulation, two handles.** Drag the **◢ corner of the image** to
 change its resolution, and drag the **rail inside the connector** to change the
 merge factor `k` (or, in resampler mode, the query count). The token count, the
 cost curve's marker, and the context bar all move under your hand. A projector
 has no rail -- it has nothing to trade, which is the point.
- **Connector switch** (`projector` / `merge` / `resampler`) redraws the same
 image's path through a different mechanism, and highlights that shape's curve.
- **Hover any patch** -- on the image or on the encoder grid -- for the output
 token it lands in: an index for a projector, a merge group (or "dropped, outside
 the merged grid") for patch-merge, and "no single token: every learned query
 attends to it" for a resampler. Hover the context bar for what occupies a
 position.
- **Resize the problem**: patch size, images in the chat, text tokens, context
 budget.

## Render tier

T1/T2 (Canvas2D): the pipeline strip (procedural image + patch grid, encoder
panel, per-connector mechanism drawing, output tokens), a live log-scale
tokens-vs-resolution curve for all three connectors at once, a context-budget
bar, and the `seq²` attention comparison.

## Wiring

`layout.mount` + `controls` (`conn`, `res`, `patch`, `merge`, `queries`,
`imgs`, `text`, `ctx`, `seed`) + a 4-step `transport` (autoplay, loop)
`animate` (the dataflow dots and the patch sweep) + `onPointer` for the two drag
handles + hover. Every control is a URL hook (the framework mirrors state into
the query string); `?step=N`, `?hover=x,y` and `?play=1` are the headless
stand-ins for the transport and the cursor. Source: [`page.js`](page.js).
