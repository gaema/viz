# viz — concept visualizations of CNN/LLM internals

Interactive HTML pages that show the **math** of CNN/LLM internals — one page per
mechanism, no build step. Hover to inspect, drag to scrub, watch the tensors flow.

**Live:** <https://viz.gaema.ai>

## ▶ Open the demos

- **New here? [`tour.html`](tour.html)** is a guided tour — a narrated walkthrough of
  the demos in curriculum order, each with a plain-language "why it matters", an
  embedded live preview, and a progress track (`←`/`→` to move).
- **[`index.html`](index.html)** is the central index: open it and click any
  card to launch that demo.

The demos use ES modules, so serve the folder over **http** (not `file://`):

```sh
python3 -m http.server 8099
# then open  http://localhost:8099/index.html
```

Every page's own `README.md` also has an **▶ Open this demo** link at the top.

**Navigating.** Every live demo carries an in-demo nav strip (`← all demos` +
`‹ prev` / `next ›` across the curriculum order + a position counter + a family tag +
a `🔗 copy link` button), keyboard nav (`←` / `→` between demos, `/` jumps to the
catalogue), and deep links (control state is mirrored into the URL, so `copy link`
reproduces the exact view). These come from the shared `framework/` `mount()`.

## Page families

| Family | Pages | Theme |
|--------|-------|-------|
| A | matmul, dot-product, dtype-bits, softmax | Foundational primitives |
| B | qkv-projection, scaled-dot-attention, multi-head, gqa-mqa, causal-mask, attention-patterns, kv-cache, rope, flash-attention | Attention |
| C | tokenization, embedding, mlp-gated, activations, normalization, residual-stream, lm-head, transformer-block, prefill-vs-decode, sampling | Transformer block |
| D | moe-routing, moe-balance | Mixture of Experts |
| E | ssm-scan, mamba-block, gated-deltanet, hybrid-by-layer | SSM / linear attention |
| F | convolution, receptive-field, pooling, feature-maps, batchnorm, residual-block, depthwise-separable, patch-embedding | CNN / vision |
| G | quantization, vram-budget, multimodal-inject | Quant / multimodal |
| H | forward-pass, backprop | End to end & training |

The **forward-pass** capstone chains the atoms into one end-to-end next-token
prediction; **backprop** shows the backward pass / training. The `real-*` pages run
real GPT-2 weights in the browser.

## Layout

```
index.html      central index
tour.html           guided tour
<concept>/          one dir per mechanism: index.html + page.js + README.md
framework/          shared runtime: layout.mount() chrome, controls, render, tensor, order
```

Static site, no build step — every page is plain HTML + ES modules.
