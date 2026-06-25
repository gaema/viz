# multimodal-inject -- vision/audio tokens in the text stream

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: **multimodal injection** -- how a vision-language / audio-language
model feeds a non-text input into a text LLM. **Anchor**: G (multimodal /
deployment); Family G (see `../plan/curriculum.md`).

## What it shows

There is no separate "image input" to an LLM -- the image is turned into **tokens**
that live in the **same embedding space as words**:

1. **Encode** -- a ViT (image) or audio encoder turns the input into per-patch /
   per-frame feature vectors.
2. **Project** -- a **projector MLP** (`D_enc → D`) maps those features into the
   LLM's token-embedding dimension `D`.
3. **Inject** -- the resulting "soft" media tokens are **spliced into the text
   token sequence** at a `<image>` / `<audio>` placeholder.
4. **Attend** -- the merged sequence (text embeddings + media tokens, every column
   the same `D`-dim vector) is run through ordinary **self-attention**, which
   attends across text and media **uniformly** -- the model never sees a "modality
   type", only token vectors.

This is exactly how LLaVA, Qwen-VL, and audio-LMs work. The merged-sequence heatmap
colour-codes each column by source (text = blue, image = orange, audio = green) and
brackets the injected media block.

A **causal self-attention grid** below the LLM bar makes the "uniform attention"
point concrete rather than asserted: row `i` attends to columns `j ≤ i`, coloured
by weight, with source-coloured row/column ticks. A query-row readout breaks down
how much of that row's attention lands on **text vs image vs audio** -- e.g. the
final `?` / `.` token attends ~half to the image tokens (image mode), or splits
three ways across text/image/audio (both mode). Hover any row to inspect it, or any
cell for the exact `attn[i→j]` weight.

**Drag** the media block ↔ to move where it is injected; switch the **modality**
(image / audio / both), the **token count**, and the **prompt**; hover a column to
see its source and how it was produced; the injection connector animates.

## Render tier

T2 (Canvas2D: the encoder/projector panel with a procedural image + waveform, the
prompt boxes, the colour-coded merged-sequence heatmap feeding an LLM bar, and the
causal self-attention grid with its per-modality query-row breakdown).

## Wiring

`layout.mount()` + controls (`modality`, `N`, `prompt`, `seed`) + `animate` (the
injection flow dot) + `onPointer` drag-the-media-block + hover. `?modality` / `?N` /
`?prompt` / `?seed` / `?inject` / `?hover` hooks. Source: [`page.js`](page.js).
