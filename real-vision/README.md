# real-vision -- a ViT classifying a real image (Phase 9)

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server + a one-time ~340 MB model download: `python3 -m http.server 8099`

Interactive page: the **vision entry** to the real-* family. A real **Vision
Transformer** (ViT-base) classifies a real photo in your browser. **Anchor**:
[`patch-embedding`](../patch-embedding/README.md) (ViT image→patches). Part of
**Phase 9 — real-model grounding**.

## What it shows

A ViT splits the image into 16×16 **patches** (14×14 = 196 of them), linearly
embeds each into a token, and runs a transformer — the *same machinery* as a
language model, on pixels instead of words. The page shows:

- the real photo with the **patch grid** overlaid — the ViT "tokens", the image
 analog of the BPE tokens the curriculum started with (grounds the
 `patch-embedding` page);
- the model's real **top-5 class predictions**.

It is the vision counterpart of [`real-logits`](../real-logits/README.md) (real
model *output*). Here **transformers.js is the trusted engine** — `image-
classification` with `Xenova/vit-base-patch16-224` — so there's no hand-written
vision forward; the page's job is to show the patches and the real predictions.

## Verified against PyTorch

The top class is checked against `ViTForImageClassification`
(`google/vit-base-patch16-224`): on the sample images the page agrees with
PyTorch — **cats → "Egyptian cat" (~94%)**, **tiger → "tiger" (~89%)** (reference
[`vit-groundtruth.fixture.json`](vit-groundtruth.fixture.json)). Live-verified in
a real browser by.

> Deeper future extension (noted in the plan): a hand-written ViT forward to show
> real *attention over patches* / feature maps — transformers.js can't emit
> intermediate activations (same ONNX limit as `real-attention`), so it needs the
> same hand-written-forward treatment GPT-2 got, plus image-preprocessing match.

## Render tier

T2 (Canvas2D image + patch grid + prediction bars).

## Wiring

`layout.mount` + an `image` select + classify / "load real ViT" buttons. Uses
transformers.js `image-classification`; async, degrades to a labelled synthetic
stand-in (with the real patch grid) offline. Two **challenges** (`?ch=N`): load
the real model, and classify the cats image to a cat. Headless hooks: `?image=`,
`?real=0`. Source: [`page.js`](page.js).
