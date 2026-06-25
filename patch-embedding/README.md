# patch-embedding -- ViT: image → patches → tokens

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **Vision Transformer (ViT) patch embedding** -- how an image
becomes a **sequence of tokens** a transformer can consume. **Anchor**: F (CNN /
vision), the CNN→transformer bridge; Family F, builds on
[convolution](../convolution/README.md) ().

## What it shows

A ViT has no convolutions in its trunk, so the image must be turned into tokens:

1. **Patchify** -- cut the `S×S×3` image into a grid of non-overlapping `P×P`
   patches (`N = (S/P)²` of them).
2. **Flatten + project** -- each patch is `P·P·3` raw numbers; a shared linear
   matrix `W` (`D × P²·3`) maps it to a `D`-dim **token**: `token = W·flatten`.
   This is **mathematically identical to a single `Conv2d` with
   `kernel = stride = P`** -- the CNN→transformer bridge.
3. **CLS + position** -- a learnable `[CLS]` token is prepended and a positional
   embedding `pos[i]` is added to each token (a transformer is otherwise
   order-blind), giving a length-`N+1` sequence of `D`-dim tokens.

The page shows the gridded image, the selected patch flattening → projecting →
landing as one column in the token-sequence heatmap (`[CLS]` + `N` patch tokens,
`D` rows). Smaller `P` → more, smaller patches → a **longer sequence** (cost grows
~`1/P²`).

**Drag** on the image (or the slider) to pick a patch and watch its token light up
in the sequence; tune the patch size `P`, embed dim `D`, image preset, seed; the
rasterize-to-tokens sweep animates; hover a patch or a sequence column to inspect.

## Render tier

T2 (Canvas2D: the procedurally-rendered image via an offscreen canvas + the
patch/projection/token panels + the token-sequence heatmap).

## Wiring

`layout.mount()` + controls (`preset`, `P`, `D`, `patch`, `seed`) + `animate` (the
rasterize sweep) + `onPointer` drag-to-pick-patch + hover. `?preset` / `?P` / `?D` /
`?patch` / `?seed` / `?hover` hooks. Source: [`page.js`](page.js).
