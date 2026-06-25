# feature-maps -- what a CNN sees: edges → textures → parts

Interactive page: the channel **activations** (feature maps) a convolutional stack
produces at each depth -- the famous progression from **edges** (shallow) to
**textures** (mid) to **parts/blobs** (deep). **Anchor**: F (CNN / vision);
Family F, builds on [convolution](../convolution/README.md) and
[receptive-field](../receptive-field/README.md) ().

## What it shows

A small fixed CNN runs over an input image:

- **Layer 1 — edges**: four real **oriented edge filters** (Sobel: horizontal,
  vertical, two diagonals) → four feature maps that light up along edges of that
  orientation. Same resolution as the input.
- **Layer 2 — textures**: filters that combine the layer-1 edge maps (a 3×3×4
  conv + ReLU, stride 2) → six maps responding to edge *combinations* — corners,
  junctions, repeated patterns. Half the resolution (downsampled).
- **Layer 3 — parts**: filters over the layer-2 maps (3×3×6 + ReLU, stride 2) →
  six coarse, abstract maps — blob/part detectors. Quarter resolution.

The story is the **progression**: each layer is built from the one below, gets
smaller (downsampling), and grows more abstract — early neurons fire on simple
edges, deep neurons on whole parts. This is the clearest intuition for "what is
the network learning?"

**Drag (paint) on the input** to draw your own shape and watch the edges,
textures, and parts respond; pick a preset shape; **hover** any feature map to
see its layer and what it detects; the forward pass reveals layer by layer
(auto-plays + loops).

## Render tier

T2 (Canvas2D grids of the per-channel feature maps + the input; many small
heatmaps).

## Wiring

`layout.mount()` + controls (`shape`, `seed`) + a 4-step `Transport` (reveal
input → L1 → L2 → L3) + `onPointer` paint-the-input + hover. `?step` / `?shape` /
`?paint` / `?hover` hooks. Source: [`page.js`](page.js).
