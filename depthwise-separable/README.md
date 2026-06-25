# depthwise-separable -- MobileNet conv, the FLOP cut

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **depthwise-separable convolution** (MobileNet) -- factorize
a standard conv into a **depthwise** conv (one `k×k` filter per input channel, no
channel mixing) followed by a **pointwise** `1×1` conv (mixes channels, no spatial
extent), and see the MAC/FLOP reduction it buys. **Anchor**: F (CNN / vision);
Family F, builds on [convolution](../convolution/README.md) ().

## What it shows

A standard conv mixes **space and channels jointly**: every output channel reads a
`k×k` patch of *every* input channel, so the cost is a **product**:

```
full MACs = H·W·Cin·Cout·k²
```

MobileNet splits that into two cheap steps:

- **Depthwise** -- one `k×k` filter per input channel, applied independently
  (spatial filtering, NO channel mixing): `H·W·Cin·k²`. Drawn as **parallel**
  channel→channel links (no crossing).
- **Pointwise** -- a `1×1` conv that mixes channels (channel mixing, NO spatial
  extent): `H·W·Cin·Cout`. Drawn as **dense** links with a `1×1` badge.

So the cost becomes a **sum** instead of a product, and the ratio is

```
full / dwsep = (k²·Cout) / (k² + Cout) = 1 / (1/Cout + 1/k²)  ≈ 8–9×  for 3×3
```

The bottom bar compares `full` (one bar) vs `depthwise + pointwise` (two segments)
to scale, with the live `×` reduction and the parameter-count ratio.

**Drag** the input or output channel stack (↕) to change `Cin`/`Cout` and watch the
bar and ratio update; sliders for `k`, `Cin`, `Cout`, and the feature-map size; the
two stages (depthwise / pointwise) animate in turn; hover a stack or the bars to
inspect counts.

## Render tier

T1 (Canvas2D: the two connectivity diagrams + the MAC-comparison bar).

## Wiring

`layout.mount()` + controls (`k`, `Cin`, `Cout`, `HW`) + `animate` (the two-stage
highlight) + `onPointer` drag-the-channel-stack + hover. `?k` / `?Cin` / `?Cout` /
`?HW` / `?hover` hooks. Source: [`page.js`](page.js).
