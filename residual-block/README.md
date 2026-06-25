# residual-block -- the skip connection y = F(x) + x

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: the **ResNet residual block** -- a small learned function `F`
runs in parallel with an **identity shortcut** that copies the input straight to
the output, so `y = F(x) + x`. **Anchor**: F (CNN / vision); Family F, builds on
[convolution](../convolution/README.md) (see `../plan/curriculum.md`).

## What it shows

Two linked views:

- **One block** (top): `x → F(x) = W₂·relu(W₁x) → ⊕ → y = F(x) + x`. The `F`
  branch is scaled to a **gain `g`** relative to `x` (so `|F(x)| ≈ g·|x|`). At
  initialization `g` is small, `F(x) ≈ 0`, and the block is **≈ identity**
  (`y ≈ x`) -- a small refinement on top of the signal, not a replacement. The
  dashed green arc is the identity skip.
- **Depth / gradient chart** (bottom): the gradient magnitude as it propagates
  **back** through `L` stacked blocks (log scale). Per block the local factor is
  `g` without the skip vs `1+g` with it, because `∂y/∂x = I + ∂F/∂x`. So the
  gradient reaching layer 0 is `g^L` (vanishes for `g<1`) **without** the skip
  but `(1+g)^L ≈ O(1)` **with** it -- the `+I` is a **gradient highway** straight
  back to the early layers. An animated pulse travels the active curve.

Toggle the **skip** off and watch the gradient curve crash to ~0 at the input
(the early layers stop learning) -- the single clearest reason residual
connections made 100+ layer networks trainable.

**Drag** the input `x` bars to refeed the block; tune **depth `L`**, **gain `g`**,
toggle the **skip**, pick a **seed**. The backprop pulse animates.

## Render tier

T1 (Canvas2D: the three vector bar-charts + the log-scale depth/gradient chart).

## Wiring

`layout.mount()` + controls (`L`, `gain`, `skip`, `seed`) + `animate` (the
backprop pulse) + `onPointer` drag-the-input + hover. `?L` / `?gain` / `?skip` /
`?drag` / `?hover` hooks. Source: [`page.js`](page.js).
