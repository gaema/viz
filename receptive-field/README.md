# receptive-field -- how stacked convs grow what each output sees

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **receptive field** -- the patch of input that a single
output unit depends on -- and how **stacking** convolution layers (and
**dilating** them) grows it. **Anchor**: F (CNN / vision); Family F, companion to
[convolution](../convolution/README.md) ().

## What it shows

A 1-D stack of conv layers (input at the bottom, output at the top). Each unit
reads `k` units in the layer below (the conv), so its dependency **cone** widens
downward. The receptive field is how wide that cone is when it reaches the input:

    RF = 1 + Σₗ (k − 1)·dₗ          (stride-1; dₗ = dilation at layer l)

- **stacking** (depth `L`): each `3×3` layer adds `k−1 = 2` to the RF, so `L`
  layers of `3×3` see `2L + 1` inputs -- a deep stack of cheap small kernels
  reaches a wide RF (two 3×3 ≈ one 5×5, three ≈ one 7×7, but fewer params).
- **dilation**: spacing the taps apart grows the RF for the same `k`. With
  **doubling** dilation (`dₗ = 2^(l−1)`, WaveNet/TCN), the RF grows
  **exponentially**: `RF = 1 + (k−1)(2^L − 1)`.

The focus unit's cone is highlighted down to the input, where the RF interval is
bracketed; the readout gives the RF size + the formula.

**Drag** the *conv layers* slider to stack more layers and watch the cone (and
the RF) grow; **hover** any unit to focus its cone and inspect its RF; the focus
unit sweeps the top layer (the RF shifts with position).

## Render tier

T1 (Canvas2D: the layered unit grid, the dependency cone, the input RF bracket).

## Wiring

`layout.mount()` + controls (`N`, `L`, `k`, `mode`, `dil`) + `animate` (the focus
sweep) + hover-to-focus. `?L` / `?k` / `?mode` / `?dil` / `?focus` / `?hover`
hooks. Source: [`page.js`](page.js).
