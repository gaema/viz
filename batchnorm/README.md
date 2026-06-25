# batchnorm -- running mean/variance + train vs inference

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **Batch Normalization** -- normalize each channel's activations,
**track** a running mean/variance during training (an exponential moving
average), and at **inference** use that tracked average instead of the batch's
own statistics. **Anchor**: F (CNN / vision); Family F, builds on
[convolution](../convolution/README.md) ().

## What it shows

For one channel of a feature map (the paintable input), BatchNorm computes
`y = γ · (x − μ)/√(σ² + ε) + β` -- but **which μ, σ it uses differs by mode**:

- **Training**: use the **current batch's** mean/variance (so each batch is
  recentered to ~0 mean, unit variance), *and* update a **running** estimate:
  `running_μ ← (1−m)·running_μ + m·batch_μ` (same for variance). Real batches are
  noisy, so the running average **smooths** the jittery per-batch stats -- the
  page tracks `batch_μ` (jittery dots) vs `running_μ` (smooth line) converging to
  the true channel mean over steps.
- **Inference**: there's no batch, so BN uses the **frozen running** mean/variance
  it learned. The consequence: a test input that's *brighter than average* comes
  out **brighter** (measured against the learned population) -- it is **not**
  recentered to zero the way training would. Toggle train ↔ inference on a bright
  input to see the output shift.

`γ` (scale) and `β` (shift) are learned per channel and let the network undo the
normalization if it wants. BN does all of this **independently per channel**.

**Drag (paint)** the input to change its values and watch the running average
drift to track it, and the normalized output respond; toggle **train / inference**;
adjust **momentum** (EMA rate), `γ`, `β`. The training EMA animates.

## Render tier

T1 (Canvas2D: the paintable input + normalized output grids, the
running-statistics tracking chart).

## Wiring

`layout.mount()` + controls (`shape`, `mode`, `momentum`, `gamma`, `beta`, `seed`)
+ `animate` (the training EMA) + `onPointer` paint-the-input + hover. `?mode` /
`?shape` / `?paint` / `?hover` hooks. Source: [`page.js`](page.js).
