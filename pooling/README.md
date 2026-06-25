# pooling -- max / average pooling, downsampling

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: spatial **pooling** -- a window slides over the input and
reduces each window to one number (its **max** or its **average**), shrinking the
feature map. The cheap, parameter-free downsampler in a CNN. **Anchor**: F (CNN /
vision); Family F, companion to [convolution](../convolution/README.md) ().

## What it shows

A `k×k` window steps over the `n×n` input with stride `s` (usually `s = k`, so
the windows **tile** the input without overlap). Each window becomes one output
cell:

- **max pooling**: `out = max(window)` -- keeps the strongest activation in the
  window (the *argmax* cell is highlighted); translation-tolerant, drops the rest.
- **avg pooling**: `out = mean(window)` -- smooths the window into its average.

Output size = `⌊(n − k)/s⌋ + 1`, so a `2×2` pool with stride 2 **halves** each
spatial dimension (downsampling). No weights -- pooling just summarises. The
current window is outlined on the input (with the max cell marked for max-pool);
the smaller output grid fills as the window slides (auto-plays + loops).

**Drag** any input cell to change its value and watch the pooled output recompute
(for max-pool, the output only moves if you change the window's max); **hover**
any input or output cell. Toggle max ↔ avg to compare.

## Render tier

T1 (Canvas2D: the input grid + sliding window, the downsampled output grid, the
per-window reduction).

## Wiring

`layout.mount()` + controls (`n`, `k`, `stride`, `pool`, `seed`) + an
output-position `Transport` (slide the window) + `onPointer` input-cell drag +
hover. `?step` / `?n` / `?k` / `?stride` / `?pool` / `?drag` / `?hover` hooks.
Source: [`page.js`](page.js).
