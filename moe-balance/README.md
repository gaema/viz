# moe-balance -- load balancing, collapse, and why experts starve

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: the MoE **balance problem** over a whole batch -- without a
balancing pressure the router *collapses* (a few experts get everything, the
rest starve); the load-balance loss flattens the distribution. **Anchor**: A6
MoE variant; Family D, companion to [moe-routing](../moe-routing/README.md)
(which shows a single token's top-k routing). See
`../../design/emerging-architectures.md`
MoE.

## What it shows

A batch of `T` tokens is dispatched to `E` experts as a **load histogram** (the
bars flow smoothly as the routing changes). A balance knob `λ` interpolates the
effective routing:

- **`λ = 0` (no balance loss)**: routing follows the router's raw preference, a
  skewed distribution -- a couple of experts take most tokens and the rest
  **starve** (≈ 0 load, marked). Starved experts never get gradients, so they
  stay bad and never get picked: the **rich-get-richer collapse**.
- **`λ = 1` (strong balance loss)**: load is pushed toward uniform `T/E`; no
  expert starves.

Also shown:

- **capacity + drops**: each expert holds `≈ factor · T/E`; tokens routed to a
  full expert are **dropped** (red overflow). Drag the capacity line.
- **load-balance aux loss** `E · Σₑ fₑ·Pₑ` (`fₑ` = load fraction, `Pₑ` = router
  prob), the **coefficient of variation**, and the starved-expert count.
- **shared expert** (toggle): an always-on expert (DeepSeek-MoE style) that
  absorbs common patterns so the routed experts can specialize.

**Drag any expert's bar up or down to shift load onto or off it** -- the other
experts rebalance, and the aux loss, capacity drops, and starved count respond
live. Drag the capacity line to trade drops for compute; move `λ` to watch the
collapse flatten out; **hover** any bar. The bars ease to their targets.

## Render tier

T1 (Canvas2D bars + capacity line + the live fill; no heatmap needed).

## Wiring

`layout.mount()` + controls (`E`, `lam`, `shared`, `cap`, `seed`) + `animate`
(bars ease to their targets via `api.t`) + `onPointer` bar drag (shift load) +
capacity-line drag (routes through `controls.set`) + hover. `?lam` / `?cap` /
`?shared` / `?shift` / `?hover` hooks. Source: [`page.js`](page.js).
