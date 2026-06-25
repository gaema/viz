# mamba-block -- the full Mamba block

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the whole Mamba block end to end -- the macro view that wraps
the selective scan ([ssm-scan](../ssm-scan/README.md)) in a causal conv, a gated
branch, and in/out projections, on a residual stream. The SSM analog of
[transformer-block](../transformer-block/README.md). **Anchor**: N1 sequence
mixer / N2 recurrent state (Family E; see
`../../design/emerging-architectures.md`
Mamba).

## What it shows

The block, on a residual spine `x → RMSNorm → … → ⊕ → out`:

1. **in_proj**: `RMSNorm(x)` is projected up and **split** into two branches,
   `u` and `gate` (each `E = expand·D` channels).
2. **main branch**: `u →` **causal Conv1d** (a short depthwise conv that mixes a
   few neighbouring positions) `→` **SiLU** `→` **selective SSM** (the per-channel
   scan with input-dependent `Δ`) `→ y`.
3. **gate branch**: `gate →` **SiLU** `→ g`.
4. **gate**: `y ⊙ g` -- the SSM output is gated by the parallel branch (this is
   what replaces attention's value-mixing).
5. **out_proj** back to `D`, then the **residual** add: `out = x + out_proj(y⊙g)`.

So Mamba mixes the sequence with `conv + selective SSM` (linear-time, a constant
state) instead of attention, and gates it with a SiLU branch. Heatmaps show `x`,
the SSM output `y`, the gate `g`, the gated product `y⊙g`, and `out`; boxes mark
the ops; the SSM box breadcrumbs to [ssm-scan](../ssm-scan/README.md).

Step the transport through the **sequence**: the SSM state evolves token by
token and every heatmap fills column by column (the future is veiled), so you
watch the block process the stream one position at a time. **Drag any x cell**
to steer the input and watch the whole block recompute; **hover** any cell.

## Render tier

T2 (WebGL2 heatmaps for the branch tensors; Canvas2D for the boxes, arrows,
gate ⊙, and the residual spine).

## Wiring

`layout.mount()` + controls (`L`, `D`, `expand`, `sel`, `seed`) + an `L`-step
`Transport` (the scan over the sequence; the state + heatmaps fill column by
column) + `onPointer` x-cell drag + hover. `?step` / `?drag` / `?hover` hooks.
Source: [`page.js`](page.js).
