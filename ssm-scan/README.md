# ssm-scan -- the selective state-space scan (Mamba / S6)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: how a selective state-space model carries a recurrent state
through a sequence -- the **scan** `hₜ = Āₜ ⊙ hₜ₋₁ + B̄ₜ · xₜ`, `yₜ = C · hₜ` --
and what makes it **selective**: the timestep `Δₜ` (hence the decay `Āₜ` and the
input gain `B̄ₜ`) depends on the input. **Anchor**: N1 sequence mixer / N2
recurrent state (Family E; see
`../../design/emerging-architectures.md`
Mamba/SSM).

## What it shows

A scalar input sequence `x [L]` is scanned into an `N`-dim hidden state. At each
step:

- **Δₜ = softplus(bias + selectivity · xₜ)** -- the input-dependent timestep.
  Large `Δ` on a *salient* token ⇒ **write** (the state captures `xₜ`); small `Δ`
  on filler ⇒ **hold** (the state coasts, retaining memory). With selectivity 0
  it is a plain linear SSM (constant `Δ`).
- discretize: **Āₜ,ₙ = exp(Δₜ · Aₙ)** (per-dim decay in (0,1)),
  **B̄ₜ,ₙ = (Āₜ,ₙ − 1)/Aₙ · Bₙ**;
- recurrence: **hₜ,ₙ = Āₜ,ₙ · hₜ₋₁,ₙ + B̄ₜ,ₙ · xₜ** -- slow dims (small |Aₙ|)
  remember for a long time, fast dims forget quickly;
- output: **yₜ = Σₙ Cₙ · hₜ,ₙ**.

The panels share a horizontal time axis: input `x`, the selective `Δ`, the state
trajectory `[N×L]` heatmap, and the output `y`. A side inset shows the current
step's per-dim retention `Ā` and new state `hₜ`. The scan steps left to right
(auto-plays + loops).

**Drag an input bar** to change `xₜ` and watch the state + output re-scan from
that point (memory in action); set the **state size N** and **sequence length L**
with the steppers; **hover** any cell. Companion: `mamba-block` (the full block).

## Render tier

T2 (WebGL2 heatmap for the state trajectory; Canvas2D for the bars, the time
axis, and the recurrence inset).

## Wiring

`layout.mount()` + controls (`N`, `L`, `sel`, `dbias`, `seed`) + an `L`-step
`Transport` (scan step by step) + `onPointer` input-bar drag + hover.
`?step` / `?drag` / `?sel` / `?hover` hooks. Source: [`page.js`](page.js).
