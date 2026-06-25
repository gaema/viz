# moe-routing -- router, top-k experts, load balancing

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a Mixture-of-Experts layer routes each token to a few
experts -- router logits → softmax gate → top-k selection → per-expert load,
capacity, and dropped tokens. **Anchor**: A6 MLP / MoE variant (the gated-MLP
sublayer replaced by routed experts); Family D (see
`../../design/emerging-architectures.md`
MoE and).

## What it shows

A small **router** (gating network) scores every token against every expert:
`router_logits [N × E]` → `softmax` per token → gate weights. Each token is sent
to its **top-k** experts (usually k=2); those k gate weights are renormalized
and the expert outputs are combined by them.

- **Token distribution**: the `[N × E]` gate heatmap with each token's chosen
  experts outlined; arrows from the current token to its experts.
- **Load + capacity**: per-expert bars showing how many tokens each expert got.
  Each expert has a **capacity** `≈ capacity_factor · N·k / E`; tokens routed to
  a full expert are **dropped** (shown red). A lopsided router overloads a few
  experts and drops tokens.
- **Load balance**: the Switch-Transformer auxiliary loss
  `aux = E · Σₑ fₑ·Pₑ` (`fₑ` = fraction of tokens to expert e, `Pₑ` = mean router
  prob for e) -- minimized when load is uniform. The readout shows the per-expert
  counts, the imbalance, and the drop count.

**Drag** any router-gate cell to re-score a token and watch the load (and drops)
shift; **hover** a gate cell or an expert bar for its detail; step the transport
to route token by token (auto-plays + loops).

## Render tier

T2 (WebGL2 heatmap for the gate matrix; Canvas2D for the load bars, capacity
line, routing arrows, and the balance readout).

## Wiring

`layout.mount()` + controls (`N`, `E`, `k`, `capacity`, `seed`) + an `N`-step
`Transport` (route token by token) + `onPointer` gate-cell drag + hover.
`?step` / `?drag` / `?cap` / `?hover` hooks. Source: [`page.js`](page.js).
