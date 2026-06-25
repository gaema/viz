# rope -- rotary position embedding (per-pair 2-D rotation)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: how RoPE encodes position by rotating each dimension-pair of a
query/key vector by an angle proportional to position. **Anchor**: A1 RoPE
shape (Family B8; see `../plan/curriculum.md`).

## What it shows

A `d`-dim vector is split into `d/2` pairs `(x₂ᵢ, x₂ᵢ₊₁)`. At position `p`,
pair `i` is rotated in its own 2-D plane by `Δ = p · θᵢ`, where the per-pair
frequency `θᵢ = base^(−2i/d)`. The main view is `d/2` little 2-D planes — each
shows the original pair (gray), the rotated pair (blue), and a **spiral tracing
the total sweep** `p·θᵢ`. Low pairs (high frequency) spin fast as you scrub the
position; high pairs (low frequency) barely move — the multi-scale positional
code. A supporting `[N×d]` heatmap shows the rotated components across **all**
positions (the RoPE "wave"), with the current position's row outlined.

Because rotations compose (`R(mθ)ᵀR(nθ) = R((n−m)θ)`), the query·key dot product
ends up depending only on the **relative** position `m−n` — noted in the
readout. Sliders for `head_dim`, the rotary `base`, and `seed`.

## Render tier

T2 (WebGL2 heatmap for the across-positions wave; Canvas2D overlays for the
rotation planes, arrows, and spirals).

## Wiring

`layout.mount()` + controls (`head_dim`, `positions`, `base`, `seed`) + a
per-position `Transport`, drawn with `render.heatmap`/`arrow` + `ctx` and
`tensor.rope`/`ropeAngles`/`seededRandn`. Direct-manipulable per the interaction
contract: **drag horizontally** (or the position track) to move the token
position so every pair’s `Δ = p·θᵢ` updates live; **hover** a rotation plane for
its `θᵢ` / `Δ` / `(cos,sin)` / rotated `(x,y)` or a heatmap cell for its value;
when not dragging the position **sweeps on its own** (`animate`). Headless hooks:
`?pos=N` (set position + freeze the sweep), `?hover=x,y` (fake the cursor for the
tooltip), plus the existing `?step=N`/`?play=1`/`?theta`. Source: [`page.js`](page.js).
