# gated-deltanet -- the linear-attention delta-rule state update

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: the **GatedDeltaNet** recurrence -- a linear-attention variant
that carries a **matrix** state (an associative key→value memory) and updates it
with a **gated delta rule** (Qwen3-Next / Gated DeltaNet hybrids). **Anchor**: N1
sequence mixer (linear attention; constant-size matrix state instead of a KV
cache); Family E, companion to [ssm-scan](../ssm-scan/README.md) and
[mamba-block](../mamba-block/README.md). See
`../../design/emerging-architectures.md`.

## What it shows

Instead of attention's `softmax(QKᵀ)V` (which re-reads the whole KV cache),
linear attention keeps a fixed `[d×d]` state matrix `S` -- a sum of key→value
associations -- and reads it with the query: `oₜ = Sₜ qₜ`. **GatedDeltaNet**'s
update rule:

    Sₜ = αₜ · Sₜ₋₁ · (I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ

- **αₜ (gate / decay)**: shrinks the whole memory each step (forgetting) -- the
  "gated" part.
- **delta rule (βₜ)**: `(I − βₜ kₜ kₜᵀ)` first **erases** the value currently
  stored at key `kₜ` (the term `−βₜ (Sₜ₋₁ kₜ) kₜᵀ`), then `+ βₜ vₜ kₜᵀ` **writes**
  the new value `vₜ` there. So a repeated key **overwrites** instead of piling up
  -- the memory doesn't saturate (vs plain linear attention `Sₜ = Sₜ₋₁ + vₜ kₜᵀ`).

The state matrix `S [d×d]` is the centrepiece; it evolves as you step the
sequence. Each step shows the token's `k / v / q`, the gate `α` and write `β`,
the written association `βvkᵀ`, and the output `o = S q`. Toggle the delta rule
off to watch the memory saturate.

**Drag** a `v` (value) component to change what gets written and watch `S` + the
output re-run; **hover** any state cell, k/v/q, or output. The scan auto-plays +
loops.

## Render tier

T2 (WebGL2 heatmaps for the `[d×d]` state + written-association matrices;
Canvas2D for the k/v/q/o bars and the formula).

## Wiring

`layout.mount()` + controls (`L`, `d`, `alpha`, `beta`, `delta`, `seed`) + an
`L`-step `Transport` (the state evolves over the sequence) + `onPointer` value
drag + hover. `?step` / `?drag` / `?delta` / `?hover` hooks. Source:
[`page.js`](page.js).
