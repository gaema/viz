# forward-pass -- one prompt → the next token, end to end

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: **the capstone.** Every other page shows one mechanism in
isolation; this one **chains them**, running a short prompt all the way through a
tiny (but real) transformer to predict the next token. **Anchor**: synthesis of
Families A–G; Family H (see `../plan/curriculum.md`).

## What it shows

A toy transformer — `d=8`, 1 layer, 2 heads, gated MLP, tied lm-head, random
weights — runs end to end, and you watch the **same residual stream** transform at
each stage (the strip on the left is the through-line):

1. **tokenize** — prompt words → integer IDs (vocab indices).
2. **embed** — `X₀[i] = E[id[i]] + position[i]` (a table lookup + a positional
   offset) → the residual stream.
3. **attention** — `softmax(QKᵀ/√dₕ)` (causal) · `V`, projected by `Wₒ` and
   **added back**: `X₁ = X₀ + attnOut`. The `[n×n]` attention matrix is shown.
4. **gated MLP** — `down( silu(x·W_g) ⊙ (x·W_u) )`, **added back**:
   `X₂ = X₁ + mlp`.
5. **lm-head** — the **last** token's final (normed) hidden is projected onto
   every vocab embedding → one logit per word.
6. **sample** — `softmax(logits / T)` → the next token.

The recurring idea — every block **reads the stream, computes a correction, and
adds it back** (the residual stream) — is made literal: the left strip shows `X₀ →
X₁ → X₂` as you step.

> The weights are **random**, so the predicted word is **arbitrary** — this page is
> about the *mechanism*, not a trained model's answer.

**Step** the transport (or **play**) to walk the six stages; **drag a prompt
token** ↕ to swap its word and watch the whole pipeline recompute; tune
**temperature** and the **attention head**; hover any tensor.

## Render tier

T2 (Canvas2D: the stage ribbon, the persistent residual-stream heatmap, and the
per-stage detail panels — attention matrix, gated-MLP bars, logits, sampling
distribution).

## Wiring

`layout.mount()` + a 6-step `transport` (the stages) + controls (`prompt`, `temp`,
`head`, `seed`) + `onPointer` drag-a-token + hover. `?step` / `?prompt` / `?temp` /
`?head` / `?seed` hooks. Source: [`page.js`](page.js).
