# logit-lens -- watching a prediction form across layers (Phase 9)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server + a one-time ~548 MB weight download: `python3 -m http.server 8099 --directory viz`

Interactive page: the **logit lens** (nostalgebraist) on a **real GPT-2 (124M)**.
It applies the model's final `ln_f` + tied lm_head to the residual stream at
**every** layer — not just the last — so you can watch the next-token prediction
crystallize with depth. **Anchor**: residual stream + A5 lm_head. Part of **Phase
9 — real-model grounding** (`../plan/phase9.md`).

## What it shows

Type a prefix. The page runs the verified GPT-2 forward
([`../real-attention/gpt2.js`](../real-attention/gpt2.js), `GPT2.lens()`), capturing
the residual stream after the embedding and after each of the 12 blocks (13
depths). It applies `ln_f` + the tied lm_head to each, giving a next-token
prediction *at every depth*. The rows show, bottom-up (embedding → layer 12), each
depth's top-1 token and a bar = the **final** token's probability at that depth —
the convergence curve.

The default `the cat sat on the mat . the cat ran` walks
`ran → running → through → away → … → up`: early layers echo the last token, the
middle wanders, and the answer locks in near the top. Confident prompts lock in
early; open-ended ones only at the final layer.

## Verified against PyTorch

`GPT2.lens()` is checked against `output_hidden_states` by
[`../real-attention/gpt2.test.mjs`](../real-attention/gpt2.test.mjs): the per-layer
top-1 token ids at all 13 depths match PyTorch exactly. (Subtlety the test pins:
HF's `hidden_states[-1]` is already post-`ln_f`, so the reference must use the
model's true final logits for the last depth rather than applying `ln_f` twice —
`GPT2.lens()` captures the *pre*-`ln_f` residual and applies `ln_f` once, which is
correct.)

## Render tier

T1/T2 (Canvas2D rows; the "compute" tier is the GPT-2 forward + 13 lm_head
read-outs in JS).

## Wiring

`layout.mount()` + `controls.text('prompt')` + "load real GPT-2". Reuses the
verified [`gpt2.js`](../real-attention/gpt2.js) loader; async, degrades to a
labelled idealized synthetic trajectory offline. Two **challenges** (`?ch=N`): load
the real model, and find a confident prediction (final-layer top token > 50%).
Headless hooks: `?prompt=`, `?ids=` (inject token ids), `?real=0`. Source:
[`page.js`](page.js).
