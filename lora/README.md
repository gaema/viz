# lora — a low-rank update instead of a full one

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099` in this directory

Interactive page: fine-tuning one layer by training **two thin matrices** instead
of the layer itself. The update is their product, `ΔW = B·A`, so the number of
trained values is `r·(din + dout)` instead of `din · dout` — and the price for
that is that the product has rank at most `r`, so it can only write an
`r`-dimensional slice of the layer.

Source (public paper): Hu et al., **LoRA: Low-Rank Adaptation of Large Language
Models**, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

## Scope — what this page does NOT re-teach

- [matmul](../matmul/README.md) owns the matrix product the frozen layer already
 runs, and [qkv-projection](../qkv-projection/README.md) owns one such layer
 inside a block.
- [quantization](../quantization/README.md) owns making the *frozen* weights
 smaller. LoRA does not shrink the layer at all — it shrinks what you
 **train**, which is a different saving bought a different way.

This page holds a single frozen `din × dout` layer fixed at 8 × 8 and moves only
the two things LoRA actually asks you to choose: the rank, and whether the
adapter is folded in.

## The mechanism

Freeze `W`, and train `B` (`dout × r`) and `A` (`r × din`) instead:

 ΔW = B · A rank(ΔW) ≤ r ≤ min(din, dout)
 y = x·W + x·(B·A)

`B·A` is a sum of `r` rank-1 terms, which is why the adapter bar is drawn as `r`
equal bands: each term contributes one row of `A` (`din` numbers) and one column
of `B` (`dout` numbers), so every band costs the same `din + dout` and the whole
adapter costs `r·(din + dout)`.

Two consequences the two bars report at the same time:

| Rank | Trained numbers | What the update can reach |
|---|---|---|
| small `r` | `r·(din+dout)`, a small share of `din·dout` | `r` of the layer's `min(din,dout)` directions |
| `r = din·dout / (din+dout)` | equal to `din·dout` — the break-even line | still only `r` directions, for the same price |
| `r = min(din, dout)` | more than `din·dout` | every direction — the constraint has dissolved |

On an 8 × 8 layer break-even is `64 / 16 = 4`: at rank 4 the blue bar tops out
exactly on the dashed line, above it the adapter is the *costlier* of the two,
only at rank 8 is `rank(B·A) = 8`, i.e. as free as a full update.

## The serving choice, which is a separate question

Merging changes the arithmetic but not the picture — the trained numbers are the
same either way, so the toggle deliberately does not move the bars:

| | what runs per token | extra cost |
|---|---|---|
| **merged** | one GEMM against `W + B·A` | **0** |
| **unmerged** | the base GEMM, then the thin side path `x·A` then `·B` | `r·(din+dout)` MACs — `+50%` over the base `din·dout` at rank 2 |

Kept unmerged, an adapter can be swapped per task without touching the base
weights. Merged, it costs nothing forever and cannot be swapped at all. That
trade, not the parameter count, is what the toggle is for.

## Interactions

- **Rank slider** `r = 1…8` — both bars, the `r` bands and the rank readout all
 move together.
- **Merged toggle** — switches which serve-cost row is lit; the bars stay put.
- **Hover** — the blue bar gives its count, its `A`/`B` shapes and how many of
 the layer's directions it reaches; the teal bar gives `din·dout` and marks the
 level the dashed line is drawn at.
- **Challenge mode** — three goals: find the rank where the adapter stops being
 the cheap option, give the adapter the whole layer back, and serve with zero
 extra arithmetic at rank 4 or higher.

## Render tier

T1 (Canvas2D). Two bars and a small cost panel; there is no large tensor here.

## Wiring

`layout.mount` + controls (`r`, `merged`) + hover tooltips + `challenges`; the
arithmetic is imported from [`math.js`](math.js), which is unit-tested by
`math.test.mjs`. URL hooks are the control values themselves:
`?r=5` and `?merged=0`. Source: [`page.js`](page.js).
