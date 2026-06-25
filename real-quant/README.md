# real-quant -- int4 on actual GPT-2 weights

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server + network (range-fetches ~9 MB): `python3 -m http.server 8099`

Interactive page: the **real** counterpart to the synthetic
[`quantization`](../quantization/README.md) page. It runs group-wise integer
quantization on **actual GPT-2 weights** and shows the real error. **Anchor**:
quantization. Part of **real-model grounding**.

## What it shows

Pick a weight tensor. The page **range-fetches just that tensor** (~9 MB via HTTP
Range — `fetchTensor()` reads the safetensors header, then only that tensor's
bytes, not the whole 548 MB file) and draws:

- its **distribution** — real GPT-2 weights are a tight Gaussian (σ≈0.14) with a
  few big outliers (to ~4.6), with the int-N quantization levels overlaid; and
- **RMSE vs bits** — the quantization error at 2/3/4/5/6/8 bits for the chosen
  group size, the current setting highlighted.

Drag **bits** and **group size**: more bits and smaller groups (each block gets
its own scale, so one outlier doesn't coarsen the rest) drop the error fast —
exactly why int4/int8 quantization works. This grounds the abstract
`quantization` page in numbers from a real model.

## Verified against numpy

The quant math ([`quant.js`](quant.js)) and the range-fetch
([`fetchTensor()`](../real-attention/gpt2.js)) are checked against numpy by
[`quant.test.mjs`](quant.test.mjs) on the real `h.0.mlp.c_fc.weight` tensor: the
Range-read values, the stats (σ=0.141, range [−2.31, 4.59]), and the group-quant
RMSE (int4/int8 × group 64/128) all match the reference (fixture
[`quant-groundtruth.fixture.json`](quant-groundtruth.fixture.json)). Run it:
`WEIGHTS_URL=https://huggingface.co/gpt2/resolve/main/model.safetensors node real-quant/quant.test.mjs`

## Render tier

T1 (Canvas2D histogram + bar chart).

## Wiring

`layout.mount()` + a `tensor` select + `bits` / `group` controls + "load real GPT-2
weights". Reuses `fetchTensor()` from [`../real-attention/gpt2.js`](../real-attention/gpt2.js);
async, degrades to a labelled synthetic Gaussian+outliers stand-in offline. Two
**challenges** (`?ch=N`): range-fetch a real tensor, and get RMSE < 0.005.
Headless hooks: `?tensor=`, `?bits=`, `?group=`, `?real=0`. Source:
[`page.js`](page.js), [`quant.js`](quant.js).
