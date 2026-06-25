# real-generate -- watch GPT-2 write

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server + a one-time ~548 MB weight download: `python3 -m http.server 8099`

Interactive page: the **capstone** of the real-* family — a **real GPT-2 (124M)**
writing text in your browser, one token at a time. **Anchor**: A5 lm_head + the
sampling + prefill-vs-decode concepts. Closes **real-model grounding**.

## What it shows

Autoregressive generation is just the verified next-token step in a loop:
**logits → sample a token → append it → repeat**. Type a prompt, hit **generate**,
and watch the model write, token by token, the new tokens highlighted as they land.

- **greedy** (argmax) — deterministic; the same prompt always gives the same text.
  `the cat sat on the mat . the cat ran` → ` up and down the hall and down the
  hall …` (GPT-2's famous greedy loop).
- **sampling** — turn greedy off for temperature / top-k / top-p with a **seed**;
  change the seed for different continuations from the same prompt.

It reuses [`../real-attention/gpt2.js`](../real-attention/gpt2.js) `logits()` —
the same forward verified against PyTorch — so the generation is real, not faked.

## Verified against PyTorch

Greedy generation is deterministic, so it must reproduce HF
`model.generate(do_sample=False)` exactly.
[`../real-attention/gpt2.test.mjs`](../real-attention/gpt2.test.mjs) runs the JS
loop (argmax of `logits()` → append → repeat) and checks **all 12 generated token
ids match** the reference.

## Render tier

T2 (Canvas2D streaming text; the "compute" tier is the GPT-2 forward per token in
JS — one full forward per generated token, no KV cache).

## Wiring

`layout.mount()` + `controls.text('prompt')` + `tokens` / `greedy` / `temperature`
/ `top-k` / `top-p` / `seed` + a generate/stop button + "load real GPT-2". Sampling
uses a seeded RNG (reproducible). Async streaming loop; degrades to a labelled
synthetic continuation offline. Two **challenges** (`?ch=N`): load the real model,
and generate ≥ 12 tokens. Headless hooks: `?prompt=`, `?ids=`, `?gen=N` (run
synchronously, no streaming delay), `?greedy=0|1`, `?temp=`/`?topk=`/`?topp=`/
`?seed=`/`?ntok=`, `?real=0`. Source: [`page.js`](page.js).
