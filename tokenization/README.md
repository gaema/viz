# tokenization -- Byte-Pair Encoding builds the vocab

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: **Byte-Pair Encoding (BPE)** -- the step that turns text into the
integer token IDs a model consumes, the step *before* [`embedding`](../embedding/README.md).
**Anchor**: C (transformer block input). Family C (see `../plan/curriculum.md`).

## What it shows

BPE starts from raw **characters** and repeatedly does one thing: find the most
frequent **adjacent pair** of symbols across the whole corpus (weighted by each
word's frequency), **merge** that pair into a single new symbol, and record a
**merge rule**. Frequent letter sequences (`"e"+"s"→"es"`, `"es"+"t"→"est"`) thus
become single tokens — the **vocabulary grows** and the **token count shrinks**.

- left: each corpus word as its current sequence of symbol boxes (the just-merged
  symbol highlighted), with a draggable `×freq`;
- right-top: the **pair-frequency table** at this step, the winner (next/just
  merged) highlighted;
- right-bottom: the **merge rules** learned so far, in order;
- bottom: the **vocab size**, the **token count** (vs the original character
  count), and the compression.

This is exactly how GPT / LLaMA tokenizers are **trained**; at inference the learned
merge rules are simply **replayed** on new text. The trailing `_` marks a word
boundary (so a merge can't cross between words).

**Step / play** the transport to watch the merges form; **drag** a word's `×freq`
↕ to change which pairs win (make `lowest` frequent and watch `est` merge sooner);
pick a **corpus**; hover a pair for its count.

## Render tier

T1 (Canvas2D: the symbol-segmentation rows + the pair-frequency bars + the
merge-rule list + counts).

## Wiring

`layout.mount()` + a `transport` over the merge steps + a `corpus` select +
`onPointer` drag-the-frequency + hover. `?corpus` / `?step` hooks. Source:
[`page.js`](page.js).
