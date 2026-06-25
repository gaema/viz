# real-embeddings -- a trained model's semantic geometry

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server + network for the real model: `python3 -m http.server 8099`

Interactive page: the **real** counterpart to [`embedding/`](../embedding/README.md).
The synthetic page shows the *mechanism* (a token id is a row lookup in a seeded
table `E[V×D]`); this page shows the numbers are **real** by fetching a small
trained sentence-embedding model and running it in your browser on the words you
type. **Anchor**: A1 token embedding (same attribute as the synthetic sibling).
First increment of **real-model grounding**.

## What it shows

You type a list of words. Each becomes a unit vector from
[`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (≈23 MB,
384-dim, mean-pooled + L2-normalized). Two views of the **same real numbers**:

- **cosine-similarity heatmap** — `cosine(i,j) = vᵢ·vⱼ` (a dot product, since the
  vectors are unit norm — the very same inner product the attention pages use,
  here on *trained* vectors). Hover a cell for the exact value.
- **2D map (MDS / PCA scores)** — classical MDS on the centered vectors (top-2
  eigenvectors of their double-centered Gram). Words that mean similar things sit
  close: `king·queen·prince`, `paris·tokyo·berlin`, `cat·dog·horse` form visible
  clusters with **no demo trickery** — that structure is in the trained weights.

The point: every other page runs *real math on synthetic
weights*; here both the math **and** the weights are real.

## Real model, fetched at runtime (no build, no committed weights)

The model is loaded via [transformers.js] as a CDN ES module and cached by the
browser on first open — same no-build ethos as every other page, and nothing
heavy enters git (per the viz CLAUDE.md "fetch them at runtime"
rule). A banner shows the state: `downloading… NN%` → `running…` →
`● real model`.

**Offline / no network** (`file://` with no fetch, CDN blocked): the page never
goes blank — it renders a clearly-labelled **synthetic stand-in** (the default
clusters are hand-seeded so it still teaches) and swaps in the real vectors the
moment the model finishes downloading.

[transformers.js]: https://github.com/huggingface/transformers.js

## Render tier

T1 (Canvas2D — similarity heatmap + scatter). The "compute" tier is the real
model running under WASM/WebGPU inside transformers.js.

## Wiring

`layout.mount()` + `controls.text('words')` (free input) + two action buttons
(`re-embed`, `load real model`), drawn with `render.heatmap`/`grid` + `ctx`. The
real run is async and never throws into `mount()` — failures degrade to the
synthetic stand-in. Two **challenges** (`?ch=N`): switch to the real model, and
type a tight single-theme cluster (min pairwise cosine ≥ 0.30). Headless hooks:
`?words=a,b,c` (override the list), `?real=0` (suppress the download — fast
deterministic synthetic capture), `?hover=x,y` (fake cursor for the tooltip).
Source: [`page.js`](page.js).
