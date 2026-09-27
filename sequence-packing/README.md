# sequence-packing — many documents, one row

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: packing several short sequences into one fixed-width
training row, and the block-diagonal mask that keeps the packed documents
independent. **Anchor**: A3 attention pattern (Family B).

## What it shows

Three documents of lengths `[2, 3, 1]` share **one row of six tokens**, drawn
as a 6×6 attention mask. In **block-diagonal** mode a cell `[i,j]` is allowed
only when tokens `i` and `j` belong to the same document — the mask is a
staircase of square blocks along the diagonal and the cross-document pair
count is **0**, so each document trains as if it were alone in the row.
Switch to **full** mode: the mask is true everywhere, training still runs,
the count becomes positive — every off-block pair is a token quietly reading
context across a document boundary. The block outlines show the packing
structure; a dashed outline marks an allowed cross-document cell. Hover any
cell for the per-pair verdict.

The mask and the pair count come from [`math.js`](math.js)
(`packMask`, `crossDocumentPairs`), checked by `math.test.mjs`.

## Render tier

T1 (Canvas2D — a grid + labels).

## Wiring

`layout.mount` + one `select` control (`mask mode: block | full`), mask math
imported from [`math.js`](math.js). The control state deep-links via the
shared URL sync (`?mode=full`). Source: [`page.js`](page.js).
