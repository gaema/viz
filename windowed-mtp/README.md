# windowed MTP — the draft does not read the whole context

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

The draft reads `min(length, sink)` opening tokens and `min(length, window)` recent tokens. Overlap is counted once.

A fact inside that span is accepted at 1. A fact in the gap is accepted at the blind rate. The target still sees the whole context, so a rejected guess is not what the model answers. The draft cache is the visible count, not the context length.
