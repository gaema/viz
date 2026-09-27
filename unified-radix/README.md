# unified radix — one match, three kinds of state

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

One matched prefix feeds three stores.

Full-attention KV covers every matched token. Sliding-window KV covers `min(match, window)` tokens, the tail of the match, and never tokens the match did not include. A recurrent layer reuses one checkpoint, and only when the match is longer than nothing. An empty match reuses none of the three.
