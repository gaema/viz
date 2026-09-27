# attention / expert split — a wire at every layer

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

KV shipped once is `2 × layers × 4 × seq × 2`. Hidden states shipped to the experts and back, for one step, are `2 × layers × batch × 8 × 2`.

The first grows with the context and ignores the batch. The second grows with the batch and ignores the context. Which bar is taller is whichever of those two products is larger.
