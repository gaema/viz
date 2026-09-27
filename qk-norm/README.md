# QK-norm — a long key stops winning by being long

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

The query is `[1, 0]`. The aligned key is the same vector. The off-axis key is `[scale, scale]`.

Raw scores are the bare dot products. QK-norm divides each vector by its RMS first. The mass beside each score is the softmax of those two printed scores. At scale 3 the raw winner is the off-axis key and the normalized winner is the aligned key.
