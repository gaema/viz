# RoPE splice — rotate a cached chunk to its new position

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

Pair `i` of a key is rotated by `position × base^(−2i / dim)`. A key stored at the source position still carries that angle. Rotating it again by the gap `destination − source` matches a key computed at the destination.

The score drawn for a query at the destination is the cosine of the printed angle difference. The unrotated key uses the source angle. The spliced key uses the destination angle, so its score matches a native key. The move error is how far the rotated vector sits from that native key.
