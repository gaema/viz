# kv fusion — reuse a chunk that is not a prefix

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

A prefix cache reuses a stored chunk only when the request starts with it. The same chunk later in the request is recomputed in full.

Fusion keeps that chunk. It recomputes `fraction × chunk` tokens and always recomputes the new header in front of the chunk. The residual error is `(1 − fraction) × mismatch`. On a true prefix the prefix cache pays nothing for the chunk, and fusion still pays its fraction.
