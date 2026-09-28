# cross-layer KV — later layers reuse an earlier cache

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

Every `share`-th layer writes keys and values. The layers until the next writer reuse that cache and compute only a query. The tail of the stack still gets a writer, so the writer count is the layer count divided by the share, rounded up.

Writers plus readers is the layer count. The cache is the writer count out of the layers. A share of 1 means every layer writes its own cache.
