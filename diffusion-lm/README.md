# diffusion language model — unmask the whole sequence, both directions

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

A diffusion language model starts with every position masked. Each step scores every still-masked position from the whole sequence, including positions that come later, and reveals the highest score. The next step sees the updated tokens and the updated mask. It does not receive a causal cache from the step before.

While any position stays masked, the masked count falls by one. Once the mask is empty, another step changes nothing.
