# ulysses — all heads become a full sequence on fewer heads

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

The sequence starts split across devices, with every head on each device. One exchange hands each device the full sequence and an equal share of the heads. Attention on those heads matches attention computed on one device for the same heads. A second exchange returns the sequence shards, now carrying every head.

Four heads on two devices is two heads per device. Three heads on two devices is refused, because the heads do not divide evenly.
