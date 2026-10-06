# structured sparsity — two nonzeros in every four

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

Each group of four neighbouring weights keeps at most two nonzeros. The two largest magnitudes stay, and a tie keeps the earlier index.

The group 0.90, 0.20, -0.80, 0.10 keeps 0.90 and -0.80. Their magnitudes add to 1.70, and the other two entries become zero. A group that already has two or fewer nonzeros drops nothing.
