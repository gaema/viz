# quantization-aware training — round in the forward, step the full weight

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

The forward value is the high-precision weight after it is rounded onto a grid and scaled back. At weight 1.20 and scale 0.50 the code is 2 and the forward value is 1.00.

The backward step treats that rounding as slope one, so the stored weight moves by the full step. A step of 0.05 takes 1.20 to 1.15, and 1.15 is still code 2. The forward value changes only when the weight crosses a rounding boundary.
