# fill-in-the-middle — the middle can see the ending

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

One example is ordered prefix, then suffix, then middle. A causal mask on that order lets the first middle token see every suffix token.

Two examples under one causal mask add a cross pair for every token of the first example seen by every token of the second. That count is the square of one example's length. Isolating the examples sets it to 0, and each middle token still sees its own suffix.
