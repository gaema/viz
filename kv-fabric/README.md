# kv fabric — one pool across machines

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

Each arrival of prefix `p` in wave `r` is assigned to machine `(p + r) mod machines`.

A private cache hits only when that machine has already seen the prefix. A shared pool hits when any machine has. The percent on each bar is `100 × hits / requests` from the two counts printed beside it. One machine makes the two counts equal. A later wave on a different machine is a shared hit and a private miss.
