# hybrid-by-layer -- interleaved SSM + periodic full attention

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how **hybrid** models (Jamba, Zamba, Qwen3-Next, …) stack
cheap **SSM / linear-attention** layers and sprinkle in **full-attention** layers
every so often -- and exactly **where the KV cache lives and where it doesn't**.
**Anchor**: N3 hybrid sequence architecture (Family E; see
`../../design/emerging-architectures.md`).
Builds on [ssm-scan](../ssm-scan/README.md) / [gated-deltanet](../gated-deltanet/README.md)
(the SSM layers) and [kv-cache](../kv-cache/README.md) (the attention layers).

## What it shows

A stack of `N` layers. Most are **SSM** layers (recurrent constant-size state,
`O(L)` per token, **no KV cache**); every `P`-th layer is a **full-attention**
layer (`O(L²)`, a **KV cache** `[L × 2·d]`). The interleave pattern -- e.g. one
attention every 5–7 layers -- is the whole design knob.

- **Where KV exists**: only the attention rows hold a KV cache (shown as a cache
  strip); the SSM rows carry just a small recurrent state (a flowing line). A
  forward-pass pulse rises through the stack.
- **The memory win**: total KV cache = `#attn · L · 2·d`, a fraction
  `#attn / N` of an all-attention model. The rollup shows the KV total, the
  saving vs full attention, and the SSM/attention compute mix.
- **The trade-off**: more attention layers ⇒ better exact long-range recall but
  more KV memory + `O(L²)` compute; fewer ⇒ cheaper but the model leans on the
  SSM state to summarise the past.

**Drag** the attention-period slider to steer the pattern (and the KV memory)
live; **click** any layer to flip it SSM↔attention by hand; **hover** any layer
to inspect its type, KV size, and cost.

## Render tier

T1 (Canvas2D: the layer stack, the per-layer SSM-state lines / attention KV-cache
strips, the forward-pass pulse, and the memory rollup).

## Wiring

`layout.mount()` + controls (`N`, `P`, `L`, `d`) + `animate` (the forward-pass
pulse) + `onPointer` layer click-to-toggle + the period slider (drag-to-steer) +
hover. `?p` / `?toggle` / `?hover` hooks. Source: [`page.js`](page.js).
