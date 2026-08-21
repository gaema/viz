# kv-quant -- quantizing the KV cache, not the weights

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **group-wise quantization of the KV cache** -- cached keys
values stored as 4-bit or 8-bit codes with a scale per group, dequantized on
read -- and what the **grouping axis** does to the reconstruction error.
**Anchor**: G (quantization / deployment); Family G.

## Why the KV cache is the right target

Weights are a **fixed** cost: one file, sized once at load. The KV cache is the
term that **grows** -- one K vector and one V vector per token, per layer, per
kv-head, held for as long as the conversation runs:

```
KV bytes = 2 · layers · kv_heads · head_dim · bits/8 · tokens
```

The page prices this against a memory budget you set, using a generic mid-size
open LM shape (32 layers, 8 kv-heads, head_dim 128, ~8B parameters at ~4.5-bit
weights). At fp16 KV, that cache passes the size of the **entire quantized
weight file** at a few tens of thousands of tokens -- the number is computed live
on the page and moves with the budget slider. Past that point the cache, not the
model, is what decides how much context fits.

So: **weight quantization buys model size, once. KV quantization buys context
length, every token.** Same arithmetic, different tensor, different consequence.
The sibling [`quantization`](../quantization/README.md) page is the weight half.

## The mechanism

Identical group-wise affine form as the weight page, applied to cached K and V:

```
s = (max − min) / (2^bits − 1) per group
z = round(−min / s)
q = clamp(round((x − min)/s), 0, 2^bits−1) the stored code
x′ = min + s·q what attention reads back
```

Three heatmaps run across the top: the original fp16 slice, the reconstruction
`x′`, and `|x − x′|`. The **per-group scales are drawn explicitly** as their own
strip, laid out in the same orientation as the grouping, so "which elements
share one scale" is a thing you can see rather than infer. Below them, one
selected element is taken apart all the way down to its stored bits: the value,
the group's `min`/`max`/`s`/`z`, the integer code in binary, the reconstruction
and the error.

Storage is charged honestly: `bits` per element **plus** the group's own 16-bit
scale and 8-bit zero-point amortised over its members, so a small group shows up
as a real overhead rather than a free lunch. The readout reports the effective
bits/element for the live slice, which is why an almost-empty per-channel group
early in decode reads as *exact but expensive*.

## The axis is the interesting knob

A group has to be cut somewhere, and there are two natural directions:

| Axis | A group is | Scale is final… |
|---|---|---|
| **per-token** | `G` channels of **one** token (down a column) | the moment the token is written |
| **per-channel** | `G` tokens of **one** channel (along a row) | never — a later token can widen the group's range |

Because K and V do not hold their large values in the same layout, the axis you
pick moves the error a long way. Cut a group **along** the direction the large
values line up with and the group spans a narrow range, so its scale stays
small and every member stays fine-grained. Cut **across** it and one large value
stretches the group's `[min, max]`, stretches `s`, and coarsens every neighbour
that shares that scale.

The page shows this three ways, all live: a 2×2 table of RMSE (K and V × both
axes) with the better axis marked, an RMSE-vs-bits curve carrying **one line per
axis**, and the A/B compare button, which renders the whole page twice -- once
per axis.

**The slice on screen is synthetic and deliberately structured** (a
channel-aligned band of large values in K, a token-aligned one in V) so a 10×20
slice can show the effect at all. Which axis wins for a given real model is a
property of that model's tensors: something to measure, not to assume. Published
KV-cache-compression work does treat the two halves asymmetrically -- for
instance, TurboQuant (arXiv:2504.19874) rotates KV vectors precisely so that
large values *stop* concentrating in particular coordinates before per-coordinate
quantization is applied, and the `shard` KV compressor pairs a low-rank treatment
of keys with a rotated vector-quantized treatment of values.

## The tradeoff, all three axes live

- **bits per element** -- the slider, 2 to 8. More bits = more levels = finer
 steps = less error.
- **reconstruction error** -- RMSE and max |err| for the live slice, plus the
 per-axis curve.
- **context that fits** -- horizontal bars for fp16 / 8-bit / 4-bit / current,
 each labelled with the token count reachable in the budget after the weights
 are paid for, and the effective bits/element that produced it.

## Interactions

- **Transport** -- decode auto-plays and loops; each step writes one more K/V
 column, so the cache fills, the scales appear, and the memory bars move.
 `?step=N` / `?pos=N` seek it; `?play=1` starts it.
- **Direct manipulation** -- press any written cell and drag ↕ to change its
 value. Push one into an extreme and watch its group's scale stretch and every
 neighbour sharing that scale get coarser. `?drag=r,c,value` (`;`-separated for
 several) is the headless stand-in.
- **Hover** -- any cell reports the full derivation: original value, which group
 it is in and how many elements share that scale, the group's `min`/`max`/`s`/`z`,
 the code in decimal and binary, the reconstruction and the error.
- **Controls** -- tensor (K / V), grouping axis, bits, group size, context
 length, memory budget, seed. Every one has a `?key=` URL hook, plus `?sel=r,c`
 to pin the bit-level panel and `?hover=x,y` to fake a cursor.
- **Challenges** -- find a setting where one axis costs ≥ 1.5× the other's error;
 reach 3× the fp16 context while keeping RMSE under 0.08.

## Render tier

T1 (Canvas2D: three heatmaps, the scale strip, the bit boxes, the memory bars
the per-axis error curve).

## Wiring

`layout.mount` + `controls` (`tensor`, `axis`, `bits`, `G`, `ctx`, `budget`,
`seed`) + `transport` (autoplay, loop) + `animate` (the group scan) + `onPointer`
drag + hover, `compare` on the grouping axis, two `challenges`. Colours come from
`framework/theme.js` tokens, read at draw time. Source: [`page.js`](page.js).
