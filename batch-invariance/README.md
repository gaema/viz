# batch-invariance — your reply depends on who else was batched with you

An interactive page about one uncomfortable fact: a language-model server can
give you a different answer to the *same* prompt, with the *same* weights, the
*same* seed and greedy decoding, because of **who else's requests were in flight
at the same time**. That is not a bug in the sampler and not hidden randomness.
It is floating-point arithmetic doing exactly what it is specified to do.

Open `index.html` over a static server (ES modules are blocked on `file://`):

```sh
python3 -m http.server 8099
# then open http://localhost:8099/batch-invariance/
```

## The mechanism

Floating-point addition is **not associative**. `(a + b) + c` and `a + (b + c)`
are different numbers in general, because each `+` rounds its result to the
nearest representable value, and rounding twice in a different order does not
land in the same place.

A GPU does not add a row left to right. A reduction kernel splits the row across
some number of parallel lanes, sums each lane, and then combines the lane totals
in a tree. **How many lanes it picks is a function of the launch shape** —
the launch shape depends on how many requests are being served at once. More
requests in flight means fewer lanes per row, which means a different chunking,
which means a different addition order, which means different last bits.

Then something downstream *thresholds* on those bits. Greedy decoding is exactly
such a threshold: take the largest logit. When the top two logits are close —
in a real vocabulary they very often are — the winner is decided by the bits the
reduction order moved. One token changes, and from there the two continuations
diverge completely.

So the chain is: batch size → launch shape → reduction split → addition order →
last bits of the logit → argmax → a different reply.

## Everything on this page is computed, not stored

There is no recorded "difference" anywhere in the source. JavaScript numbers are
IEEE-754 binary64, and `Math.fround` rounds to binary32 after each operation,
so the accumulators carry exactly the bits a float32 kernel would carry. The page
builds one real row of float32 values and sums it **twice**:

- **sequential** — one accumulator, left to right, rounding after every add;
- **split** — contiguous chunks summed per lane, then combined pairwise up a
 tree, with the number of lanes derived from the batch size.

The bit strips are read straight out of the same `Float32Array`/`Uint32Array`
pair that holds the values, so a displayed bit pattern cannot disagree with the
number above it. The first differing bit is `Math.clz32(bitsA ^ bitsB)`, the
distance is the difference of the two bit patterns read as integers, and the
token flip is a real `argmax` over a real softmax.

Because it is computed, it is also allowed to come out **negative**: the "all
ones" preset gives bit-identical sums at every batch size, and some batch sizes
on the other presets happen to re-round back onto the sequential answer. A page
that could only ever show a difference would be showing a rigged example, not
arithmetic.

## The three views

| View | What it shows |
|---|---|
| **the reduction (bits)** | The row, the lane split, the combine tree, the two sums, and their 32 bit cells side by side with the differing bits marked. |
| **greedy argmax (the flip)** | Four candidate logits under both orders, their softmax probabilities, and which token greedy decoding actually picks. |
| **the cost of invariance** | The batch sweep: what each batch size makes the kernel answer, and how much of the lane pool a fixed split leaves idle. |

## Things to try

- **Press play.** The transport walks the reduction: each lane accumulating, then
 each combine. The sequential accumulator advances on the same element counter,
 so both orders are visible at once and you can watch them separate.
- **Drag the batch strip.** The tree re-shapes as the lane count changes, and the
 bits at the bottom move with it.
- **Drag a value cell up or down.** The drag is exponential, so one gesture can
 span decades — make one element huge and another tiny and watch the small ones
 get absorbed in one order but not the other. The `i:v` text field takes exact
 values if you would rather type them.
- **Switch presets.** "catastrophic cancellation" (±1024 plus crumbs) pushes the
 disagreement thousands of steps and up into the exponent; "all ones" shows a
 row with nothing to lose.
- **Hover anything.** Every node reports its partial sum, its exact bit pattern,
 and which elements it is made of.
- **Turn on the batch-invariant kernel.** The split pins to a fixed width and the
 answer stops moving with the batch. Note what it does *not* do: it does not
 become the sequential answer. It becomes **one** answer.

Deep links: `?view=`, `?batch=`, `?n=`, `?preset=`, `?invariant=1`, `?rival=`,
`?edit=3:900,7:0.0001`, `?step=N`, `?theme=`, `?compare=1`, `?ch=N`.

## The trade

Batch invariance is not free. A kernel that must reduce in the same order at
every batch size cannot pick the split that suits the launch it actually got, so
at small batches it leaves parallelism on the table. The page shows that part
*arithmetically* — the idle-lane fraction comes from the same lane counts the
reduction really used.

What the page **cannot** do is measure a real kernel's slowdown; nothing about a
browser canvas can. The source below reports it as a real but workable cost from
batch-invariant attention and matmul kernels, and that is a citation here, never
a number this page produced.

## Source

Thinking Machines, **"Defeating Nondeterminism in LLM Inference"** (2025-09) —
identifies batch-size-dependent reduction order as the cause of run-to-run
nondeterminism in servers that are otherwise fully deterministic, and proposes
batch-invariant kernels as the fix:
<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>

The lane-split rule on this page (a fixed pool of lanes shared across the rows in
flight) is the smallest honest model of shape-driven split selection. It is a
teaching model of the mechanism, not a transcription of any particular vendor's
reduction kernel.
