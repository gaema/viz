# chunked-prefill -- splitting a prefill so it stops stalling everyone else

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **scheduling** consequence of the two inference regimes.
A serving engine runs one batched step at a time, so a long prompt's prefill --
one big compute step -- blocks every other sequence's next token while it runs.
**Chunked prefill** splits that prefill into pieces of at most a per-step token budget — a few hundred to a few thousand tokens in a
real engine (vLLM's `max_num_batched_tokens`, tuned around 2048 for latency
8192+ for throughput); the toy budget on this page is far smaller so the packing
is visible
packs each piece into a step alongside the waiting sequences' decode tokens, up
to a per-step **token budget**. **Anchor**: A4 KV cache / runtime shape (Family
J, serving time). Companion to [prefill-vs-decode](../prefill-vs-decode/README.md), which
shows *why* the two regimes have different shapes; this page shows what a
scheduler does about it.

## What it shows

- **The stall.** With one big prefill, the whole prompt is a single step and no
 decode slot exists inside it. The waiting sequences' token strips show one
 wide hole -- their inter-token latency spikes to the length of the entire
 prefill.
- **Chunking.** With a token budget, each scheduler step carries a **mix**: one
 prefill chunk of sequence A plus one decode token for each waiting sequence.
 The timeline bars stack exactly that way -- decodes at the bottom, the prefill
 chunk on top, the dashed budget line as the cap.
- **The trade, stated honestly.** Two live numbers, both lower-is-better
 both quoted as a percent of the one-big-prefill baseline: the long prompt's
 **time to first token** gets *worse* (its prefill is spread over more steps,
 each paying the fixed per-step overhead again) while the other sequences'
 **worst gap between tokens** gets much *better*. Neither is hidden.
- **The budget is the knob.** A budget sweep alongside the numbers plots both
 halves against the budget: shrink it and the gap curve falls while the
 time-to-first-token curve climbs, so there is a middle rather than a best.
 Push the budget below the number of waiting sequences and the page says so --
 the scheduler still forces one prefill token through, and the step overruns.

The millisecond figures come from a deliberately simple illustrative cost model
-- a fixed per-step overhead plus a per-token cost -- chosen so the *shape* of
the trade is readable. They illustrate the mechanism; they are not a measurement
of any particular engine or accelerator.

Press play (or scrub) to walk the scheduler steps; **drag** the dashed budget
line up and down to re-pack the timeline and watch the two numbers move in
opposite directions; **hover** a step for exactly what it carried, or a gap in a
token strip for what occupied the scheduler while that sequence waited.

## Render tier

T1 (Canvas2D throughout: stacked step bars, arrival strips, budget sweep).

## Wiring

`layout.mount` + controls (prompt length, waiting sequences, token budget,
scheduler mode) + a Transport over the scheduler steps (autoplay + loop)
`onPointer` budget-line drag + hover-to-inspect + two challenges + an A/B
compare across the two scheduler modes. URL hooks: every control is mirrored
into the query string (`?P=` `?S=` `?budget=` `?mode=`), plus `?step=N`,
`?hover=x,y` and `?play=1`. Source: [`page.js`](page.js).
