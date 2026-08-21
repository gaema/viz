# hybrid-cache-allocator -- two memory shapes, one pool

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: what happens to a serving runtime's memory allocator when a
model interleaves **attention** layers with **recurrent / linear-attention**
layers, so the pool has to hold two memory shapes that scale along different
axes -- and the failure class that creates.

**Anchor**: J-family serving-time memory management, one layer above
[`paged-attention`](../paged-attention/README.md) (which owns block-table paging
for a pure-attention model) and beside
[`hybrid-by-layer`](../hybrid-by-layer/README.md) (which owns *which* layers are
which). Neither mechanism is re-taught here; this page owns the **allocator that
has to serve both at once**.

## The mechanism

| | Attention layers | Recurrent layers |
|---|---|---|
| Per-sequence memory | KV cache: **grows one slot per token**, without bound | State matrix: **fixed size**, allocated once |
| Natural storage | pages handed out on demand | one slab per sequence |
| Scales with | **context length** | **concurrency** |
| Charged when | as the sequence lengthens | **up front, at admission** |
| Freed when | pages return as the sequence ends | slab returns as the sequence ends |

Both live under **one capacity line**. The page draws that line as a single bar
you can split, a page pool that fills as sequences lengthen, and a slab row that
does not fill at all -- it is either held or free, at exactly one size.

## The failure, which is the point

The two shapes fail in **opposite directions**, and the page lets you reach both:

| Regime | What you do | What you watch happen |
|---|---|---|
| **(a) sized for the slabs** | drag the split left -- few pages, many slabs | sequences are admitted, then **preempted mid-generation** when the page pool runs dry. The fixed side is nearly idle and cannot help. |
| **(b) sized for the pages** | drag the split right -- many pages, few slabs | sequences **cannot be admitted at all**: each one needs its whole fixed slab before it emits a single token, so they queue while most of the page pool sits free. |

So **maximum concurrency is set by the fixed part and maximum context by the
growing part**, and one knob sets both -- in opposite directions. Every page you
add costs a slab; every slab costs pages. The chart sweeps the split and shades
the band where **both** ceilings clear the demand; for a demanding workload that
band is **empty**, which is the honest answer that no split can give.

Two URLs reaching the two regimes, with the readout each shows:

- **(a) page starvation** --
 `index.html?split=20&seqs=4&ctx=4096&page=128&attnevery=4&step=8`
 → *concurrency 17 sequences, context 384 tok/seq at 4 sequences; 2 live, 2
 preempted mid-generation; idle 12.0 MiB of pages + 270 MiB of slabs.*
- **(b) admission starvation** --
 `index.html?split=88&seqs=12&ctx=512&page=128&attnevery=4&step=13`
 → *concurrency 2 sequences, context 512 tok/seq at 12 sequences; 2 live, 10
 refused for want of a slab while 93% of the page pool sits free.*

## Baseline

Every ratio on the page is stated against a **named** baseline: the same pool run
by a **pure-attention allocator**, which has no recurrent state and therefore
needs no slab at all, so all of its capacity is KV pages. At regime (a) above the
hybrid allocator's context ceiling is **18.8% of that pure-attention baseline**
(higher is better; 100% = parity); at regime (b) it is **80.0%**, but its
concurrency is hard-capped at 2 where the baseline's is not capped by fixed state
at all. Both figures are computed in-page from the same simulation that draws the
pools -- nothing is quoted from elsewhere.

## Interactions

| | |
|---|---|
| **Transport** | Auto-plays + loops: an admission phase (one step per sequence, admitted or refused with the reason) then generation rounds. Each step says what was allocated, completed or preempted. |
| **Direct manipulation** | Drag the **⇔ grip** on the capacity bar to move the split; drag the **sequences** and **target context** demand bars; drag **any one sequence's row** to change only that sequence's target. Everything recomputes under your hand. |
| **Hover** | Any KV page → owning sequence, its logical page index, the token range it covers, and the arithmetic behind its size. Any state slab → owning sequence, its size, that it does **not** depend on length, and how many tokens of KV the same bytes would have bought. Any sequence row → target, emitted, status and why. The capacity bar and the chart report the ceilings at the split under the cursor. |
| **Resize the problem** | Sequence count, target context, KV page size, and the attention:recurrent layer ratio (which moves both shapes' sizes at once). |
| **A/B compare** | Split 22% vs 92% -- regime (a) and regime (b) stacked. |
| **Challenges** | Starve the growing side; starve the fixed side; find a split inside the green band (and discover when there is none). |

## Render tier

T1 (Canvas2D). The lesson is allocation topology and two ceilings on one axis --
a few hundred cells, two curves and a shaded band. Nothing is gained by a higher
tier.

## Wiring

`layout.mount` + controls (`split`, `seqs`, `ctx`, `page`, `attnevery`, `lens`)
+ a step `Transport` whose `compute` runs the admission/generation simulation
and snapshots `pageOwner[]`, `slabOwner[]` and every sequence's status after each
event. Occupancy, admissions, refusals, preemptions, both ceilings, the feasible
band and the wasted bytes in each regime are all derived from those snapshots.
Drawn with `render.label` + `ctx`, coloured with `theme.js` tokens only
(`categorical` per sequence, `T.accent` for the growing side, `T.violet` for
the fixed side, `T.ok` for the feasible band, `T.bad` for stranded capacity), so
it reads in light and dark. Headless hooks mirror every drag:
`?split=&seqs=&ctx=&page=&attnevery=&lens=`, plus `?step=N`, `?hover=x,y`,
`?play=1`. Source: [`page.js`](page.js).

## Data

None fetched. The pool, both shapes and the whole run are simulated in-page from
a fixed 384 MiB pool and plain arithmetic (KV bytes per token per attention
layer; state bytes per recurrent layer), stated in the source. Same numbers on
every reload, no download, no weights, and no claim about any particular
product's allocator -- the shape of the problem is what public hybrid-serving
design discussions (vLLM / SGLang hybrid allocators, Jamba-
Qwen3-Next-class models) describe.
