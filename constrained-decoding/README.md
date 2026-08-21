# constrained-decoding -- a grammar as a bitmask over the vocabulary

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a JSON schema or grammar forces valid output, at the logit
level -- the compiled automaton, the per-step vocabulary bitmask, the two-class
mask cache that makes it affordable, and jump-forward decoding with the
retokenization seam it leaves behind. **Anchor**: the decode-time companion to
[sampling](../sampling/README.md) (which draws from the distribution this page
truncates) and [tokenization](../tokenization/README.md) (whose token boundaries
are what the seam is about).

## What it shows

A schema does not persuade a model to emit valid JSON. It **deletes every other
option**:

1. **Compile.** The schema becomes a **pushdown automaton** -- a finite-state
 skeleton plus a stack, because a grammar with nesting is not regular. The
 page compiles the schema you type: a key chain per field (`" n a m e ":`),
 shared `int` / `string` sub-rules entered with a **return address pushed**,
 and `[` / `{` pushing bracket frames that the closing tokens pop.
2. **Mask.** At each decode step the decoder needs one bit per token over the
 **entire** vocabulary: does this token keep the string inside the grammar?
 Everything else gets `−∞` added before the softmax, so an invalid token has
 probability exactly zero.
3. **Sample.** From what survives.

Step 2 is where the engineering is. Done naively it is a rescan of a 128k+ entry
vocabulary *per token*, which costs more than the model step it guards. The
production trick ([XGrammar](https://arxiv.org/pdf/2411.15100), MLSys 2025) is to
split the vocabulary:

| Class | What decides it | Cost per step |
|---|---|---|
| **context-independent** (the large majority) | the automaton **node** alone -- the walk never touches the stack | precomputed once per node into a **mask cache**, then reused |
| **context-dependent** (the small remainder) | the **stack** -- these are the tokens that CLOSE something (`",`, `"}`, `]`), or that end a number, where "which terminator is legal" is a question only the stack can answer | re-checked live, every step |

The page implements exactly that split. Each vocabulary cell is coloured by
legality (green / red) and **shaded** by class, with a teal outline on the
context-dependent ones; the readout's cache counters -- node masks built, token
scans, node hits reused, share of checks that were context-dependent -- are
produced by the code, not narrated. The share is much higher here than in a real
deployment (a 200-token illustrative vocabulary has proportionally far more
structural tokens than a 128k one, where the context-dependent share is ~1%),
the readout says so on screen.

**Jump-forward decoding** ([SGLang's compressed
FSM](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)) is the second half
of the trick. Where the automaton has a run of single-transition edges -- after
`{` the schema forces `"name": ` -- no model call could change the answer, so the
decoder emits the characters itself. The default schema hands 28 of its 40
characters over for free -- 8 model calls instead of 16.

And it has a real cost, which is the teaching moment: **the mask is built over
CHARACTERS and the decoder works over TOKENS**, and at the end of a jump those
disagree. A tokenizer would rather emit `":` as ONE token than `"` then `:`; a
jump has already emitted them as raw characters, so the model resumes on a
boundary its training rarely saw. The top ribbon draws both segmentations -- what
was emitted, and how a tokenizer would split the same string -- and marks every
disagreement with a red **▲**.

## The trades

| Trade | Where you see it |
|---|---|
| The distribution is **truncated, not corrected** | drag *model agrees with schema* to 0: kept probability mass falls under 1%, the top-20 bars all go `−∞`, and the token the grammar forces was ranked ~130/200 in the model's own preference. It parses. It is gibberish. |
| Mask compute is **per-request state** | the cache is per (schema, node) and the stack is per sequence, so this work does not batch across a request group the way a GEMM does -- every sequence in a batch owns its own automaton position |
| **Compilation** is paid per unique schema | the readout reports the node count compiled once per schema. Cached, it is free; a workload where every request carries a *different* schema pays it every time |

The often-quoted "constrained decoding costs 5-15%" figure predates the mask
cache. With a warm cache on a repeated schema the per-step overhead is near zero,
and jump-forward makes some schemas *faster* than free-running decode by skipping
model calls outright.

## Interactions

| # | Contract item | Here |
|---|---|---|
| 1 | scrub / step | transport steps the decode, autoplays and loops; `?step=N` |
| 2 | hover-to-inspect | any vocabulary cell → legal or masked, the node that decided it, **which transitions that node accepts**, and whether the verdict was cached or re-checked against the stack |
| 3 | direct manipulation | **edit the schema** and watch the legal set collapse under your hand (`name:colour` does not compile → dead state, mask all red); plus a canvas handle for *model agrees with schema* |
| 4 | live animation | the current automaton node pulses on the ambient clock while the transport plays |
| 5 | resize the problem | the schema itself is the size knob -- fields, types, and `int[]` (the case that makes the stack non-trivial) |

**A/B compare** renders mask ON against mask OFF: with the mask, every token is
grammar-legal by construction; without it, the raw argmax walks out of the
grammar and the page shows it drifting. **Challenge mode** asks for a dead state,
and for a step where under 1% of the model's probability mass survives.

## Honest about the numbers

There is **no model on this page**. The logits are synthetic: seeded noise
three pulls -- toward the document the model "meant" to write, toward ordinary
prose (which JSON mostly forbids), and, once it has left its own target, toward
closing what it opened so a run terminates. The *fit* slider weighs intent
against prose. The vocabulary is an illustrative 200 tokens, not a real
tokenizer's 128k.

Everything else is really computed in the page: the automaton and its stack, the
mask, the two-class cache accounting, the jump-forward runs, the greedy
retokenization and its seams. Seeded, so one URL replays one exact run.

## Render tier

T1 (Canvas2D: chip ribbons, an automaton strip, a 200-cell mask grid, and the
before/after probability bars; no heatmap needed).

## Wiring

`layout.mount` + controls (`schema` text, `jump`, `mask`, `fit`, `seed`)
`transport` over the decode steps (autoplay, loop) + `animate` (the node pulse
via `api.t`) + `onPointer` handle drag (routed through `controls.set`) + hover.
URL hooks: `?step`, `?schema`, `?jump`, `?mask`, `?fit`, `?seed`, `?play=1`,
`?hover=x,y`, `?compare=1`, `?ch=N`. Source: [`page.js`](page.js).
