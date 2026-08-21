# context-editing -- compacting an agent transcript, and what the edit costs

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: a long-running agent's context fills with tool results that
went stale several turns ago. **Context editing** clears them once occupancy
crosses a threshold, keeping a tail of recent results and (optionally) a running
summary of what was folded away. The page simulates the conversation, the
policy, and the cost -- including the cost almost nobody budgets for.
**Anchor**: A3/A4 attention + KV cache (Family B/J), one layer above
[`../kv-eviction/`](../kv-eviction/README.md).

## The same idea as KV eviction, one layer up

[`kv-eviction`](../kv-eviction/README.md) owns token-level eviction under a
**fixed cache**: score individual tokens, drop the weakest, keep the cache a
constant size. This page is the same grammar at the granularity of **messages**.

| | `kv-eviction` | `context-editing` (this page) |
|---|---|---|
| Unit scored | one token's KV entry | a whole **turn** or **tool result** |
| What is edited | the **cache** | the **visible transcript** |
| Trigger | the cache is full | occupancy crosses a **threshold** |
| Score | accumulated attention mass | age, kind, and a protected recent tail |
| Cost of getting it wrong | attention mass the survivors no longer carry | **prefix-cache invalidation**, paid in prefill |

**The two interact, and not always in the same direction.** Token-level eviction
shrinks the cache while the transcript stays byte-identical, so the shared prefix
survives and prefix caching keeps working. Message-level editing shrinks the
transcript, which is the thing the prefix cache is keyed on -- so it frees far
more room per edit and invalidates the cache to do it. Run both and the cache
budget is set by eviction while the *re-prefill bill* is set by how often the
transcript is edited.

## The subtle cost: editing history invalidates the prefix

Everything before the edit point is a shared **prefix** that the serving stack
was re-reading cheaply (see [`../radix-attention/`](../radix-attention/README.md)
for how that prefix is stored and shared). A prefix cache is **positional**: it
matches a token run from the start, so changing one message invalidates every
token after it, whether or not those later tokens changed.

So a compaction that saves 60k tokens of context can force a large prefill
recompute on the very next turn. On the page that is the **red bar** -- the red
height *is* the number of tokens re-prefilled at full price. Drag the `▾`
threshold left and every turn becomes an edit turn, and the chart turns solid
red: a too-eager threshold makes every turn expensive.

## The counter-point: sometimes keeping everything is cheaper

This is the disagreement the page is built to let you find rather than assert.
With prompt caching, **a stable prefix is nearly free to re-read while an edit is
never free.** Push the `cached prefix re-read cost` slider low enough and the
never-compacting baseline wins outright.

Two configurations, both reachable by URL, both computed by the page:

| Config | Cumulative prefill cost of `clear + summarise` | Verdict |
|---|---|---|
| `?policy=summary&cache=0&thresh=30&win=200&turns=60&step=59` | **111.8% of never-compacting** (lower is better; 100% = parity) | 🔴 compaction **loses** by 11.8%, and the baseline never overflows, so it is a real alternative |
| `?policy=summary&cache=0.02&thresh=30&win=200&turns=60&step=59` | **81.6% of never-compacting** | 🟢 compaction **wins** by 18.4% -- the same conversation, one slider moved |

(Append either query to the demo's own URL. Both are readings this page computed
and printed in its readout; nothing here is estimated.)

The whole verdict turns on a knob nobody looks at. That is the lesson.

**And the baseline has a wall.** At the default 64k window the never-compacting
arm overflows partway through, and the page says `OVERFLOWED` when it does. An
overflowed baseline is "cheaper" only in the sense that a run which cannot happen
is cheap -- so when compaction loses against an overflowed baseline the readout
says so and tells you to widen the window until the comparison is honest.

## Everything numeric is computed

There is no model behind the page and no illustrative number in it. A
deterministic conversation is generated from `(turns, seed)` alone -- user turns,
tool calls, tool results with a heavy size tail, assistant text -- and **all
three policies see exactly the same conversation**, so the comparison is a
controlled one. Then, per turn:

- **occupancy** = the summed token cost of every block still in the transcript
 (a cleared result costs its 12-token marker; a folded turn costs nothing; the
 summary costs ~6% of what it replaced, floored and capped);
- **tokens cleared** = what each edit actually freed;
- **tokens re-prefilled** = everything from the edit point to the end of the
 transcript, because the prefix cache is positional;
- **turn cost** = `cache_rate × cached_prefix_tokens + 1 × full_price_tokens`,
 in **prefill token-equivalents (TE)**, where 1 TE is the compute of prefilling
 one uncached token;
- **cumulative cost**, reported as a **percent of the named `never compact`
 baseline** -- a *lower-is-better* axis, so `>100%` means compaction is
 **worse**, and the readout spells the direction out in words.

The cached re-read rate is a **slider, not a claim**. Real caches differ by stack
and deployment, and the interesting question is how the verdict moves across the
range, not what any one product charges.

## Interactions

- **Transport** — one step per turn as the conversation grows, autoplay + loop,
 so the context visibly fills, crosses the threshold and gets edited.
- **Direct manipulation** — drag the `▾` **clear threshold** along the tape
 drag the `◂▸` **kept-tail bracket** to change how many recent tool results are
 protected. Occupancy, what gets cleared, and the resulting prefix-invalidation
 cost all move together under your hand (the whole simulation re-runs; nothing
 is interpolated).
- **Click to pin** — click any block on the tape to pin/unpin it. A pinned block
 is never cleared and never folded, so you can protect something and watch the
 consequence: the edit point moves right, less is freed, and the following turns
 cost more.
- **Hover-to-inspect** — any block reports its kind, its **token count**, its
 **age in turns**, its share of the current context, and **whether it survives
 and why**. Hover the cost chart for a turn's cached/full-price split and what
 the edit there freed.
- **Resize the problem** — context window, conversation length, seed.
- **A/B compare** — `never compact` beside `clear + summarise` on the same
 conversation.
- **Challenge mode** — make compaction lose; then make it win by ≥20%; then pin
 a block and keep the policy under the window.

Headless hooks (`--screenshot` has no pointer, so every handle has a URL twin):
`?step=N`, `?play=1`, `?policy=never|clear|summary`, `?thresh=N`, `?keep=N`,
`?cache=F`, `?win=N`, `?turns=N`, `?seed=N`, `?pins=3,7`, `?hover=x,y`,
`?theme=light|dark`. Source: [`page.js`](page.js).

## Render tier

T1 (Canvas2D). The lesson is a timeline, a tape and two bar charts -- structure
and arithmetic, not pixel throughput -- so nothing here needs a GPU tier.

## Wiring

`layout.mount` + controls (`policy`, `thresh`, `keep`, `cache`, `win`, `turns`,
`seed`, `pins`) + a per-turn `Transport` whose `compute` runs all three arms
over one generated conversation and snapshots the transcript state per turn, so
scrubbing back and forth replays exact states. Pins live in one comma-separated
`pins` state key, which the framework's deep-link sync mirrors into the URL for
free -- which is also the headless stand-in for a click.
