# sparse-attention-select -- choosing what to attend to

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: how a long-context model decides **which** past tokens are
worth reading, instead of reading all of them. **Anchor**: A3 attention pattern
(Family B).

## The mechanism

A cheap side network — the **indexer** — scores every past token for the current
query. The top-k scoring tokens are kept, and real attention runs only on those.
Everything else is never read.

The page draws four attention types the reader switches between:

| Mode | What happens | Selection? |
|---|---|---|
| `full attention` | dense causal attention — every query reads every past KV entry | none (the baseline) |
| `indexer + top-k` | the indexer scores every past token; the k best go to real attention | learned, per query |
| `compress, then select` | KV is first compressed **along the sequence dimension**, then the same top-k selection runs over the compressed entries | learned, over compressed entries |
| `compress harder, then dense` | much heavier compression, then **dense** attention over the short compressed sequence | none |

The last two are the pair a recent DeepSeek model card describes as two attention
types mixed inside one model: *"combines Compressed Sparse Attention (CSA)
Heavily Compressed Attention (HCA)... CSA compresses KV caches along the
sequence dimension and applies DeepSeek Sparse Attention (DSA); HCA applies
heavier compression with dense attention."*

## How this differs from `attention-patterns`

[`attention-patterns`](../attention-patterns/README.md) shows **fixed** masks —
causal, sliding-window, hybrid, sink. Which key a query may read is decided in
advance by its *position*, and the same picture holds for every input.

This page shows a **learned, query-dependent** selection. The attended set is
computed at run time from a score, so it is different for every query and every
input, and it can reach far back in the sequence when the content warrants it.
The two pages are complements: one is the geometry of masking, this one is the
policy that picks a subset.

A sibling technique with a different structure is **NSA**
(<https://arxiv.org/abs/2502.11089>), which splits attention into three parallel
branches — compressed, selected, and a sliding window — rather than having one
indexer choose a single subset.

## The two trades, both draggable

1. **The indexer is itself quadratic.** It still scores *every* past token, so
 its cost grows with the square of the sequence just like attention does. It
 only pays off once the attention term it replaces is big enough — the page
 computes the **crossover sequence length N\*** from the reader's own settings
 and says whether the current N is above or below it. Widen the indexer
 raise `k` and the crossover marches out past the slider range — at
 `N=12, k=24, d=32, indexer dim=32` the page reports `N* = 82`, `N=12 is BELOW
 it`, and a cost of **150% of full attention (lower is better; 100% =
 parity)**: the "sparse" path is the more expensive one there.
2. **Top-k is a hard recall cliff.** A needle outside the k contributes exactly
 nothing — not a little, nothing. Hide a needle at position `p`, then shrink
 `k` (or move the needle away from the query, or make the indexer cheaper
 noisier) and watch its rank cross the cut. When it does, the panel names the
 share of the true attention mass that leaves with it.

## Every number on the page is computed live

No figure here is quoted from anywhere. Multiply counts for one causal prefill
of `N` tokens, evaluated on the reader's own `k`, `d`, indexer dim
compression factor, `2·d` per attended entry (the QK score plus the PV
accumulation):

- full — `2·d·Σ(i+1)`
- indexer + top-k — `d_idx·Σ(i+1)` for scoring, plus `2·d·Σ min(k, i+1)`
- compress, then select — a linear `N·d` compression term, plus scoring
 attention over `⌈(i+1)/c⌉` compressed entries
- compress harder, then dense — the same linear compression term,
 `2·d·Σ⌈(i+1)/c_heavy⌉`

The scores are real too: query and key vectors are deterministic seeded tensors,
the true logit is their scaled dot product, and the indexer score is that same
dot product taken through a **low-rank random projection** into `d_idx` dims —
so a cheaper indexer really is a noisier one, and the page's "recovered `m` of
the true top-k" counter falls as you drag the indexer dim down. That projection
is *untrained*, so the recall it reports is a worst case; a trained indexer
agrees with full attention far more often at the same cost. The mechanism
its failure mode are what is being shown, not a model's accuracy.

## Render tier

T1 (Canvas2D). The picture is a token strip, a ranked-score bar and an `N×N`
mask — none of it needs a GPU tier to teach the concept.

## Wiring

`layout.mount` + controls (mode, `N`, `k`, compression factors, head dim,
indexer dim, needle position + relevance, seed) + a per-query `Transport`.
Per the interactivity contract:

- **transport** steps the query position along the sequence, auto-playing
 looping;
- **direct manipulation** — drag the top-k cut across the ranked-score bar, drag
 the first compressed block's edge to change the compression factor, drag the
 needle along the sequence strip; every counter recomputes under the cursor;
- **hover-to-inspect** — any token in the strip, any bar in the ranked chart, or
 any cell of the attention matrix gives its indexer score, its rank,
 whether it made the cut;
- **challenge mode** — drop the needle, find the crossover, make the indexer lie.

Headless hooks: `?step=N` (query position), `?mode=`, `?N=`, `?k=`, `?c=`,
`?hc=`, `?d=`, `?di=`, `?p=`, `?nb=`, `?seed=`, `?hover=x,y`, `?play=1`.
Source: [`page.js`](page.js).

## Sources

- DeepSeek-V3.2 — <https://arxiv.org/pdf/2512.02556>
- vLLM's day-0 write-up of it — <https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html>
- NSA, the three-branch sibling — <https://arxiv.org/abs/2502.11089>

Only the **mechanism** is taken from these. Compression factors, mix ratios,
FLOP shares and KV-cache percentages circulate in secondary write-ups and are
deliberately absent from this page; every quantity it shows is derived from the
reader's own settings instead.
