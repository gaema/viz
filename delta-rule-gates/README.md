# delta-rule-gates -- how the gate in a delta rule became two gates

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **one** recurrent matrix state, **three** gate designs. Family E
(linear attention); the companion to
[gated-deltanet](../gated-deltanet/README.md), which teaches the delta-rule
recurrence itself. This page holds the recurrence fixed and varies only the
**gate**, because that is the axis the family actually moved along.

## The arc it teaches

All three designs write the same object -- a fixed `[d×d]` matrix state `S`, a
key→value associative memory -- with the same delta rule. They differ only in
what multiplies the two terms:

 Sₜ = diag(bₜ) · Sₜ₋₁ · (I − kₜkₜᵀ) + diag(wₜ) · vₜkₜᵀ
 └── erase ──┘ └── write ──┘

| Design | Gate | What it buys |
|---|---|---|
| **Gated DeltaNet (GDN)** | `bₜ = wₜ = αₜ·1` in this page's `b`/`w` form | Forgetting at all: the state decays instead of saturating. But the *whole* matrix decays at one rate, so no feature can be kept while another is dropped. |
| **Kimi Delta Attention (KDA)** | `bₜ = wₜ = γₜ` -- a **diagonal**, one rate per channel | Each feature dimension of the state forgets at its own rate: fine-grained control over what decays fast and what is held. Ships in **Kimi Linear**, hybridised 3:1 with full attention. |
| **Gated DeltaNet-2 (GDN-2)** | `bₜ` and `wₜ` **independent**, both channel-wise | GDN and KDA both use a *single* gate to decide erase **and** write at once. Splitting them lets the state keep old content (`b` near 1) while still admitting new content (`w` free), or the reverse. |

Progression, in one line: **scalar → per-channel → decoupled**, where *per-channel*
means **the decay**.

### The gate column is this page's stand-in, not any paper's notation

Each design carries a decay **and** a delta strength; the two-gate `b`/`w` form
folds them together so all three sit on one widget. Check the page against the
real rules, not against the table:

| | Rule | Decay | Delta strength |
|---|---|---|---|
| GDN | `Sₜ = αₜ·Sₜ₋₁·(I − βₜkkᵀ) + βₜvkᵀ` | scalar `αₜ` | scalar `βₜ` |
| KDA | `Sₜ = (I − βₜkkᵀ)·Diag(αₜ)·Sₜ₋₁ + βₜkvᵀ` | **channel-wise** `αₜ` | still **scalar** `βₜ` |
| GDN-2 | `Sₜ = (I − k(bₜ⊙k)ᵀ)·Diag(αₜ)·Sₜ₋₁ + k(wₜ⊙v)ᵀ` | channel-wise `αₜ` | **separate** `bₜ` (key axis) and `wₜ` (value axis) |

Two things this table exists to stop you concluding. KDA did **not** make the
erase/write gate channel-wise — its `βₜ` is a scalar per head, and GDN-2's own
abstract says KDA "still uses a single scalar gate to control two different
things". And the reduction runs the way the paper states it: GDN-2 recovers KDA
when `bₜ = wₜ = βₜ·1`, i.e. both gates collapse to the same **scalar** while the
channel-wise decay is retained, and recovers GDN by then also setting
`αₜ = αₜ·1`.

Sources (public papers): Gated DeltaNet, arXiv:2412.06464 · Kimi Linear / KDA,
arXiv:2510.26692 · Gated DeltaNet-2, arXiv:2605.22791.

## What it shows

- The **state matrix** `S [d×d]` evolving token by token, with the update split
 into its two visible halves: the **kept** term (`b`·erase) and the **written**
 term (`w`·vkᵀ).
- The **gate as the thing that differs** -- drawn as handles row-aligned with `S`:
 one handle that moves every row together (GDN, which is exactly the limitation),
 one handle per row (KDA), or two independent columns of handles, erase
 write (GDN-2).
- A **memory written at step 0** (its key becomes the probe key `k★`), tracked
 separately as it is decayed and erased inside `S` -- so "how long does this
 survive?" is a number on screen, per channel and overall. The recurrence is
 linear in the state, so that one write's own contribution can be carried
 forward exactly, with the other tokens' memories (different associations, not
 survival of this one) kept out of the number. The delta rule erases it further
 wherever a later key overlaps `k★` -- that part is shared by all three designs,
 which is why none of them reaches 100%.
- A **closing panel**: the same sequence scored under all three gate designs at
 the final step, with per-channel bars. Under GDN every channel's bar is the
 same height by construction; under KDA and GDN-2 they are not.

## Interactions

- **Transport** -- step the recurrence token by token; autoplays and loops.
- **Direct manipulation** -- drag the gate handles themselves (↔) and watch the
 retention curve move under your hand; drag a `vₜ` component (↕) to change what
 gets written.
- **Hover** -- any state cell gives its value *and the update that produced it*
 (the decay applied, the delta erased, the amount written), not just the number;
 hovering a gate handle names which half of the update it controls.
- **Mode control** -- GDN / KDA / GDN-2, plus `L`, `d`, seed and the scalar `α`.
- **Challenge mode** -- two goals, the first of which is unreachable under the
 scalar gate on purpose.

## Render tier

T2 (WebGL2 heatmaps for the `[d×d]` state and the two update terms; Canvas2D for
the gate handles, the retention chart and the three-way comparison), with the
usual Canvas2D fallback.

## Wiring

`layout.mount` + controls (`mode`, `alpha`, `L`, `d`, `seed`) + an `L`-step
transport + `onPointer` gate/value drag + hover tooltips + `challenges`. URL
hooks for every handle: `?mode=` `?alpha=` `?g=` `?b=` `?w=` `?v=i,val` `?step=N`
`?L=` `?d=` `?seed=` `?play=1` `?hover=x,y`. Source: [`page.js`](page.js).
