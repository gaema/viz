# hyper-connections -- the residual highway, widened

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the residual connection, generalised from **one** stream to
**n** streams with learned connection weights. The classic residual is the
special case `n = 1` with every weight pinned to `1`, and the page carries that
case as live arithmetic so you can check the equivalence rather than take it on
trust.

Source (public paper): Zhu et al., **Hyper-Connections**,
[arXiv:2409.19606](https://arxiv.org/abs/2409.19606).

## Scope -- what this page does NOT re-teach

- [residual-stream](../residual-stream/README.md) owns the **single highway**:
 read, add, write back, never overwrite; pre-norm vs post-norm placement; the
 magnitude growing with depth.
- [transformer-block](../transformer-block/README.md) owns what happens **inside**
 a block.

This page holds both of those fixed and varies only the **wiring between
blocks** -- the axis hyper-connections actually move along. The block here is a
deliberately plain stand-in (pre-norm read → SiLU → one dense projection),
identical on both paths, so every difference on screen comes from the
connections and from nothing else.

## The mechanism

Carry `n` vectors `s₀ … s₍ₙ₋₁₎` instead of one. Each block `b` does three things
with weights the network learns:

 read x = Σ_m A[b][m] · s_m (depth connection, in)
 transform y = block(x)
 write s'_k = B[b][k] · y + Σ_m M[b][k][m] · s_m (depth connection, out
 + width connection)

`A` chooses **which streams the block reads**, `B` chooses **which streams it
writes into**, and `M` -- the width connection -- decides **how the streams mix
with each other** across the block. Its diagonal is the familiar identity skip;
its off-diagonal is content moving between streams, which a single highway has
no way to express. The expansion rate `n` and this depth/width split are the
parameters the page exposes.

`n = 1, A = B = M = [1]` collapses all three lines back to `s' = y + s` -- the
plain residual.

## The tension it resolves

Pre-norm residuals train stably but let later blocks contribute little: the
stream magnitude grows with depth while each block's write stays the same size,
so its share of the answer shrinks -- representation collapse. Post-norm gives
each block a strong voice but trains badly. The architect normally picks one of
these two for the whole model.

Hyper-connections let the network sit **wherever it wants between them, per
block**, because the same knobs express both:

| Preset | Wiring | What you see |
|---|---|---|
| **classic residual (n=1)** | one stream, all weights 1 | `max\|Δ\|` against the plain reference reads **0** |
| **widened, identity weights** | `n` streams, `A = 1/n`, `B = 1`, `M = I` | the streams stay exact copies -- widening alone changes nothing |
| **learned (streams diverge)** | per-block read/write targets + off-diagonal mixing | streams take different magnitudes; blocks specialise |
| **strong-write (post-norm side)** | identity path damped (`M = 0.72·I`) | late blocks keep a large share; the carried signal fades |
| **collapsed (late blocks silent)** | write weights decay with depth | contribution bars shrink to nothing; effective depth falls toward 1 |

Two numbers report this, both defined on screen rather than assumed:

- **contribution** `c_b = ‖what block b wrote‖ / ‖readout after block b‖` -- the
 share of the answer's magnitude that block just added.
- **effective depth** `(Σ c_b)² / Σ c_b²` -- the participation ratio of those
 contributions. It equals `L` when every block contributes equally and falls
 toward `1` when one block dominates and the rest are silent.

## The n=1 equivalence check

The page runs **two** stacks every frame over the same embedding with the same
block matrices: the `n`-stream hyper-connected one, and a plain single-stream
residual `r ← r + block(r)`. The right-hand panel prints both readout norms
`max|Δ|` across all `D` features. At `n = 1` with unit weights the two paths
reduce to the same arithmetic -- `1·y + 1·s` versus `s + y` -- and the reported
difference is exactly `0`, not merely small. Move any weight and it stops being
zero; that is the check working.

## Interactions

- **Transport** -- step block by block; autoplays and loops. `?step=N`.
- **Direct manipulation** -- drag any connection handle ↕: read handles between
 a lane and the block, write handles between the block and a lane, and the
 width-connection handles between lanes (shown for the stepped block). The
 weight moves, that block's contribution bar moves, and every level below it
 re-runs under your hand.
- **Widen the highway** -- drag the stream-count slider `n`; also `L`, `D`, seed.
- **Preset switch** -- the five wirings in the table above, plus `custom` once
 you have dragged something.
- **Hover** -- any handle gives its weight *and what that weight does*; any
 stream node gives its magnitude and where it came from; any bar gives the two
 norms its ratio is made of.
- **Challenge mode** -- three goals, starting with "reduce the widened highway
 to the classic residual and get `max|Δ|` to exactly 0".

## Render tier

T1 (Canvas2D). The picture is a small graph of lanes, edges and handles rather
than a large tensor, so nothing here needs a GPU tier.

## Wiring

`layout.mount` + controls (`preset`, `n`, `L`, `D`, `seed`) + an `L`-step
transport + `onPointer` weight drag + hover tooltips + `challenges`. URL hooks:
`?preset=` `?n=` `?L=` `?D=` `?seed=` `?step=N` `?play=1` `?hover=x,y`
`?w=KIND,b,i[,j],value` -- the headless stand-in for a drag
(`?w=B,3,0,0` zeroes block 3's write into stream 0; `?w=M,1,0,2,0.3` sets the
width connection stream 2 → stream 0 at block 1). Repeat `?w=` for several
edits. Source: [`page.js`](page.js).
