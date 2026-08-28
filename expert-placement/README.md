# expert-placement -- which GPU an expert lives on

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: under **expert parallelism** each device holds a *subset* of a
Mixture-of-Experts layer's experts, so **where an expert lives** decides how much
of the batch has to cross the interconnect. **Anchor**: A6 MoE variant; Family D,
one level in from [parallelism](../parallelism/README.md) (which draws *what*
crosses the wire under each parallelism strategy) and a different question from
[moe-routing](../moe-routing/README.md). See
MoE.

## The distinction this page owns

| Question | Page |
|---|---|
| Which experts does this token want? | [moe-routing](../moe-routing/README.md) |
| Does the router spread the load evenly across experts? | [moe-balance](../moe-balance/README.md) |
| **Where do those experts physically live, and what does that cost?** | **this page** |
| What crosses the wire under tensor / pipeline / expert / data parallelism? | [parallelism](../parallelism/README.md) |

The routing page already answers *which* experts a token picks. This page holds
that answer completely fixed and moves only the **placement** — the same tokens
want the same experts throughout, and every number on screen changes anyway.
Routing is not re-taught here.

## What it shows

**The mechanism.** Each device holds a subset of the experts. A token whose
chosen expert lives elsewhere has to be shipped there and its result shipped
back — the all-to-all. The dispatch copies a token **once per DISTINCT
destination device**, not once per expert, which is what makes placement a
lever:

- **co-locating experts that are frequently chosen TOGETHER** collapses a
 token's destination set, so its work stays local;
- **scattering them** maximises the number of devices each token must visit;
- expert popularity is **heavily skewed** in practice, so a hot expert sitting
 alone on a device makes that device the bottleneck while the others idle —
 the step waits for the slowest device either way.

**Two shipped responses**, both drawn:

- **Replicating hot experts** onto every device, so they always execute at home
 and never travel — paid for in a whole extra copy of their weights per device.
- **Rebalancing the placement** periodically from *observed* load
 co-occurrence, rather than from the model definition.

Both appear in DeepSeek-V3's deployment description
([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)), which keeps redundant
copies of high-load experts and periodically recomputes the expert placement
from measured load. The broader expert-parallel load-balancing literature —
[FasterMoE](https://doi.org/10.1145/3503221.3508418) (PPoPP '22 — no arXiv
preprint; 2203.10924 is an unrelated condensed-matter paper),
[SmartMoE](https://www.usenix.org/conference/atc23/presentation/zhai) (USENIX
ATC '23 — likewise no arXiv; 2304.11414 is Pipeline MoE, a different system),
[Tutel](https://arxiv.org/abs/2206.03382) — treats placement, not routing, as
the tunable. Nothing here describes any particular deployment's configuration:
the interconnect and compute rates are **reader-set sliders** whose defaults are
public nominal figures.

**Everything on screen is counted off the simulated batch** — token copies
crossing per layer, bytes on the busiest link, per-device (token, expert) pairs
and utilisation, idle time per device, and the step time set by whichever of
(slowest device, busiest link) is larger. The batch is **seeded**, so a reload
and a shared link show the same picture.

**Reported against a named baseline.** The step time is quoted as a percent of
**round-robin placement**'s step time, with the direction stated — *lower is
better; 100% = parity*. Measured on the page's own defaults (16 experts, 4
devices, top-2, 256 tokens, 32 MoE layers, skew 1.20, seed 7, 100 GB/s links,
200 TFLOP/s devices):

| 🟢🟡🔴 | Placement | Step | vs round-robin | Busiest device | Crossings/token |
|---|---|---|---|---|---|
| ⚪ | round-robin (baseline) | 1.39 ms | 100.0% | 1.35× fair share | 1.44 |
| 🟢 | rebalance from observed load | 1.20 ms | **86.1%** (faster) | 1.16× | 1.30 |
| 🟢 | rebalance + 3 replicated hot experts | 1.08 ms | **77.5%** (faster) | 1.05× | 0.73 |
| 🔴 | worst case | 2.51 ms | **180.3%** (slower) | 2.44× | 1.20 |

The worst case is instructive precisely because it moves *fewer* tokens than the
baseline (1.20 crossings per token against 1.44) and is still the slowest thing
on the page: piling the hot experts together does co-locate them, and then one
device does 2.44× its share of the work while the other three sit idle. Traffic
is not the only cost of a placement.

## Interactions

- **Transport** steps one MoE layer: routing → dispatch → compute → combine.
 Autoplays and loops.
- **Direct manipulation — drag an expert between devices.** Pick a chip up
 drop it on another device; the wires, the per-device bars and the step time
 recompute immediately.
- **Direct manipulation — drag the popularity ribbon.** Vertically changes the
 demand **skew**; horizontally changes how many hot experts are **replicated**.
- **Hover a device** for its resident experts, its share of the work, its
 utilisation and its idle time; **hover a wire** for how many tokens cross it,
 why, and which experts pulled them there; **hover the ribbon or a chip** for
 one expert's demand and placement.
- **Buttons**: round-robin (the named baseline), rebalance from observed load,
 and worst case.
- **Challenge mode**: beat the baseline by 20%, manufacture a bottleneck,
 get more than half the batch to stay home.

## Render tier

T1 (Canvas2D: device panels, expert chips, all-to-all curves with moving
packets, utilisation bars).

## Wiring

`layout.mount` + controls (`devices`, `experts`, `topk`, `tokens`, `skew`,
`rep`, `layers`, `dmodel`, `bw`, `tflops`, `seed`, three placement buttons)
`animate` (packets flow along the wires via `api.t`) + `onPointer` chip
drag-and-drop and ribbon two-axis drag + hover + `challenges`. Deep links: every
control is mirrored into the query string, plus `?step=N`, `?place=<one device
digit per expert>`, `?preset=rr|rebalance|worst` and `?hover=x,y` (the
screenshot-verifiable stand-in for a real cursor). Source: [`page.js`](page.js).
