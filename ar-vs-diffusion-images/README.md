# ar-vs-diffusion-images -- a sequence of tokens, or a canvas refined

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: the **two genuinely different mechanisms** for generating an
image, running side by side on one target under one shared compute budget --
**autoregressive** (a sequence of discrete tokens, predicted one step at a time)
and **diffusion** (the whole canvas refined N times). **Anchor**: image
generation, discrete-token vs iterative-refinement.

**This page does not stage a winner, because the argument is not settled.** It
puts the real trade on screen -- the two methods do not even pay in the same
currency -- and cites papers for the empirical claims rather than manufacturing
them from a toy.

## Scope

| This page | Not this page |
|---|---|
| the two mechanisms, side by side, and what each costs | the noise schedule and the training objective -- [`diffusion-noise`](../diffusion-noise/README.md) |
| ordering, tokenization, and the pass/update split | the sampler and its integration error -- [`diffusion-sampler`](../diffusion-sampler/README.md) |
| | conditioning strength -- [`guidance`](../guidance/README.md) |

The diffusion panel here is deliberately reduced to the one property being
compared: **every step touches every region**. The loop itself belongs to the two
pages above and is not re-derived.

## What is on screen

| Element | What it is |
|---|---|
| header thumbnail | the 16×16, 3-channel target both panels are trying to reach, plus the **codebook floor** |
| left panel | the autoregressive run: the image as an 8×8 grid of **discrete token ids**, filled in placement by placement |
| blue outline | the cell being written by the forward pass at the current transport position |
| bar strip under the left panel | `p(token)` over the 12-entry codebook **for the token being placed right now** -- filled bar = chosen, outlined bar = the id the tokenizer itself would have used |
| right panel | the diffusion run: the whole canvas, re-estimated on every step |
| orange cell outlines | how much **each region moved on this step** -- the point being that all of them moved |
| bar strip under the right panel | mean `|Δ|` per step; one bar is one whole-canvas pass |
| bottom bar | the **shared budget**, drawn as the same compute spent two ways -- drag it |

## The mechanisms

### Autoregressive: the image is a token sequence

A codebook of `K = 12` **2×2 patch prototypes** is fitted in-page over the
target's 64 patches (farthest-point init, then Lloyd iterations -- deterministic,
no RNG). That is the tokenizer: the image becomes an 8×8 grid of ids, exactly the
discrete representation a VQ-VAE / VQGAN produces. Tokens are then emitted one at
a time, each conditioned on the ones already emitted -- the same loop a language
model runs over text, with the same KV cache and the same serving machinery.

Two orders ship, and they cost very differently:

| Order | Sequential forward passes | Token-updates | Conditioning for one cell |
|---|---|---|---|
| `raster` | 64 -- one per token | 64 | decoded pixels to the left and above |
| `next-scale` | 4 -- one per **scale** | 85 (1 + 4 + 16 + 64) | the upsampled reconstruction of the previous scale |

Next-scale prediction is [VAR](https://arxiv.org/abs/2404.02905): the model
predicts the whole 1×1 token map, then the whole 2×2, then 4×4, then 8×8. Cells
*within* a scale are conditioned on the coarser scale rather than on each other,
so a scale is one pass -- which is why the sequential-pass count collapses from
64 to 4 while the token count goes *up*.

### Diffusion: the whole canvas at once

Start from a noise sample; refine every position together, N times. There is no
order, no token, and no per-position bar to hover -- one step is one whole-canvas
pass. Cost is `N` sequential passes and `N × 64` token-updates.

## The trade, which is not settled

Three things the page puts on screen rather than asserting:

**1. The cost axes disagree.** Sequential passes are latency-shaped -- what you
cannot parallelise away. Token-updates are work-shaped. At 6 diffusion steps,
raster AR is **1067% of diffusion's 6 sequential passes** (lower is better on
this axis; 100% = parity) and simultaneously **16.7% of diffusion's 384
token-updates** (lower is better; 100% = parity). Both readings are correct.
Which one *is* the cost depends on your batch size, your memory bandwidth,
whether you already run a language model's serving stack -- an AR image model
drops into a KV cache and continuous batching unchanged, while diffusion
parallelises along a different axis.

**2. Ordering is an AR-only problem, and stopping is a diffusion-only one.**
Raster order makes long-range structure hard: a token sees only what has already
been scanned. That is the problem next-scale prediction exists to remove, and it
is a problem diffusion simply does not have, because it has no order at all. The
converse: AR **stops** -- the grid is finite, and past 64 updates it cannot spend
another pass. Diffusion has no natural stopping point, only a step count you
chose. Turn the shared budget up and watch the AR side leave hundreds of
token-updates unspent.

**3. The discrete representation has a floor.** The readout prints the
**codebook floor**: the target re-encoded through the codebook and decoded
straight back. No AR run beats it at any pass count -- that error lives in the
tokenizer, not the sampling loop, which is why tokenizer quality is the lever
that paper after paper reaches for
([arXiv 2310.05737](https://arxiv.org/abs/2310.05737)). The diffusion side has no
equivalent floor *in this toy*; a real latent-diffusion model has an autoencoder
floor of its own.

On quality, the honest position is a citation, not a measurement: diffusion has
historically led on fine texture and photorealism at a matched budget
([arXiv 2105.05233](https://arxiv.org/abs/2105.05233)), while the AR side offers
exact likelihoods, composition with a language model's own token stream, and --
with a strong enough tokenizer -- competitive samples
([arXiv 2310.05737](https://arxiv.org/abs/2310.05737),
[arXiv 2404.02905](https://arxiv.org/abs/2404.02905)). The page does not settle
that and neither column on screen is evidence about it.

## ⚠ The quality numbers here are a TOY

Stated plainly because it would be easy to misread the two error percentages as a
result:

- The AR predictor is a **stand-in for a trained model**, not a trained model.
 The query it hands the codebook is the decoded context blended with what the
 tokenizer itself would have said, at a weight that **falls where the context is
 thin** (few causal taps in raster order; finer scales in next-scale order).
 That is what makes the *ordering* visible on screen. It is not an accuracy
 claim.
- The diffusion side refines toward an **oracle** progressively-less-low-passed
 target, so it has no model error at all.
- Therefore the two `err` figures are **not comparable to each other**
 neither transfers to a trained model. They are there to show what *kind* of
 error each mechanism makes: AR fails by choosing a wrong token (blocky, ordered,
 discrete), diffusion fails by being unconverged (soft, texture last, uniform
 over the canvas).

The **cost columns are the real content**, and those are exact counts.

## Interactions

| Interaction | How |
|---|---|
| step both processes together | transport play / pause / step / scrub; auto-plays and loops. The axis is **normalised progress**, so both start and finish together however differently they are priced -- the absolute pass counts stay on screen |
| drag the shared budget | **drag the bottom bar**. Under `same budget` it re-derives the diffusion step count and shows what the AR side cannot spend |
| drag the diffusion step count | **drag the strip in the right panel header**. With `same budget` on it moves the shared budget too, so the two panels never silently fall out of iso-budget mid-drag |
| change the AR ordering | **click either half of the strip in the left panel header** (raster ⟷ next-scale), or use the widget |
| inspect a token | **hover** any cell of the left panel: which placement wrote it, which forward pass, at which scale, its top-3 candidates with probabilities, and whether it matched the tokenizer's own id |
| inspect a region | **hover** any cell of the right panel: how many steps have touched it (all of them), its latent value vs the target, and which step moved it most |
| inspect the budget | hover the budget bar, the order strip, the step strip, the candidate strip, or the target thumbnail |
| re-allocate fairly | the `same budget` toggle. Off, the readout says the two sides are **not** iso-budget and that any quality difference is confounded by compute |

### URL hooks

Everything is seeded, so one URL replays one exact run.

```
?step=3 pause the transport at progress step 3
?order=scale next-scale prediction instead of raster
?budget=768&fair=1 shared budget, re-allocated to the diffusion side
?fair=0&dsteps=12 unshared: 12 diffusion steps regardless of budget
?temp=0.08 candidate temperature (how peaked p(token) is)
?seed=11 which synthetic image
?hover=420,300 place the cursor (headless stand-in for a hover)
?theme=dark pin the theme for this view
```

## Sources

- Tian, Jiang, Yuan, Peng, Wang, *Visual Autoregressive Modeling: Scalable Image
 Generation via Next-Scale Prediction* -- <https://arxiv.org/abs/2404.02905>.
 The scale-ordered branch of this page.
- Esser, Rombach, Ommer, *Taming Transformers for High-Resolution Image
 Synthesis* -- <https://arxiv.org/abs/2012.09841>. Token images.
- van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning* --
 <https://arxiv.org/abs/1711.00937>. The codebook this page fits in miniature.
- Dhariwal, Nichol, *Diffusion Models Beat GANs on Image Synthesis* --
 <https://arxiv.org/abs/2105.05233>. The photorealism claim, cited rather than
 reproduced.
- Yu, Lezama, Gundavarapu et al., *Language Model Beats Diffusion -- Tokenizer is
 Key to Visual Generation* -- <https://arxiv.org/abs/2310.05737>. The other
 side of the argument, and the reason the codebook floor is on screen.
