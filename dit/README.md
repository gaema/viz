# dit -- a transformer over latent patches, conditioned by modulating the norm

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page for the architecture that replaced the U-Net in image and video
generation: a **DiT** patchifies the noisy **latent** into tokens and runs an
ordinary transformer over them. **Anchor**: the diffusion backbone.

**Scope.** This page owns the **architecture** only. Its three neighbours own the
rest of the diffusion story and are not re-taught here:
[`diffusion-noise`](../diffusion-noise/README.md) (the noising schedule and what
the network is trained to predict), [`diffusion-sampler`](../diffusion-sampler/README.md)
(the integrator and the step count), and [`guidance`](../guidance/README.md)
(the conditioning knob, `w`).

## The one idea

Patchifying a latent and running a transformer over it is unsurprising. The part
that is worth drawing is **how the conditioning gets in**.

| Backbone | How the timestep / class / text conditioning enters |
|---|---|
| U-Net | concatenated onto the feature map, or read through cross-attention |
| DiT (in-context) | the conditioning is appended as extra tokens |
| DiT (cross-attention) | a cross-attention sublayer per block |
| **DiT (adaLN-Zero)** | **the conditioning becomes the normalisation's scale, shift and gate** |

The last one is what the paper found worked best, and it is the page's subject.
One linear layer per block turns the conditioning vector `c` into **six** vectors:

```
c = timestep-embed(t) + class/text-embed (one vector per SAMPLE)
γ₁ β₁ α₁ γ₂ β₂ α₂ = Linear(SiLU(c)) (six vectors per BLOCK)

h = LayerNorm(x)·(1 + γ₁) + β₁ modulate
a = Attention(h) attend
x = x + α₁·a GATE, then residual
h = LayerNorm(x)·(1 + γ₂) + β₂ modulate
m = MLP(h)
out = x + α₂·m GATE, then residual
```

Note there is no `γ`/`β` learned per-dimension inside the norm any more. The norm
is un-affine, and the affine parameters are *produced from the conditioning* --
which is the whole trick. The conditioning is no longer something the token
attends to; it is something that reshapes the token's own statistics.

## Why the ZERO in adaLN-Zero

`α₁` and `α₂` are **gates**: they multiply each branch before it is added back.
The layer that emits all six is initialised to **exactly zero** -- weight
bias -- so at step 0 of training `α₁ = α₂ = 0`
```
out = x + 0·a + 0·m = x
```

Every block is an **identity**. A 28-block DiT therefore starts life as a clean
residual pass-through, and each block learns its way in from zero rather than
having to unlearn a random perturbation it was born with. That is the detail the
page is built to make visible: drag **training progress** to 0 and watch all six
vectors go to `0.00`, the per-token `‖Δ‖` map go flat, and the "token in"
"gate + add" vector norms become the same number. Both branches are still
computed and drawn -- they simply contribute nothing.

## What is on screen

| Element | What it is |
|---|---|
| the diverging grid, top left | the noisy latent, with the patch lattice drawn over it; the blue box is the token being followed |
| the small sequential grid beside it | `‖out − in‖` per token, i.e. how much this one block moved each token. **Flat at zero-init** |
| conditioning panel | three **draggable** tracks (timestep, conditioning strength, training progress), then the sin-cos t-embed, the class embed, and their sum `c` |
| six labelled strips | `γ₁ β₁ α₁ γ₂ β₂ α₂`, live, with the two **gates** outlined |
| the seven-box row | one block, stage by stage, with the tracked token's feature vector under each stage and its norm |
| the two dashed green arcs | the residual paths -- drawn **solid** when the gates are zero, because then they are the only path |
| bottom left | the cost, computed from your settings |
| bottom right | the MMDiT comparison |

Nothing is staged. The latent, the patch projection, the frozen 2-D sin-cos
position embedding, the sinusoidal timestep embedding, the adaLN projection,
LayerNorm, single-head attention over **every** token, the GELU MLP and both
gated residual adds are all arithmetic the page does in-frame.

## The trade: patch size is the whole cost story

`N = (G/p)²`. Halving `p` **quadruples** the tokens, so the `N²` attention term
goes up **sixteenfold**. The page computes this live from your grid, patch size
and `d_model` -- no FLOP number in the source is hard-coded:

```
attention 4·N²·d the quadratic term
qkv + out 8·N·d²
MLP 16·N·d²
adaLN 12·d² once per SAMPLE, not per token
```

Read it honestly: at a realistic `d_model` the quadratic term is a *minority* of
a block's FLOPs until the token count gets large -- at `N = 64, d = 1152` it is
under 1%, at `N = 256` it is a few percent. What halving `p` really buys is
finer spatial detail, and what it charges is a term that grows as the square.
The panel prints both directions (`p → p/2` and `p → 2p`) as a **percent of the
current setting**, so the direction is never ambiguous.

**Second trade, stated honestly: adaLN is not free.** That per-block projection
of the conditioning vector is `6·d² + 6·d` real parameters, against `12·d²` for
the block's own attention + MLP -- so **adaLN is about a third of the block's
parameter count**. Its *FLOP* cost is negligible (well under 1% of the block)
because it runs once per sample rather than once per token, which is exactly why
it is a cheap idea in compute and an expensive one in weights. Both numbers are
in the panel and both move with `d_model`.

## MMDiT, in one panel

MMDiT (Stable Diffusion 3) keeps the modulation idea and adds a second stream:
text and image tokens each get their **own** qkv / MLP / adaLN weights (roughly
double the block parameters), but the two token sets are **concatenated for one
joint attention**, so they attend to each other directly instead of through a
pooled conditioning vector. Attention pairs become `(N_img + N_txt)²`. That is
the whole of what this page says about it -- one comparative panel, not a second
lesson.

## Interactions

| Interaction | How |
|---|---|
| step through one block | transport play / pause / step / scrub over the seven stages; auto-plays and loops |
| **drag the conditioning** | drag the **timestep** or **conditioning strength** track on the canvas -- all six modulation vectors move under your hand |
| **zero the gate** | drag **training progress** to 0, or press the button; the block becomes an exact identity and the readout prints `max‖out−in‖ = 0.000000` |
| **change the patch size** | the `p` select -- token count and attention cost move together, priced live |
| follow a different token | click any patch in the latent |
| hover to inspect | hover any patch (its index, its `p×p` source cells, its projected value), any modulation cell (its value, which row of the adaLN Linear produced it, and what it does), any stage cell, or any of the three tracks |
| A/B compare | wired to `init`: gates-at-zero beside gates-open, same seed |
| challenge mode | two goals: make the block an identity; push the quadratic term past 10% of the block |

### URL hooks

Every control is mirrored into the query string, so the copy-link button
reproduces the exact frame.

```
?step=3 pause the transport at stage 3
?init=0 the adaLN-Zero initialisation (block ≡ identity)
?p=1&G=16 patch size and latent grid -> N = 256 tokens
?t=850&cscale=2.4 timestep and conditioning strength
?cls=2&seed=7&tok=40 class, seed, and which token is followed
?dm=1152 the d_model the cost panel prices
?drag=t,850 headless stand-in for dragging a canvas track
?hover=520,60 headless stand-in for a hover
?compare=1 ?ch=1 open the A/B panes / a challenge
```

## Render tier

T1 (Canvas2D). Two small heatmaps, a few dozen labelled cells and a box diagram;
there is no large grid that would justify WebGL2.

## Sources

- Peebles & Xie, *Scalable Diffusion Models with Transformers* --
 <https://arxiv.org/abs/2212.09748>. The DiT block, the four conditioning
 variants, adaLN-Zero, and the patch-size scaling study.
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image
 Synthesis* -- <https://arxiv.org/abs/2403.03206>. MMDiT: separate weights per
 modality, joint attention.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* --
 <https://arxiv.org/abs/1709.07871>. The earlier feature-wise scale-and-shift
 conditioning that adaLN generalises.
- Bachlechner et al., *ReZero is All You Need* -- <https://arxiv.org/abs/2003.04887>.
 The zero-initialised residual gate as a way to make deep stacks trainable.
