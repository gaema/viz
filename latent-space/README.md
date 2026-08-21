# latent-space -- the compression that makes image generation affordable, and its ceiling

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **why a diffusion model can afford to generate an image at
all**, and the **hard ceiling** that affordability buys. The whole sampling loop
runs inside a compressed space produced by an autoencoder -- and whatever that
autoencoder cannot represent is gone before the first denoising step, with no
sampler able to recover it.

## What it shows

A diffusion loop run on pixels touches every pixel on **every** step, and a
transformer denoiser's attention costs the square of the token count. Latent
diffusion pays the pixel-resolution price exactly **twice** -- encode once at the
start, decode once at the end -- and runs the tens of intermediate steps in a
latent instead. The page walks that pipeline on a transport:

| step | what appears |
|---|---|
| 0 | `x` -- the source image at full resolution |
| 1 | encode -- one strided transform per `f×f` block |
| 2 | **truncate** -- everything past channel `C` is discarded, permanently |
| 3 | `z` -- the arena the sampling loop runs in |
| 4 | diffuse -- every step, in here |
| 5 | decode -- one pass back to pixels |
| 6 | **the reconstruction floor** -- the residual with no diffusion at all |

**Everything numeric is computed in-page from the reader's own settings** --
element counts, the compression ratio, the attention-work ratio, the discarded
energy, and a real reconstruction error from an actual encode/decode of the
in-page image. Nothing is quoted.

### The part worth staying for

Step 6 encodes and decodes with **no diffusion at all** and draws
`|x̂ − x|`. Set the sampling error to zero -- a perfect loop -- and the residual
panel does not go blank. The **ceiling bar** splits the total error into the
**irreducible** part (the encoder threw it away) and the **reducible** part (the
sampling loop), and only the second one responds to a better sampler.

Small text and fine high-frequency texture are the classic casualties, so the
default source image contains both: a `3×5`-pixel word, a 1-pixel grating, a
smooth ramp and some hard edges, side by side. At the standard `8×` / 4-channel
setting the text and the grating are the parts that vanish, and they are exactly
what lights up in the residual.

### The trade, under your hand

Dragging the **downsample factor** `f` or the **channel count** `C` moves the
compute saving and the reconstruction error in **opposite directions**, live.
The trade panel plots every `(f, C)` the reader can reach as one dot -- x axis
compression, y axis floor -- and you can drag along the cloud to jump between
operating points. The degenerate corner is instructive: keep every channel of a
block and the transform is lossless, the floor is zero, and the compression is
`1×`. The saving *is* the loss.

### The toy encoder -- stated plainly

There is **no trained VAE here and the page does not claim one**. The encoder is
an orthonormal **block DCT-II** over `f×f` blocks, keeping the `C`
lowest-frequency coefficients per block in zig-zag order; the decoder is the
inverse transform with the discarded coefficients set to zero. Because the
transform is orthonormal, the discarded energy **is** the squared reconstruction
error -- the floor is a sum you can point at, not an estimate, and hovering a
basis cell tells you the exact share of the image it carries.

A trained autoencoder differs in every way that matters for the numbers: it
learns its basis rather than fixing it, it is not orthonormal, and its
adversarial and perceptual objectives deliberately trade measured error for
perceived quality -- buying back detail this toy cannot. **Read the shape of the
trade here, never the exact numbers.**

### Error is reported against a named baseline

The floor is given as RMSE and as a **percent of the `8×`/4-channel reference
floor**, recomputed in-page on the same source image (**lower is better; 100% =
parity**). At `f = 8, C = 8` on the default scene that reads `84.2%` of the
reference; at `f = 16, C = 2` it reads `121%`.

## This recipe is contested -- and the page says so

`8×` down with 4 channels came from the original latent-diffusion paper
stuck for years, but it is not settled, and the page puts the disagreement
on screen rather than picking a side:

| position | argument |
|---|---|
| 🟡 the latent is **too thin** | The rectified-flow scaling work raised the autoencoder's channel count and reported better reconstruction *and* better samples. |
| 🟡 the latent is the **wrong kind** | Representation-autoencoder work replaces the VAE with a pretrained representation encoder plus a trained decoder, arguing a purely reconstruction-trained, low-dimensional latent limits generative quality; related alignment work argues representation quality is the training bottleneck. |
| 🟡 the latent is **not compressed enough** | Deep-compression autoencoders push spatial compression far past `8×` and still generate well, trading the other way. |
| ⚪ the metric is **not perception** | A trained autoencoder spends adversarial and perceptual loss precisely on what RMSE does not measure, so a floor like this one over- and under-states different failures. |

Nobody has settled where the knee is. That is the honest state of it.

## Sources

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* -- <https://arxiv.org/abs/2112.10752>
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* -- <https://arxiv.org/abs/2403.03206>
- Yu et al., *Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think* -- <https://arxiv.org/abs/2410.06940>
- Chen et al., *Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models* -- <https://arxiv.org/abs/2410.10733>
- *Diffusion Transformers with Representation Autoencoders* -- <https://arxiv.org/abs/2510.11690>
- *Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders* -- <https://arxiv.org/abs/2601.16208>

## Not this page

The noising/denoising objective itself is [`diffusion-noise`](../diffusion-noise/index.html);
step counts and sampler choice are [`diffusion-sampler`](../diffusion-sampler/index.html);
conditioning strength is [`guidance`](../guidance/index.html). None of them runs
here -- this page is the arena those loops run inside, and the price of the door.
The error-versus-compression trade in weights rather than activations is
[`quantization`](../quantization/index.html).

## Interactions

| interaction | how |
|---|---|
| 🟢 transport | encode → diffuse → decode as 7 steps, autoplay + loop, `?step=N` |
| 🟢 direct manipulation | drag `f` and `C` and watch saving and error split; **drag the trade cloud** to jump to any `(f, C)` |
| 🟢 paint | drag on `x` with a brush (fine 1-px checker / bright dot / erase) and watch what survives the round trip |
| 🟢 hover | any latent cell reports its channel, its value, and the exact source block it covers; any basis cell reports its energy share and whether it is kept |
| 🟢 A/B compare | `C = 4` against `C = 16` at the same downsample |
| 🟢 challenges | make the floor vanish; turn the sampler off and watch the residual survive; find a setting past `64×` |

## Render tier

T1 (Canvas2D): five heatmap panels, the ceiling bar, and the trade scatter.

## Wiring

`layout.mount` + controls (`scene`, `down`, `chan`, `noise`, `brush`, `seed`)
+ `animate` (the loop shimmer on `z`) + `onPointer` for painting and for
dragging the trade cloud + hover. URL hooks: `?step` `?scene` `?down` `?chan`
`?noise` `?seed` `?brush` `?stamp=r,c;r,c` `?paint=i,v;i,v` `?cell=panel,r,c`
`?hover=x,y` `?ch=N` `?compare=1` `?play=1`. Source: [`page.js`](page.js).
