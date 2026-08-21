# audio-tokens -- a waveform becomes a code stack (RVQ)

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · needs an http server (ES modules): `python3 -m http.server 8099`

Interactive page: **residual vector quantization** -- how a continuous waveform
becomes a short sequence of integers a language model can predict, and what the
two knobs (bitrate, sequence length) actually trade. **Anchor**: G (quantization
/ deployment); Family G.

## The mechanism

A neural audio codec closes the gap between "tens of thousands of floating-point
samples per second" and "integers a model predicts" in two moves.

| Move | What happens |
|---|---|
| **encode** | a strided encoder downsamples the waveform to a low **frame rate** -- one *D*-dimensional vector per frame instead of thousands of samples |
| **residual-quantize** | codebook 1 replaces the frame vector with its nearest codeword; the **residual** (what codebook 1 got wrong) is quantized by codebook 2; *that* residual by codebook 3, and so on |
| **decode** | sum the chosen codewords, run synthesis, get audio back |

So one frame becomes a small **stack of integer codes, coarse to fine**. Because
each stage only ever refines what the earlier stages missed, dropping the
deepest codebooks degrades quality **gracefully** instead of destroying it --
that is the whole point of the *residual* structure, and it is what the
transport shows: press play and the reconstruction climbs onto the waveform one
codebook at a time.

```
lat[f] = encoder(frame f) one vector per frame
r ← lat[f]
for s in 1..Q: code[s][f] = argmin_j ‖r − C_s[j]‖ nearest codeword in codebook s
 r ← r − C_s[code[s][f]] quantize the RESIDUAL next
recon[f] = Σ_s C_s[code[s][f]] sum of the chosen codewords
```

## The trade, computed live

```
bitrate = frames/sec × codebooks × log2(codebook size)
```

All three are draggable, and the **staircase** plot is the consequence: each
added codebook steps **right by a constant** `fps × log2(K)` bits/s and **down
by a shrinking amount** of error. Measured on the page at its default view
(two-tone source, seed 3, 75 fps, 64-entry codebooks, 0.8 s clip):

| codebooks | RMSE | bitrate |
|---|---|---|
| 1 | 0.0825 | 0.45 kbit/s |
| 2 | 0.0303 | 0.90 kbit/s |
| 4 | 0.0073 | 1.80 kbit/s |
| 8 | 0.0031 | 3.60 kbit/s |

Eight codebooks cost 8× the bits of one and buy 27× less error -- and the last
four of them, half the entire bitrate, move the error only 0.0073 → 0.0031,
because they are chasing an **encoder floor** of 0.0029 that no depth reaches
past. That floor is drawn as a dashed asymptote.

The second trade is **frame rate against sequence length**. Every frame is a
token position a model has to predict, so `fps × 60 × codebooks` tokens buy one
minute of audio: 36,000 at 75 fps × 8, against a 32,768-token context that holds
roughly 2.7 hours of English text. A minute of audio does not fit where hours of
text do. That gap is the entire reason low-frame-rate codecs exist, and the
frame-rate control walks it -- 6,000 tokens/minute at 12.5 fps × 8.

The frame-rate control also exposes the cost of going low, honestly: the encoder
floor climbs from 0.0015 at 100 fps to 0.0852 at 12.5 fps, because this page's
encoder is a **box filter with linear synthesis, not a trained network**. Pushing
that floor back down at 12.5 frames per second is precisely the work a real
low-frame-rate codec's learned encoder does.

## What is real

Everything on screen is computed in [`page.js`](page.js) at draw time: the
waveform (sine components under an envelope, or a chirp), the encoder, the
codebooks, the code assignment, the reconstruction, and every error number. The
codebooks are fit by a few Lloyd iterations on a **synthetic corpus of other
clips**, never on the clip being coded -- fitting on the clip itself would let a
codebook memorize it and collapse the error for reasons no real codec enjoys.
No audio is played; the waveform and the residual are drawn.

## How this differs from `quantization`

[`quantization`](../quantization/README.md) is the same *shape* of picture --
error against bits -- driven by an entirely different mechanism, and reading the
two together is the point:

| | `quantization` | `audio-tokens` |
|---|---|---|
| what is quantized | each **scalar** weight, independently | each frame's whole **vector**, jointly |
| the code | a position on a uniform grid: `q = round((x−min)/s)` | an **index into a learned codebook** -- the codeword has no numeric order |
| how you spend more bits | widen the grid (more levels per weight) | add another **stage**, quantizing the previous stage's residual |
| the second knob | group size -- how many weights share one scale | frame rate -- how much time one code covers |
| what the output is for | dequantize back to a number a GEMM consumes | a **token sequence a model predicts**, so sequence length is a first-class cost |

Scalar quantization has no residual structure to drop: take bits away and every
weight coarsens at once. RVQ's stages are ordered, so a decoder that reads only
the first *k* codebooks still gets a valid, coarser signal -- which is why a
codec can stream at a bitrate chosen after encoding.

## Interactions

| | |
|---|---|
| **transport** | steps the codebook stages 0 → Q, autoplaying and looping, so the reconstruction visibly converges onto the waveform stage by stage |
| **drag the waveform** | drag ↕ anywhere on the waveform panel to push a bump into the audio -- the encoder, codes, reconstruction and every number recompute under your hand (`reset drags` clears them) |
| **drag the parameters** | codebooks (1–8), codebook size (8–256) and frame rate (12.5–100 fps) are live; the bitrate and the staircase follow |
| **hover a frame** | anywhere on the waveform strip or the code grid: its time span, its full **code stack** (which stages are in use), and that frame's residual RMSE |
| **A/B compare** | 1 codebook against 8, side by side |
| **challenges** | hit an RMSE target under a bitrate cap; fit a minute of audio into a 32,768-token context |

## Render tier

T1 (Canvas2D: waveform + reconstruction, residual, code-stack grid, staircase).

## Wiring

`layout.mount` + controls (`preset`, `fps`, `nq`, `K`, `seed`) + `transport`
(the stage axis, `autoplay` + `loop`) + `onPointer` drag-the-waveform + hover.
Deep links: `?preset` / `?fps` / `?nq` / `?K` / `?seed` / `?step` / `?bump` /
`?hover` (and `?theme`). `?step=N` parks the transport rather than letting
autoplay walk off it, so a captured frame is the one that was asked for.
Source: [`page.js`](page.js).

## Sources

- SoundStream -- <https://arxiv.org/abs/2107.03312> (residual vector quantization in a neural codec)
- EnCodec -- <https://arxiv.org/abs/2210.13438> (the RVQ codec used as an audio tokenizer)
- Moshi / Mimi -- <https://arxiv.org/abs/2410.00037> (12.5 fps + semantic distillation)
