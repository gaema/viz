# video-time — time is just another axis you tokenize

Time is not a new mechanism in a video model. It is a **third patching axis**,
and once you add it the arithmetic of attention becomes brutal in a way that is
easy to state and hard to feel — so this page makes you feel it by making you
drag it.

## What the page shows

Three stages, stepped through by the transport:

1. **Raw frames.** A clip is a pixel volume with a time axis: `W × H × 3 × F`.
2. **A causal 3-D autoencoder** compresses it in space *and* in time. A common
 published recipe divides each spatial axis by 8 and the time axis by 4. The
 **causal** part is why the first frame is handled on its own, so
 `F` original frames become `1 + floor((F-1)/t)` latent frames — and a single
 image (`F = 1`) is still representable by the same model.
3. **Patchify the latent and attend.** A `p × p` patch on each latent frame
 gives the sequence of **space-time tokens**, and self-attention scores every
 pair of them. Its cost is the **square** of the token count.

That square is the whole page. A few seconds of ordinary video is already a
sequence of hundreds of thousands of tokens *before* any compression, which is
why the 3-D autoencoder is not an optimisation bolted on afterwards — it is
what makes the model possible at all.

Every figure is counted live from the controls. Nothing is a quoted
configuration of any particular product, and nothing is a measurement: the page
counts tokens and score entries, which is exact arithmetic, and says so.

## The mitigations, priced honestly

The second panel puts the three attention shapes side by side, costs them
against each other at the current settings, and states what each one gives up.
Switching between them changes the **shape** of the cost curve, not just its
height.

| Shape | Cost | Gives up |
|---|---|---|
| 🟢 full 3-D | `N²` | nothing — exact, every space-time pair is scored. The cost is the square. |
| 🟡 factorised, spatial then temporal | `N·(S+T)` | a single layer cannot relate two tokens that differ in **both** space and time; that has to route through an intermediate position over two passes. |
| 🟡 windowed / local 3-D | `N·k` for a `k`-token neighbourhood | anything outside the window is invisible to that layer; long-range motion needs depth or extra global tokens. |

`N` = total space-time tokens, `S` = spatial tokens per latent frame,
`T` = latent frames. Costs are score entries per head per layer.

A comparison bar holds the same figures against a **128 K-token text context**
attending fully, because "large" only means something next to something else.
Costs are reported as a percent of a named baseline; attention cost is a
lower-is-better quantity, so `>100%` there is worse.

## Interactions

| Interaction | How |
|---|---|
| **Transport** | play / pause / step / scrub the four pipeline stages; autoplays and loops. `?step=N` |
| **Direct manipulation — every control** | drag resolution, frame rate, clip length or either compression factor and the raw-pixel, token and attention figures all move together |
| **Direct manipulation — the clip length** | drag the marker along the cost curve |
| **Direct manipulation — a space-time patch** | drag on the frame plate to move the selected patch and read which pixels, across which frames, it swallows into one token. `?patch=r,c` |
| **Attention-shape switch** | the control, or click a row of the mitigations table |
| **Hover** | every ladder row, bar, curve, matrix and table row carries the arithmetic behind its figure. `?hover=x,y` |
| **Challenge mode** | push one clip past a 128 K text context, then bring it back without touching resolution, frame rate or duration |

All controls are mirrored into the query string, so any view is a link.

## Siblings — read those for the parts this page does not re-teach

- [`patch-embedding`](../patch-embedding/README.md) owns **2-D patching**: how a
 still image becomes a token sequence, and why that projection is exactly a
 convolution with kernel = stride = patch size. This page assumes it and adds
 the third axis.
- the sibling `latent-space` page owns the **compression ceiling**:
 how much an autoencoder can throw away before the reconstruction stops being
 the thing you started with. This page treats the compression factors as knobs
 and does not ask what they cost in fidelity.
- [`vram-budget`](../vram-budget/README.md) is the same style of live budget
 arithmetic for inference memory rather than attention cost.

## Sources

- *Video generation models as world simulators* — space-time patches:
 <https://openai.com/index/video-generation-models-as-world-simulators/>
- *CogVideoX*, arXiv:2408.06072 — 3-D causal VAE plus an expert transformer:
 <https://arxiv.org/abs/2408.06072>

## Running it

Serve the tree over http (ES module imports are blocked on `file://`):

```sh
python3 -m http.server 8099
# then open http://localhost:8099/video-time/
```
